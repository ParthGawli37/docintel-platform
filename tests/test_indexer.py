from pathlib import Path

import pytest

from docintel.core.models import Chunk, EmbeddedChunk
from docintel.indexing.indexer import IncrementalIndexer
from docintel.ingestion.chunking.recursive_chunker import RecursiveChunker
from docintel.ingestion.loaders import bootstrap_loaders
from docintel.ingestion.loaders.base import registry
from docintel.ingestion.pipeline import IngestionPipeline
from docintel.storage.hash_registry import SqliteHashRegistry

FIXTURES = Path(__file__).parent / "fixtures"
KB_ID = "kb-index-test"


@pytest.fixture(autouse=True, scope="module")
def _bootstrap():
    bootstrap_loaders()


class _FakeEmbedder:
    model_id = "fake-model"
    dimensions = 4

    def __init__(self) -> None:
        self.embed_calls = 0

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.embed_calls += 1
        return [
            EmbeddedChunk(chunk=c, vector=[1.0, 0.0, 0.0, 0.0], model_id=self.model_id)
            for c in chunks
        ]

    async def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list[EmbeddedChunk] = []

    async def ensure_collection(self, collection, dimensions):
        pass

    async def drop_collection(self, collection):
        pass

    async def upsert(self, collection, embedded_chunks):
        self.upserted.extend(embedded_chunks)

    async def delete_by_document_id(self, collection, document_id):
        pass

    async def search(self, collection, query_vector, top_k, filters=None):
        return []

    async def get_existing_content_hashes(self, collection):
        return set()

    async def get_all_chunks(self, collection):
        return [ec.chunk for ec in self.upserted]


def _make_indexer(tmp_path, embedder=None, vector_store=None):
    loader = registry.resolve(FIXTURES / "sample.txt")
    pipeline = IngestionPipeline(chunker=RecursiveChunker(chunk_size_tokens=100, overlap_tokens=10))
    hash_registry = SqliteHashRegistry(tmp_path / "hashes.sqlite")
    return IncrementalIndexer(
        loader=loader,
        pipeline=pipeline,
        embedder=embedder or _FakeEmbedder(),
        vector_store=vector_store or _FakeVectorStore(),
        hash_registry=hash_registry,
    ), hash_registry


@pytest.mark.asyncio
async def test_indexer_indexes_new_source(tmp_path):
    embedder = _FakeEmbedder()
    vector_store = _FakeVectorStore()
    indexer, _ = _make_indexer(tmp_path, embedder, vector_store)

    result = await indexer.index_source(FIXTURES / "sample.txt", KB_ID)

    assert result.skipped is False
    assert result.chunk_count > 0
    assert result.error is None
    assert embedder.embed_calls == 1
    assert len(vector_store.upserted) == result.chunk_count


@pytest.mark.asyncio
async def test_indexer_skips_unchanged_source_on_second_run(tmp_path):
    embedder = _FakeEmbedder()
    vector_store = _FakeVectorStore()
    indexer, _ = _make_indexer(tmp_path, embedder, vector_store)

    first = await indexer.index_source(FIXTURES / "sample.txt", KB_ID)
    second = await indexer.index_source(FIXTURES / "sample.txt", KB_ID)

    assert first.skipped is False
    assert second.skipped is True
    assert embedder.embed_calls == 1  # only called once, not on the skip


@pytest.mark.asyncio
async def test_indexer_reindexes_when_content_changes(tmp_path):
    embedder = _FakeEmbedder()
    vector_store = _FakeVectorStore()
    indexer, hash_registry = _make_indexer(tmp_path, embedder, vector_store)

    await indexer.index_source(FIXTURES / "sample.txt", KB_ID)
    # Simulate a content change by clearing the registered hash for this source.
    await hash_registry.set_hash(KB_ID, str(FIXTURES / "sample.txt"), "deliberately-different-hash")

    result = await indexer.index_source(FIXTURES / "sample.txt", KB_ID)
    assert result.skipped is False
    assert embedder.embed_calls == 2


@pytest.mark.asyncio
async def test_indexer_handles_loader_failure_gracefully(tmp_path):
    class _FailingLoader:
        supported_extensions = (".txt",)
        handles_urls = False

        def can_load(self, source):
            return True

        async def load(self, source, knowledge_base_id):
            raise RuntimeError("simulated load failure")

    pipeline = IngestionPipeline(chunker=RecursiveChunker(chunk_size_tokens=100, overlap_tokens=10))
    hash_registry = SqliteHashRegistry(tmp_path / "hashes.sqlite")
    indexer = IncrementalIndexer(
        loader=_FailingLoader(),
        pipeline=pipeline,
        embedder=_FakeEmbedder(),
        vector_store=_FakeVectorStore(),
        hash_registry=hash_registry,
    )
    result = await indexer.index_source("broken.txt", KB_ID)
    assert result.error is not None
    assert "simulated load failure" in result.error


@pytest.mark.asyncio
async def test_indexer_calls_invalidate_callbacks_after_write(tmp_path):
    indexer, _ = _make_indexer(tmp_path)
    invalidated_kbs = []
    indexer.add_invalidate_callback(lambda kb_id: invalidated_kbs.append(kb_id))

    await indexer.index_source(FIXTURES / "sample.txt", KB_ID)
    assert invalidated_kbs == [KB_ID]


@pytest.mark.asyncio
async def test_indexer_does_not_invalidate_on_skip(tmp_path):
    indexer, _ = _make_indexer(tmp_path)
    invalidated_kbs = []
    indexer.add_invalidate_callback(lambda kb_id: invalidated_kbs.append(kb_id))

    await indexer.index_source(FIXTURES / "sample.txt", KB_ID)
    invalidated_kbs.clear()
    await indexer.index_source(FIXTURES / "sample.txt", KB_ID)  # unchanged -> skip
    assert invalidated_kbs == []


@pytest.mark.asyncio
async def test_index_batch_aggregates_results(tmp_path):
    embedder = _FakeEmbedder()
    vector_store = _FakeVectorStore()
    indexer, _ = _make_indexer(tmp_path, embedder, vector_store)

    batch = await indexer.index_batch(
        [FIXTURES / "sample.txt", FIXTURES / "sample.md"], KB_ID
    )
    assert batch.indexed_count == 2
    assert batch.skipped_count == 0
    assert batch.failed_count == 0

    # Second pass over the same sources should all be skipped.
    batch2 = await indexer.index_batch(
        [FIXTURES / "sample.txt", FIXTURES / "sample.md"], KB_ID
    )
    assert batch2.skipped_count == 2
