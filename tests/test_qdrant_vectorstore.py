import pytest
from qdrant_client import AsyncQdrantClient

from docintel.core.interfaces import VectorStore
from docintel.core.models import Chunk, DocumentMetadata, EmbeddedChunk, SourceType
from docintel.vectorstore.qdrant_store import QdrantVectorStore

DIMENSIONS = 4


def _embedded_chunk(
    content: str,
    content_hash: str,
    vector: list[float],
    doc_id: str = "doc-1",
    source_uri: str = "x.txt",
) -> EmbeddedChunk:
    chunk = Chunk(
        document_id=doc_id,
        content=content,
        chunk_index=0,
        metadata=DocumentMetadata(
            source_uri=source_uri,
            source_type=SourceType.TXT,
            content_hash=content_hash,
            knowledge_base_id="kb-1",
            title="Test Doc",
        ),
    )
    return EmbeddedChunk(chunk=chunk, vector=vector, model_id="fake-model")


@pytest.fixture
async def store():
    client = AsyncQdrantClient(location=":memory:")
    yield QdrantVectorStore(client)
    await client.close()


def test_store_satisfies_protocol():
    client = AsyncQdrantClient(location=":memory:")
    assert isinstance(QdrantVectorStore(client), VectorStore)


@pytest.mark.asyncio
async def test_ensure_collection_is_idempotent(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.ensure_collection("kb-1", DIMENSIONS)  # second call must not raise


@pytest.mark.asyncio
async def test_upsert_and_search_returns_nearest_match(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.upsert(
        "kb-1",
        [
            _embedded_chunk("about cats", "hash-cat", [1.0, 0.0, 0.0, 0.0]),
            _embedded_chunk("about finance", "hash-fin", [0.0, 1.0, 0.0, 0.0]),
        ],
    )

    results = await store.search("kb-1", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.content == "about cats"
    assert results[0].score > 0.9


@pytest.mark.asyncio
async def test_search_respects_top_k(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.upsert(
        "kb-1",
        [
            _embedded_chunk("a", "hash-a", [1.0, 0.0, 0.0, 0.0]),
            _embedded_chunk("b", "hash-b", [0.9, 0.1, 0.0, 0.0]),
            _embedded_chunk("c", "hash-c", [0.0, 0.0, 1.0, 0.0]),
        ],
    )
    results = await store.search("kb-1", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_round_trips_full_metadata(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.upsert("kb-1", [_embedded_chunk("content here", "hash-x", [1.0, 0.0, 0.0, 0.0])])
    results = await store.search("kb-1", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=1)
    metadata = results[0].chunk.metadata
    assert metadata.source_uri == "x.txt"
    assert metadata.source_type is SourceType.TXT
    assert metadata.title == "Test Doc"
    assert metadata.content_hash == "hash-x"
    assert metadata.knowledge_base_id == "kb-1"


@pytest.mark.asyncio
async def test_delete_by_document_id_removes_matching_points(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.upsert(
        "kb-1",
        [
            _embedded_chunk("keep me", "hash-keep", [1.0, 0.0, 0.0, 0.0], doc_id="doc-keep"),
            _embedded_chunk("delete me", "hash-del", [0.0, 1.0, 0.0, 0.0], doc_id="doc-delete"),
        ],
    )
    await store.delete_by_document_id("kb-1", "doc-delete")

    results = await store.search("kb-1", query_vector=[0.0, 1.0, 0.0, 0.0], top_k=5)
    assert all(r.chunk.document_id != "doc-delete" for r in results)


@pytest.mark.asyncio
async def test_delete_by_source_uri_removes_only_matching_source(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.upsert(
        "kb-1",
        [
            _embedded_chunk("old a", "hash-a1", [1.0, 0.0, 0.0, 0.0], doc_id="doc-a", source_uri="a.txt"),
            _embedded_chunk("old a 2", "hash-a2", [0.9, 0.1, 0.0, 0.0], doc_id="doc-a", source_uri="a.txt"),
            _embedded_chunk("keep b", "hash-b", [0.0, 1.0, 0.0, 0.0], doc_id="doc-b", source_uri="b.txt"),
        ],
    )

    removed = await store.delete_by_source_uri("kb-1", "a.txt")

    assert removed == 2
    results = await store.search("kb-1", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert all(r.chunk.metadata.source_uri != "a.txt" for r in results)
    assert any(r.chunk.metadata.source_uri == "b.txt" for r in results)


@pytest.mark.asyncio
async def test_delete_by_source_uri_returns_zero_for_missing_source(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    assert await store.delete_by_source_uri("kb-1", "missing.txt") == 0


@pytest.mark.asyncio
async def test_get_existing_content_hashes_empty_for_new_collection(store):
    hashes = await store.get_existing_content_hashes("nonexistent-kb")
    assert hashes == set()


@pytest.mark.asyncio
async def test_get_existing_content_hashes_returns_all_indexed_hashes(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.upsert(
        "kb-1",
        [
            _embedded_chunk("a", "hash-a", [1.0, 0.0, 0.0, 0.0]),
            _embedded_chunk("b", "hash-b", [0.0, 1.0, 0.0, 0.0]),
        ],
    )
    hashes = await store.get_existing_content_hashes("kb-1")
    assert hashes == {"hash-a", "hash-b"}


@pytest.mark.asyncio
async def test_search_with_filters(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.upsert(
        "kb-1",
        [
            _embedded_chunk("doc a chunk", "hash-a", [1.0, 0.0, 0.0, 0.0], doc_id="doc-a"),
            _embedded_chunk("doc b chunk", "hash-b", [1.0, 0.0, 0.0, 0.0], doc_id="doc-b"),
        ],
    )
    results = await store.search(
        "kb-1", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=5, filters={"document_id": "doc-a"}
    )
    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-a"


@pytest.mark.asyncio
async def test_collections_are_isolated_between_knowledge_bases(store):
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.ensure_collection("kb-2", DIMENSIONS)
    await store.upsert("kb-1", [_embedded_chunk("only in kb1", "hash-1", [1.0, 0.0, 0.0, 0.0])])

    kb1_hashes = await store.get_existing_content_hashes("kb-1")
    kb2_hashes = await store.get_existing_content_hashes("kb-2")
    assert kb1_hashes == {"hash-1"}
    assert kb2_hashes == set()
