from dataclasses import dataclass

import pytest

from docintel.core.interfaces import Embedder
from docintel.core.models import Chunk, DocumentMetadata, SourceType
from docintel.embeddings.cached_embedder import CachedEmbedder
from docintel.embeddings.nvidia_embedder import NvidiaEmbedder
from docintel.storage.cache_store import SqliteEmbeddingCache


def _chunk(content: str, content_hash: str, chunk_index: int = 0) -> Chunk:
    return Chunk(
        document_id="doc-1",
        content=content,
        chunk_index=chunk_index,
        metadata=DocumentMetadata(
            source_uri="x.txt",
            source_type=SourceType.TXT,
            content_hash=content_hash,
            knowledge_base_id="kb-1",
        ),
    )


# ---------------------------------------------------------------------------
# NvidiaEmbedder -- against a fake client shaped like the real OpenAI SDK
# response (openai.types.CreateEmbeddingResponse / Embedding), since this
# sandbox cannot reach api.nvidia.com for a live call.
# ---------------------------------------------------------------------------


@dataclass
class _FakeEmbeddingItem:
    embedding: list[float]
    index: int


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]


class _FakeEmbeddingsResource:
    def __init__(self, responder):
        self._responder = responder
        self.calls: list[list[str]] = []

    async def create(self, model: str, input: list[str]):
        self.calls.append(list(input))
        return self._responder(model, input)


class _FakeAsyncOpenAI:
    def __init__(self, responder):
        self.embeddings = _FakeEmbeddingsResource(responder)


class _FakeSettings:
    nvidia_embedding_model = "fake/embed-model"
    nvidia_embedding_dimensions = 4
    nvidia_api_key = "test-key"
    nvidia_api_base_url = "https://example.invalid/v1"


def _reversed_order_responder(model: str, texts: list[str]):
    """Returns embeddings out of positional order to prove index-based sorting works."""
    items = [
        _FakeEmbeddingItem(embedding=[float(i)] * 4, index=i) for i in range(len(texts))
    ]
    return _FakeEmbeddingResponse(data=list(reversed(items)))


@pytest.mark.asyncio
async def test_nvidia_embedder_satisfies_protocol():
    client = _FakeAsyncOpenAI(_reversed_order_responder)
    embedder = NvidiaEmbedder(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]
    assert isinstance(embedder, Embedder)


@pytest.mark.asyncio
async def test_nvidia_embedder_reorders_response_by_index():
    client = _FakeAsyncOpenAI(_reversed_order_responder)
    embedder = NvidiaEmbedder(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]

    chunks = [_chunk(f"text-{i}", f"hash-{i}", i) for i in range(3)]
    embedded = await embedder.embed_chunks(chunks)

    assert [e.vector for e in embedded] == [[0.0] * 4, [1.0] * 4, [2.0] * 4]
    assert all(e.model_id == "fake/embed-model" for e in embedded)


@pytest.mark.asyncio
async def test_nvidia_embedder_batches_requests():
    client = _FakeAsyncOpenAI(_reversed_order_responder)
    embedder = NvidiaEmbedder(settings=_FakeSettings(), client=client, batch_size=2)  # type: ignore[arg-type]

    chunks = [_chunk(f"text-{i}", f"hash-{i}", i) for i in range(5)]
    await embedder.embed_chunks(chunks)

    # 5 chunks with batch_size=2 -> 3 API calls (2, 2, 1)
    assert len(client.embeddings.calls) == 3
    assert [len(c) for c in client.embeddings.calls] == [2, 2, 1]


@pytest.mark.asyncio
async def test_nvidia_embedder_embed_query():
    client = _FakeAsyncOpenAI(_reversed_order_responder)
    embedder = NvidiaEmbedder(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]
    vector = await embedder.embed_query("hello")
    assert vector == [0.0] * 4


@pytest.mark.asyncio
async def test_nvidia_embedder_empty_chunks_returns_empty_without_api_call():
    client = _FakeAsyncOpenAI(_reversed_order_responder)
    embedder = NvidiaEmbedder(settings=_FakeSettings(), client=client)  # type: ignore[arg-type]
    result = await embedder.embed_chunks([])
    assert result == []
    assert client.embeddings.calls == []


# ---------------------------------------------------------------------------
# CachedEmbedder -- against a real SqliteEmbeddingCache, with a fake inner
# Embedder that records how many times it was actually called.
# ---------------------------------------------------------------------------


class _CountingFakeEmbedder:
    model_id = "fake-model"
    dimensions = 3

    def __init__(self) -> None:
        self.embed_chunks_calls: list[list[str]] = []

    async def embed_chunks(self, chunks: list[Chunk]):
        from docintel.core.models import EmbeddedChunk

        self.embed_chunks_calls.append([c.id for c in chunks])
        return [
            EmbeddedChunk(chunk=c, vector=[1.0, 2.0, 3.0], model_id=self.model_id)
            for c in chunks
        ]

    async def embed_query(self, query: str) -> list[float]:
        return [9.0, 9.0, 9.0]


@pytest.mark.asyncio
async def test_cached_embedder_satisfies_protocol(tmp_path):
    inner = _CountingFakeEmbedder()
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite")
    cached = CachedEmbedder(inner, cache)
    assert isinstance(cached, Embedder)


@pytest.mark.asyncio
async def test_cached_embedder_calls_inner_on_first_request(tmp_path):
    inner = _CountingFakeEmbedder()
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite")
    cached = CachedEmbedder(inner, cache)

    chunks = [_chunk("hello", "hash-1")]
    result = await cached.embed_chunks(chunks)

    assert len(inner.embed_chunks_calls) == 1
    assert result[0].vector == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_cached_embedder_skips_inner_on_second_request_same_hash(tmp_path):
    inner = _CountingFakeEmbedder()
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite")
    cached = CachedEmbedder(inner, cache)

    await cached.embed_chunks([_chunk("hello", "hash-1")])
    # Different chunk object/id, SAME content_hash -- should hit cache.
    result2 = await cached.embed_chunks([_chunk("hello again", "hash-1", chunk_index=1)])

    assert len(inner.embed_chunks_calls) == 1  # inner never called a second time
    assert result2[0].vector == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_cached_embedder_mixed_hits_and_misses_preserves_order(tmp_path):
    inner = _CountingFakeEmbedder()
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite")
    cached = CachedEmbedder(inner, cache)

    await cached.embed_chunks([_chunk("a", "hash-a")])  # populate cache for hash-a

    chunks = [
        _chunk("a-again", "hash-a", chunk_index=0),  # cache hit
        _chunk("b", "hash-b", chunk_index=1),         # cache miss
    ]
    result = await cached.embed_chunks(chunks)

    assert [r.chunk.id for r in result] == [chunks[0].id, chunks[1].id]  # order preserved
    assert inner.embed_chunks_calls[-1] == [chunks[1].id]  # only the miss went to inner on this call
    assert len(inner.embed_chunks_calls) == 2  # one from setup, one from the miss above


@pytest.mark.asyncio
async def test_cached_embedder_embed_query_not_cached_delegates_every_time(tmp_path):
    inner = _CountingFakeEmbedder()
    cache = SqliteEmbeddingCache(tmp_path / "cache.sqlite")
    cached = CachedEmbedder(inner, cache)

    v1 = await cached.embed_query("q")
    v2 = await cached.embed_query("q")
    assert v1 == v2 == [9.0, 9.0, 9.0]
