import pytest
from qdrant_client import AsyncQdrantClient

from docintel.core.interfaces import Reranker, SparseRetriever
from docintel.core.models import Chunk, DocumentMetadata, EmbeddedChunk, SearchResult, SourceType
from docintel.retrieval._tokenize import tokenize
from docintel.retrieval.hybrid_retriever import HybridRetriever
from docintel.retrieval.reranker import LocalBM25Reranker
from docintel.retrieval.sparse_retriever import BM25SparseRetriever
from docintel.vectorstore.qdrant_store import QdrantVectorStore

DIMENSIONS = 4


def test_tokenize_light_stemming_matches_plural_forms():
    assert tokenize("cats") == tokenize("cat")
    assert tokenize("boxes") == tokenize("box")
    assert tokenize("companies") == tokenize("company")
    # Conservative: words genuinely ending in double-s or short words untouched.
    assert tokenize("glass") == ["glass"]
    assert tokenize("bus") == ["bus"]


def _chunk(content: str, content_hash: str) -> Chunk:
    return Chunk(
        document_id="doc-1",
        content=content,
        chunk_index=0,
        metadata=DocumentMetadata(
            source_uri="x.txt",
            source_type=SourceType.TXT,
            content_hash=content_hash,
            knowledge_base_id="kb-1",
        ),
    )


def _search_result(content: str, content_hash: str, score: float) -> SearchResult:
    return SearchResult(chunk=_chunk(content, content_hash), score=score)


@pytest.fixture
async def populated_store():
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(client)
    await store.ensure_collection("kb-1", DIMENSIONS)
    await store.upsert(
        "kb-1",
        [
            EmbeddedChunk(chunk=_chunk("the cat sat on the mat", "h1"), vector=[1, 0, 0, 0], model_id="m"),
            EmbeddedChunk(chunk=_chunk("stock markets rallied today", "h2"), vector=[0, 1, 0, 0], model_id="m"),
            EmbeddedChunk(chunk=_chunk("cats are wonderful pets", "h3"), vector=[0.9, 0.1, 0, 0], model_id="m"),
        ],
    )
    yield store
    await client.close()


# ---------------------------------------------------------------------------
# BM25SparseRetriever
# ---------------------------------------------------------------------------


def test_bm25_retriever_satisfies_protocol(populated_store):
    retriever = BM25SparseRetriever(populated_store)
    assert isinstance(retriever, SparseRetriever)


@pytest.mark.asyncio
async def test_bm25_retriever_finds_keyword_matches(populated_store):
    retriever = BM25SparseRetriever(populated_store)
    results = await retriever.search("kb-1", "cat", top_k=5)
    contents = [r.chunk.content for r in results]
    assert "the cat sat on the mat" in contents
    assert "cats are wonderful pets" in contents
    assert "stock markets rallied today" not in contents


@pytest.mark.asyncio
async def test_bm25_retriever_empty_collection_returns_empty():
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(client)
    retriever = BM25SparseRetriever(store)
    results = await retriever.search("nonexistent", "anything", top_k=5)
    assert results == []
    await client.close()


@pytest.mark.asyncio
async def test_bm25_retriever_caches_index_until_invalidated(populated_store):
    retriever = BM25SparseRetriever(populated_store)
    await retriever.search("kb-1", "cat", top_k=5)
    assert "kb-1" in retriever._indexes

    retriever.invalidate("kb-1")
    assert "kb-1" not in retriever._indexes


# ---------------------------------------------------------------------------
# LocalBM25Reranker
# ---------------------------------------------------------------------------


def test_reranker_satisfies_protocol():
    assert isinstance(LocalBM25Reranker(), Reranker)


@pytest.mark.asyncio
async def test_reranker_reorders_by_lexical_relevance():
    results = [
        _search_result("completely unrelated content about weather", "h1", score=0.9),
        _search_result("machine learning and neural networks", "h2", score=0.1),
        _search_result("a third distractor document about cooking", "h3", score=0.5),
    ]
    reranker = LocalBM25Reranker()
    reranked = await reranker.rerank("neural networks", results, top_k=3)
    assert reranked[0].chunk.content == "machine learning and neural networks"
    assert reranked[0].rerank_score is not None


@pytest.mark.asyncio
async def test_reranker_respects_top_k():
    results = [_search_result(f"doc {i} about cats", f"h{i}", score=0.5) for i in range(5)]
    reranker = LocalBM25Reranker()
    reranked = await reranker.rerank("cats", results, top_k=2)
    assert len(reranked) == 2


@pytest.mark.asyncio
async def test_reranker_empty_results():
    reranker = LocalBM25Reranker()
    assert await reranker.rerank("query", [], top_k=5) == []


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    model_id = "fake"
    dimensions = DIMENSIONS

    async def embed_chunks(self, chunks):
        raise NotImplementedError

    async def embed_query(self, query: str) -> list[float]:
        # Deterministic: "cat"-like queries point toward the cat vectors.
        return [1.0, 0.0, 0.0, 0.0] if "cat" in query.lower() else [0.0, 1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_hybrid_retriever_combines_dense_and_sparse(populated_store):
    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        vector_store=populated_store,
        sparse_retriever=BM25SparseRetriever(populated_store),
        alpha=0.5,
    )
    results = await retriever.retrieve("kb-1", "cat", top_k=3)
    assert len(results) > 0
    contents = [r.chunk.content for r in results]
    assert "the cat sat on the mat" in contents


@pytest.mark.asyncio
async def test_hybrid_retriever_applies_reranker_when_provided(populated_store):
    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        vector_store=populated_store,
        sparse_retriever=BM25SparseRetriever(populated_store),
        reranker=LocalBM25Reranker(),
        alpha=0.5,
    )
    results = await retriever.retrieve("kb-1", "cat", top_k=2)
    assert len(results) <= 2
    assert all(r.rerank_score is not None for r in results)


@pytest.mark.asyncio
async def test_hybrid_retriever_dense_only_alpha(populated_store):
    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        vector_store=populated_store,
        sparse_retriever=BM25SparseRetriever(populated_store),
        alpha=1.0,  # sparse contributes nothing
    )
    results = await retriever.retrieve("kb-1", "cat", top_k=3)
    assert len(results) > 0
