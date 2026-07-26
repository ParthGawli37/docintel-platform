"""
HybridRetriever: combines dense (embedding + VectorStore) and sparse
(BM25) search results via weighted min-max-normalized score fusion,
optionally followed by a Reranker pass.

`alpha` controls the dense/sparse weighting (1.0 = dense only, 0.0 =
sparse only), sourced from Settings.hybrid_search_alpha -- not
hardcoded, since the right balance is corpus-dependent and explicitly
called out as something to tune per knowledge base in .env.example.
"""

from __future__ import annotations

from docintel.core.interfaces import Embedder, Reranker, SparseRetriever, VectorStore
from docintel.core.logging import get_logger
from docintel.core.models import SearchResult

logger = get_logger(__name__)


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_retriever: SparseRetriever,
        reranker: Reranker | None = None,
        alpha: float = 0.5,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_retriever = sparse_retriever
        self._reranker = reranker
        self._alpha = alpha

    async def retrieve(
        self,
        collection: str,
        query: str,
        top_k: int,
        candidate_pool_size: int | None = None,
    ) -> list[SearchResult]:
        pool_size = candidate_pool_size or max(top_k * 4, top_k)

        query_vector = await self._embedder.embed_query(query)
        dense_results = await self._vector_store.search(collection, query_vector, pool_size)
        sparse_results = await self._sparse_retriever.search(collection, query, pool_size)

        fused = self._fuse(dense_results, sparse_results)
        fused.sort(key=lambda r: r.score, reverse=True)
        candidates = fused[:pool_size]

        if self._reranker is not None:
            return await self._reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]

    def _fuse(
        self, dense: list[SearchResult], sparse: list[SearchResult]
    ) -> list[SearchResult]:
        dense_scores = _min_max_normalize([r.score for r in dense])
        sparse_scores = _min_max_normalize([r.score for r in sparse])

        combined: dict[str, SearchResult] = {}
        fused_scores: dict[str, float] = {}

        for result, norm_score in zip(dense, dense_scores):
            chunk_id = result.chunk.id
            combined[chunk_id] = result
            fused_scores[chunk_id] = self._alpha * norm_score

        for result, norm_score in zip(sparse, sparse_scores):
            chunk_id = result.chunk.id
            contribution = (1 - self._alpha) * norm_score
            if chunk_id in fused_scores:
                fused_scores[chunk_id] += contribution
            else:
                combined[chunk_id] = result
                fused_scores[chunk_id] = contribution

        return [
            combined[chunk_id].model_copy(update={"score": score})
            for chunk_id, score in fused_scores.items()
        ]
