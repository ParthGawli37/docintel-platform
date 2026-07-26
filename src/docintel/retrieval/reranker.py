"""
LocalBM25Reranker: default Reranker implementation.

Re-scores an already-retrieved candidate set using BM25 computed
fresh over just those candidates (not the full collection) -- this
sharpens ranking precision on the exact query wording without needing
an external cross-encoder/reranking API. It's a legitimate, fully-local,
fully-tested default; NVIDIA (or any other) reranking API can be
swapped in later by writing a new class satisfying the same Reranker
protocol -- deliberately not attempted here without a confirmed API
contract for that specific endpoint (see project notes on never
inventing unverified API integrations).
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from docintel.core.logging import get_logger
from docintel.core.models import SearchResult
from docintel.retrieval._tokenize import tokenize

logger = get_logger(__name__)


class LocalBM25Reranker:
    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []

        corpus = [tokenize(r.chunk.content) for r in results]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenize(query))

        rescored = [
            result.model_copy(update={"rerank_score": float(score)})
            for result, score in zip(results, scores)
        ]
        rescored.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        return rescored[:top_k]
