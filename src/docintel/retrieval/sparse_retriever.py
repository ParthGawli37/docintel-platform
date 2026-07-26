"""
BM25SparseRetriever: default SparseRetriever implementation, providing
the keyword-search half of hybrid retrieval (dense search is handled by
Embedder + VectorStore directly).

Builds its lexical index by pulling all chunks for a collection from the
VectorStore (via get_all_chunks) and caching a per-collection BM25 index
in memory. The cache is intentionally simple (no TTL/auto-refresh) --
call `invalidate(collection)` after indexing new documents into that
collection so the next search rebuilds with fresh content. This keeps
the retriever itself free of any indexing-lifecycle logic; the indexer
(a later module) is responsible for calling invalidate() at the right time.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from docintel.core.interfaces import VectorStore
from docintel.core.logging import get_logger
from docintel.core.models import Chunk, SearchResult
from docintel.retrieval._tokenize import tokenize

logger = get_logger(__name__)


class _CollectionIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None


class BM25SparseRetriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store
        self._indexes: dict[str, _CollectionIndex] = {}

    def invalidate(self, collection: str) -> None:
        self._indexes.pop(collection, None)

    async def _get_index(self, collection: str) -> _CollectionIndex:
        if collection not in self._indexes:
            chunks = await self._vector_store.get_all_chunks(collection)
            self._indexes[collection] = _CollectionIndex(chunks)
            logger.info("bm25_index_built", collection=collection, chunk_count=len(chunks))
        return self._indexes[collection]

    async def search(self, collection: str, query: str, top_k: int) -> list[SearchResult]:
        index = await self._get_index(collection)
        if index.bm25 is None or not index.chunks:
            return []

        scores = index.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(index.chunks, scores), key=lambda pair: pair[1], reverse=True)

        return [
            SearchResult(chunk=chunk, score=float(score))
            for chunk, score in ranked[:top_k]
            if score > 0
        ]
