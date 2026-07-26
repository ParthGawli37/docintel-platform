"""
CachedEmbedder: wraps any Embedder with an EmbeddingCache, so content that
has already been embedded by the same model is never re-sent to the
provider. Decorator pattern -- CachedEmbedder itself satisfies Embedder,
so it's a drop-in replacement anywhere an Embedder is expected.

Keyed by chunk.metadata.content_hash (computed once, from cleaned/
normalized content, by the processing stage) + the inner embedder's
model_id. embed_query is intentionally NOT cached: queries are typically
unique per request and have no stable content_hash the way indexed
chunks do, so caching them would rarely hit and isn't worth the
complexity of hashing arbitrary query strings here.
"""

from __future__ import annotations

from docintel.core.interfaces import Embedder, EmbeddingCache
from docintel.core.logging import get_logger
from docintel.core.models import Chunk, EmbeddedChunk

logger = get_logger(__name__)


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache
        self.model_id: str = inner.model_id
        self.dimensions: int = inner.dimensions

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        results: dict[str, EmbeddedChunk] = {}
        misses: list[Chunk] = []

        for chunk in chunks:
            cached_vector = await self._cache.get(chunk.metadata.content_hash, self.model_id)
            if cached_vector is not None:
                results[chunk.id] = EmbeddedChunk(
                    chunk=chunk, vector=cached_vector, model_id=self.model_id
                )
            else:
                misses.append(chunk)

        logger.info(
            "embedding_cache_lookup",
            total=len(chunks),
            hits=len(chunks) - len(misses),
            misses=len(misses),
        )

        if misses:
            newly_embedded = await self._inner.embed_chunks(misses)
            for embedded_chunk in newly_embedded:
                results[embedded_chunk.chunk.id] = embedded_chunk
                await self._cache.set(
                    embedded_chunk.chunk.metadata.content_hash,
                    self.model_id,
                    embedded_chunk.vector,
                )

        # Preserve the caller's original chunk order.
        return [results[chunk.id] for chunk in chunks]

    async def embed_query(self, query: str) -> list[float]:
        return await self._inner.embed_query(query)
