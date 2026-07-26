"""
SemanticChunker: groups consecutive sentences into a chunk while their
embedding-similarity to the running chunk stays above a threshold,
starting a new chunk when similarity drops (topic shift) or the size
budget is hit.

Depends on an injected `embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]]`
rather than the concrete Embedder implementation -- this keeps the
chunking module free of a dependency on embeddings/ (which is built in a
later step) while still letting the real NVIDIA embedder be plugged in
wherever this chunker is actually constructed (composition root).

Not the default chunk strategy (RecursiveChunker is) -- semantic chunking
costs an embedding call per document at ingest time, which is a real
trade-off callers should opt into deliberately via CHUNK_STRATEGY=semantic.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable

from docintel.core.models import Chunk, ChunkedDocument, ProcessedDocument
from docintel.ingestion.chunking.base import estimate_token_count, split_sentences

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker:
    def __init__(
        self,
        embed_fn: EmbedFn,
        similarity_threshold: float = 0.5,
        max_chunk_size_tokens: int = 512,
    ) -> None:
        self._embed_fn = embed_fn
        self._similarity_threshold = similarity_threshold
        self._max_chunk_size_tokens = max_chunk_size_tokens

    async def chunk(self, document: ProcessedDocument) -> ChunkedDocument:
        sentences = split_sentences(document.content)
        if not sentences:
            return ChunkedDocument(processed_document=document, chunks=[])
        if len(sentences) == 1:
            return ChunkedDocument(
                processed_document=document,
                chunks=[
                    Chunk(
                        document_id=document.id,
                        content=sentences[0],
                        chunk_index=0,
                        metadata=document.metadata,
                        token_count=estimate_token_count(sentences[0]),
                    )
                ],
            )

        vectors = await self._embed_fn(sentences)

        groups: list[list[int]] = [[0]]
        group_tokens = estimate_token_count(sentences[0])

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(vectors[i - 1], vectors[i])
            sentence_tokens = estimate_token_count(sentences[i])
            fits_size = group_tokens + sentence_tokens <= self._max_chunk_size_tokens

            if sim >= self._similarity_threshold and fits_size:
                groups[-1].append(i)
                group_tokens += sentence_tokens
            else:
                groups.append([i])
                group_tokens = sentence_tokens

        chunks = [
            Chunk(
                document_id=document.id,
                content=" ".join(sentences[i] for i in group),
                chunk_index=index,
                metadata=document.metadata,
                token_count=sum(estimate_token_count(sentences[i]) for i in group),
            )
            for index, group in enumerate(groups)
        ]
        return ChunkedDocument(processed_document=document, chunks=chunks)
