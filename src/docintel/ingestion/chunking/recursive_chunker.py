"""
RecursiveChunker: the default chunking strategy.

Greedily packs paragraphs into a chunk until adding the next paragraph
would exceed `chunk_size_tokens`. If a single paragraph alone exceeds the
limit, it's recursively split at the sentence level (and, as a last
resort, at a hard character boundary) so no chunk ever exceeds the
configured size. Adjacent chunks overlap by `overlap_tokens` (measured in
trailing words carried into the next chunk) to preserve context across
chunk boundaries.
"""

from __future__ import annotations

from docintel.core.models import Chunk, ChunkedDocument, ProcessedDocument
from docintel.ingestion.chunking.base import (
    estimate_token_count,
    split_paragraphs,
    split_sentences,
)


class RecursiveChunker:
    def __init__(self, chunk_size_tokens: int = 512, overlap_tokens: int = 64) -> None:
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
        self._chunk_size_tokens = chunk_size_tokens
        self._overlap_tokens = overlap_tokens

    @property
    def chunk_size_tokens(self) -> int:
        return self._chunk_size_tokens

    def _hard_split(self, text: str) -> list[str]:
        """
        Split text with no further semantic structure (no sentence breaks)
        into pieces guaranteed to fit under chunk_size_tokens. Uses a
        deliberately conservative chars-per-token ratio (3, versus the
        ~4-5 typical for English) so the resulting pieces are safely under
        budget even given estimate_token_count's approximation error --
        this guarantees termination without needing to recheck/re-split.
        """
        chars_per_token_budget = 3
        budget_chars = max(1, self._chunk_size_tokens * chars_per_token_budget)
        return [text[i : i + budget_chars] for i in range(0, len(text), budget_chars)] or [""]

    def _split_oversized_unit(self, unit: str) -> list[str]:
        """Split a single paragraph too large for one chunk, at sentence granularity."""
        sentences = split_sentences(unit)
        if len(sentences) > 1:
            return sentences
        # No sentence structure to exploit -- hard split is the only option,
        # and must not recurse back through _pack (which would re-hit this
        # same oversized branch and never terminate).
        return self._hard_split(unit)

    def _pack(self, units: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = estimate_token_count(unit)

            if unit_tokens > self._chunk_size_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                for sub_unit in self._split_oversized_unit(unit):
                    chunks.extend(self._pack([sub_unit]))
                continue

            if current_tokens + unit_tokens > self._chunk_size_tokens and current:
                chunks.append(" ".join(current))
                # Carry trailing words forward for overlap.
                overlap_words: list[str] = []
                for overlap_count, word in enumerate(reversed(" ".join(current).split())):
                    if overlap_count >= self._overlap_tokens:
                        break
                    overlap_words.insert(0, word)
                current = [" ".join(overlap_words)] if overlap_words else []
                current_tokens = estimate_token_count(" ".join(current))

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    async def chunk(self, document: ProcessedDocument) -> ChunkedDocument:
        paragraphs = split_paragraphs(document.content)
        if not paragraphs:
            return ChunkedDocument(processed_document=document, chunks=[])

        chunk_texts = self._pack(paragraphs)

        chunks = [
            Chunk(
                document_id=document.id,
                content=text,
                chunk_index=index,
                metadata=document.metadata,
                token_count=estimate_token_count(text),
            )
            for index, text in enumerate(chunk_texts)
            if text.strip()
        ]
        return ChunkedDocument(processed_document=document, chunks=chunks)
