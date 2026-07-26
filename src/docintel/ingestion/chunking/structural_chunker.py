"""
StructuralChunker: splits content along Markdown headers (#, ##, ###, ...)
first, preserving document structure/hierarchy in the chunk boundaries,
then recursively chunks within each section so no section exceeds the
configured size. Best suited for well-structured Markdown/converted
documents (e.g. MarkItDown output) where headers carry real semantic
sections.
"""

from __future__ import annotations

import re

from docintel.core.models import Chunk, ChunkedDocument, ProcessedDocument
from docintel.ingestion.chunking.base import estimate_token_count
from docintel.ingestion.chunking.recursive_chunker import RecursiveChunker

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class StructuralChunker:
    def __init__(self, chunk_size_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self._fallback = RecursiveChunker(chunk_size_tokens, overlap_tokens)

    def _split_by_headers(self, content: str) -> list[tuple[str | None, str]]:
        """Return [(header_text_or_None, section_body), ...]."""
        matches = list(_HEADER_RE.finditer(content))
        if not matches:
            return [(None, content)]

        sections: list[tuple[str | None, str]] = []
        preamble = content[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

        for i, match in enumerate(matches):
            header_text = match.group(2).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[body_start:body_end].strip()
            sections.append((header_text, body))

        return sections

    async def chunk(self, document: ProcessedDocument) -> ChunkedDocument:
        sections = self._split_by_headers(document.content)
        chunks: list[Chunk] = []
        index = 0

        for header, body in sections:
            section_text = f"{header}\n{body}" if header else body
            if not section_text.strip():
                continue

            if estimate_token_count(section_text) <= self._fallback.chunk_size_tokens:
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        content=section_text,
                        chunk_index=index,
                        metadata=document.metadata,
                        token_count=estimate_token_count(section_text),
                    )
                )
                index += 1
            else:
                sub_document = document.model_copy(update={"content": section_text})
                sub_chunked = await self._fallback.chunk(sub_document)
                for sub_chunk in sub_chunked.chunks:
                    chunks.append(sub_chunk.model_copy(update={"chunk_index": index}))
                    index += 1

        return ChunkedDocument(processed_document=document, chunks=chunks)
