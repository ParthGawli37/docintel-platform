"""
DefaultCitationBuilder: builds user-facing Citation objects from the
SearchResults that were actually used as context for a generated answer.

Excerpts are truncated verbatim substrings of the chunk's own indexed
content (not re-summarized or rewritten) -- citations must point back to
exactly what was retrieved, never to a paraphrase that could drift from
the source.
"""

from __future__ import annotations

from docintel.core.models import Citation, SearchResult

_DEFAULT_EXCERPT_LENGTH = 240


class DefaultCitationBuilder:
    def __init__(self, excerpt_length: int = _DEFAULT_EXCERPT_LENGTH) -> None:
        self._excerpt_length = excerpt_length

    def build(self, results: list[SearchResult]) -> list[Citation]:
        citations = []
        for result in results:
            content = result.chunk.content.strip()
            excerpt = (
                content[: self._excerpt_length].rstrip() + "..."
                if len(content) > self._excerpt_length
                else content
            )
            citations.append(
                Citation(
                    chunk_id=result.chunk.id,
                    document_id=result.chunk.document_id,
                    source_uri=result.chunk.metadata.source_uri,
                    title=result.chunk.metadata.title,
                    excerpt=excerpt,
                    score=result.rerank_score if result.rerank_score is not None else result.score,
                )
            )
        return citations
