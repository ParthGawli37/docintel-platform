"""
Metadata extraction: derives DocumentMetadata from a RawDocument and its
cleaned/normalized content.

This is the ONLY place content_hash is computed. Loaders never hash --
they only extract -- so the hash always reflects the same normalized
content that will actually be chunked/embedded/indexed, which is what
makes it valid for incremental-indexing and embedding-cache decisions.

Fields the RawDocument doesn't carry (author, page_count, language) are
left None here -- this extractor never infers/guesses them from content.
"""

from __future__ import annotations

from docintel.core.hashing import compute_content_hash
from docintel.core.models import DocumentMetadata, RawDocument


class DefaultMetadataExtractor:
    def extract(self, raw: RawDocument, cleaned_content: str) -> DocumentMetadata:
        return DocumentMetadata(
            source_uri=raw.source_uri,
            source_type=raw.source_type,
            title=raw.title,
            mime_type=raw.mime_type,
            file_size_bytes=raw.file_size_bytes,
            created_at=raw.created_at,
            modified_at=raw.modified_at,
            content_hash=compute_content_hash(cleaned_content),
            knowledge_base_id=raw.knowledge_base_id,
            extra=dict(raw.extra),
        )
