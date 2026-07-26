"""
Ingestion pipeline orchestrator: RawDocument -> ProcessedDocument.

Selects the appropriate Cleaner per source_type (HTML/WEBSITE get
HtmlCleaner; everything else gets DefaultCleaner), then runs the
Normalizer, then the MetadataExtractor (which computes content_hash from
the final cleaned+normalized content).

Chunking (ProcessedDocument -> ChunkedDocument) is a separate stage
invoked by the caller after processing -- kept as a distinct method here
so callers needing only ProcessedDocument (e.g. a future dedup check
against the HashRegistry before paying for chunking/embedding) can stop
after process() without being forced through chunking.
"""

from __future__ import annotations

from docintel.core.interfaces import Chunker, Cleaner, MetadataExtractor, Normalizer
from docintel.core.logging import get_logger
from docintel.core.models import ChunkedDocument, ProcessedDocument, RawDocument, SourceType
from docintel.ingestion.processing.cleaner import DefaultCleaner, HtmlCleaner
from docintel.ingestion.processing.metadata_extractor import DefaultMetadataExtractor
from docintel.ingestion.processing.normalizer import WhitespaceNormalizer

logger = get_logger(__name__)

_HTML_LIKE_SOURCE_TYPES = {SourceType.HTML, SourceType.WEBSITE}


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow. Depends only on the Cleaner /
    Normalizer / MetadataExtractor / Chunker Protocols, so any stage is
    swappable without touching this class.
    """

    def __init__(
        self,
        *,
        default_cleaner: Cleaner | None = None,
        html_cleaner: Cleaner | None = None,
        normalizer: Normalizer | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        chunker: Chunker,
    ) -> None:
        self._default_cleaner: Cleaner = default_cleaner or DefaultCleaner()
        self._html_cleaner: Cleaner = html_cleaner or HtmlCleaner()
        self._normalizer: Normalizer = normalizer or WhitespaceNormalizer()
        self._metadata_extractor: MetadataExtractor = metadata_extractor or DefaultMetadataExtractor()
        self._chunker = chunker

    def _select_cleaner(self, source_type: SourceType) -> Cleaner:
        return self._html_cleaner if source_type in _HTML_LIKE_SOURCE_TYPES else self._default_cleaner

    def process(self, raw: RawDocument) -> ProcessedDocument:
        cleaner = self._select_cleaner(raw.source_type)
        cleaned = cleaner.clean(raw.content)
        normalized = self._normalizer.normalize(cleaned)
        metadata = self._metadata_extractor.extract(raw, normalized)

        processed = ProcessedDocument(
            raw_document_id=raw.id,
            content=normalized,
            metadata=metadata,
        )
        logger.info(
            "document_processed",
            raw_document_id=raw.id,
            source_uri=raw.source_uri,
            content_hash=metadata.content_hash,
        )
        return processed

    async def chunk(self, processed: ProcessedDocument) -> ChunkedDocument:
        """Chunk an already-processed document. Split out from process_and_chunk
        so callers that need to inspect/hash the ProcessedDocument before
        deciding whether to chunk (e.g. the incremental indexer) can do so
        without re-running process()."""
        return await self._chunker.chunk(processed)

    async def process_and_chunk(self, raw: RawDocument) -> ChunkedDocument:
        processed = self.process(raw)
        chunked = await self.chunk(processed)
        logger.info(
            "document_chunked",
            processed_document_id=processed.id,
            chunk_count=len(chunked.chunks),
        )
        return chunked
