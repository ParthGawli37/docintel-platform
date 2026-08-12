"""
IncrementalIndexer: orchestrates the full path from a source (file path
or URL) into a searchable knowledge base collection.

Flow: Loader.load() -> IngestionPipeline.process() -> HashRegistry check
(skip if unchanged) -> IngestionPipeline.chunk() -> Embedder.embed_chunks()
(cache-aware if wrapped in CachedEmbedder) -> VectorStore.upsert() ->
HashRegistry.set_hash().

The HashRegistry check happens AFTER processing (so the hash reflects
cleaned/normalized content, per the staged-pipeline design) but BEFORE
chunking/embedding -- skipping unchanged documents avoids paying for
both steps, which is the whole point of incremental indexing.

When a source changes, the previous vectors for that source are removed
before the replacement chunks are upserted. Without that cleanup, each
reindex would leave stale chunks in the collection because Chunk IDs are
new for each processing pass.

Callers can register invalidate callbacks (e.g. BM25SparseRetriever.invalidate)
to run after a successful write, so retrievers with their own caches stay
in sync without the indexer needing to know how many retrievers exist or
how their caches work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from docintel.core.interfaces import Embedder, HashRegistry, Loader, VectorStore
from docintel.core.logging import get_logger
from docintel.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)

InvalidateCallback = Callable[[str], None]


@dataclass
class IndexingResult:
    source: str
    skipped: bool
    chunk_count: int = 0
    error: str | None = None


@dataclass
class BatchIndexingResult:
    results: list[IndexingResult] = field(default_factory=list)

    @property
    def indexed_count(self) -> int:
        return sum(1 for r in self.results if not r.skipped and r.error is None)

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.error is not None)


class IncrementalIndexer:
    def __init__(
        self,
        loader: Loader,
        pipeline: IngestionPipeline,
        embedder: Embedder,
        vector_store: VectorStore,
        hash_registry: HashRegistry,
    ) -> None:
        self._loader = loader
        self._pipeline = pipeline
        self._embedder = embedder
        self._vector_store = vector_store
        self._hash_registry = hash_registry
        self._invalidate_callbacks: list[InvalidateCallback] = []

    def add_invalidate_callback(self, callback: InvalidateCallback) -> None:
        self._invalidate_callbacks.append(callback)

    async def index_source(self, source: str | Path, knowledge_base_id: str) -> IndexingResult:
        source_str = str(source)
        try:
            raw_documents = await self._loader.load(source, knowledge_base_id)
        except Exception as exc:  # noqa: BLE001 -- loaders wrap heterogeneous third-party
            # exceptions (markitdown, tesseract, httpx, ...); a single bad
            # source must not abort the rest of a batch, so this is
            # deliberately broad and always logged with the source it hit.
            logger.error("indexing_load_failed", source=source_str, error=str(exc))
            return IndexingResult(source=source_str, skipped=False, error=str(exc))

        total_chunks = 0
        any_changed = False

        for raw in raw_documents:
            processed = self._pipeline.process(raw)

            changed = await self._hash_registry.has_changed(
                knowledge_base_id, raw.source_uri, processed.metadata.content_hash
            )
            if not changed:
                logger.info("indexing_skip_unchanged", source_uri=raw.source_uri)
                continue

            any_changed = True
            chunked = await self._pipeline.chunk(processed)

            # Chunk IDs are generated during chunking, so a changed source
            # gets new point IDs. Remove the old source's chunks first to
            # prevent stale content from remaining searchable.
            existing_chunks = await self._vector_store.get_all_chunks(knowledge_base_id)
            stale_document_ids = {
                chunk.document_id
                for chunk in existing_chunks
                if chunk.metadata.source_uri == raw.source_uri
            }
            for document_id in stale_document_ids:
                await self._vector_store.delete_by_document_id(
                    knowledge_base_id, document_id
                )

            if chunked.chunks:
                embedded_chunks = await self._embedder.embed_chunks(chunked.chunks)
                await self._vector_store.upsert(knowledge_base_id, embedded_chunks)
                total_chunks += len(chunked.chunks)

            await self._hash_registry.set_hash(
                knowledge_base_id, raw.source_uri, processed.metadata.content_hash
            )

            logger.info(
                "indexing_document_complete",
                source_uri=raw.source_uri,
                chunk_count=len(chunked.chunks),
                removed_stale_documents=len(stale_document_ids),
            )

        if any_changed:
            for callback in self._invalidate_callbacks:
                callback(knowledge_base_id)

        return IndexingResult(source=source_str, skipped=not any_changed, chunk_count=total_chunks)

    async def index_batch(
        self, sources: list[str | Path], knowledge_base_id: str
    ) -> BatchIndexingResult:
        batch_result = BatchIndexingResult()
        for source in sources:
            result = await self.index_source(source, knowledge_base_id)
            batch_result.results.append(result)
        return batch_result
