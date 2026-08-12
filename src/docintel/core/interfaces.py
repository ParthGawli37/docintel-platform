"""
Protocol (structural interface) definitions for every pluggable component
in the platform.

Design rule: business logic (pipeline orchestration, indexing, retrieval
services) depends only on these Protocols, never on a concrete provider class
 directly. Concrete implementations (NVIDIA, Qdrant, tesseract, ...)
live in their respective modules and are wired together at the
composition root (api/main.py's dependency setup).

Using `typing.Protocol` rather than ABCs so implementations don't need to
inherit from anything -- any class with the right method signatures
satisfies the contract (structural typing), which keeps providers
decoupled from this module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from docintel.core.models import (
    Chunk,
    ChunkedDocument,
    Citation,
    DocumentMetadata,
    EmbeddedChunk,
    GenerationChunk,
    ProcessedDocument,
    RawDocument,
    SearchResult,
)

# ---------------------------------------------------------------------------
# Ingestion: loaders
# ---------------------------------------------------------------------------


@runtime_checkable
class Loader(Protocol):
    """
    Converts a raw source (file path or URL) into one or more RawDocuments.

    Loaders are extraction-only: they do NOT clean, normalize, hash, or
    extract anything beyond what the source trivially exposes (e.g. an
    HTML <title>, a filesystem mtime, an HTTP Content-Type header).
    Cleaning, normalization, and hashing are the processing stage's job.
    """

    supported_extensions: tuple[str, ...]
    """File extensions this loader handles, e.g. (".pdf",). Empty for
    loaders that operate on URLs instead (web_loader)."""

    def can_load(self, source: str | Path) -> bool:
        """Return True if this loader can handle the given path/URL."""
        ...

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        """Load and return raw Document(s) from the source."""
        ...


# ---------------------------------------------------------------------------
# Ingestion: OCR (used by the image loader; independently swappable)
# ---------------------------------------------------------------------------


@runtime_checkable
class OCRProvider(Protocol):
    """
    Extracts text from an image. Tesseract is the default implementation;
    this abstraction exists so NVIDIA OCR, Azure OCR, Google Vision, etc.
    can be swapped in later without touching ImageOcrLoader.
    """

    provider_name: str

    async def extract_text(self, image_path: Path) -> str: ...


# ---------------------------------------------------------------------------
# Ingestion: processing (cleaning / normalization / metadata)
# ---------------------------------------------------------------------------


@runtime_checkable
class Cleaner(Protocol):
    """
    Strips noise (boilerplate, control characters, malformed markup,
    HTML tags, etc.) from raw content. Different source types may need
    different cleaners (e.g. an HtmlCleaner vs a plain-text cleaner);
    the processing pipeline selects the right one per RawDocument.source_type.
    """

    def clean(self, content: str) -> str: ...


@runtime_checkable
class Normalizer(Protocol):
    """Normalizes whitespace, encoding, casing conventions, etc. for consistent chunking."""

    def normalize(self, content: str) -> str: ...


@runtime_checkable
class MetadataExtractor(Protocol):
    """
    Derives DocumentMetadata (including content_hash) from a RawDocument
    and its cleaned/normalized content. Fields it cannot determine are
    left unset -- never inferred/invented.
    """

    def extract(self, raw: RawDocument, cleaned_content: str) -> DocumentMetadata: ...


# ---------------------------------------------------------------------------
# Ingestion: chunking
# ---------------------------------------------------------------------------


@runtime_checkable
class Chunker(Protocol):
    """Splits a ProcessedDocument into retrievable Chunks."""

    async def chunk(self, document: ProcessedDocument) -> ChunkedDocument: ...


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """
    Produces vector embeddings for chunks and queries.

    model_id and dimensions are exposed so callers (cache layer, vector
    store schema setup) can validate compatibility without hardcoding
    provider-specific assumptions.
    """

    model_id: str
    dimensions: int

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]: ...

    async def embed_query(self, query: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """
    Abstraction over the vector database. Qdrant is the default
    implementation; any provider satisfying this contract is swappable
    in without touching retrieval/indexing logic.

    All operations are scoped to a `collection` (== knowledge base id),
    per the collection-based knowledge base design.
    """

    async def ensure_collection(self, collection: str, dimensions: int) -> None: ...

    async def drop_collection(self, collection: str) -> None:
        """Permanently delete a collection and all its vectors. Used when a KnowledgeBase is deleted."""
        ...

    async def upsert(self, collection: str, embedded_chunks: list[EmbeddedChunk]) -> None: ...

    async def delete_by_document_id(self, collection: str, document_id: str) -> None: ...

    async def delete_by_source_uri(self, collection: str, source_uri: str) -> int:
        """Delete all vectors belonging to a source and return the number removed."""
        ...

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        ...

    async def get_existing_content_hashes(self, collection: str) -> set[str]:
        """
        Return content hashes already indexed in this collection, used by
        the incremental indexer to skip unchanged documents without
        needing a full re-embed.
        """
        ...

    async def get_all_chunks(self, collection: str) -> list[Chunk]:
        """
        Return every Chunk currently stored in the collection. Used by
        SparseRetriever implementations (e.g. BM25) to build/refresh their
        lexical index -- dense search doesn't need this, only sparse/hybrid.
        """
        ...


# ---------------------------------------------------------------------------
# Retrieval: hybrid search + reranking
# ---------------------------------------------------------------------------


@runtime_checkable
class SparseRetriever(Protocol):
    """Keyword/BM25-style retriever, combined with dense search for hybrid retrieval."""

    async def search(
        self, collection: str, query: str, top_k: int
    ) -> list[SearchResult]: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores an initial candidate set for final ranking precision."""

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]: ...


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@runtime_checkable
class LLM(Protocol):
    """Generation provider. Nemotron (via NVIDIA API) is the default implementation."""

    model_id: str

    async def stream_generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        """Stream a generated answer, yielding text incrementally with citations."""
        ...


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


@runtime_checkable
class CitationBuilder(Protocol):
    """Builds user-facing Citation objects from the search results used in generation."""

    def build(self, results: list[SearchResult]) -> list[Citation]: ...


# ---------------------------------------------------------------------------
# Caching (embeddings, document hashes) -- backed by the storage layer
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingCache(Protocol):
    """
    Caches embedding vectors keyed by (content_hash, model_id), so identical
    content is never re-embedded against the same model -- saving NVIDIA
    API spend and latency.
    """

    async def get(self, content_hash: str, model_id: str) -> list[float] | None: ...

    async def set(self, content_hash: str, model_id: str, vector: list[float]) -> None: ...


@runtime_checkable
class HashRegistry(Protocol):
    """
    Tracks content hashes per (knowledge_base_id, source_uri) to power
    incremental indexing: unchanged sources are skipped entirely.
    """

    async def get_hash(self, knowledge_base_id: str, source_uri: str) -> str | None: ...

    async def set_hash(self, knowledge_base_id: str, source_uri: str, content_hash: str) -> None: ...

    async def has_changed(
        self, knowledge_base_id: str, source_uri: str, current_hash: str
    ) -> bool: ...
