"""
Shared data contracts used across the ingestion -> retrieval -> generation
pipeline. These are the nouns every module agrees on; interfaces.py
defines the verbs (Protocols) that operate on them.

The ingestion side is modeled as three explicit, immutable stages:

    RawDocument       -- exactly what a Loader extracted. No hashing,
                          no cleaning. Only facts the loader could read
                          directly off the source (bytes, headers, fs stat).
    ProcessedDocument  -- output of the processing pipeline: cleaned +
                          normalized content, plus metadata (including the
                          content_hash) derived from that cleaned content.
    ChunkedDocument    -- a ProcessedDocument plus the Chunks produced from it.

Each stage is a distinct, frozen model rather than one mutable Document
that gets progressively filled in -- this makes it impossible to
accidentally read a hash computed from raw (pre-clean) content, or to
chunk something that was never cleaned/normalized.

Kept deliberately independent of any specific provider (NVIDIA, Qdrant,
etc.) so swapping an implementation never requires touching these models.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SourceType(StrEnum):
    """The origin format of a document, set by whichever LoaderPlugin handled it."""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    TXT = "txt"
    MARKDOWN = "markdown"
    HTML = "html"
    IMAGE = "image"
    WEBSITE = "website"


class RawDocument(BaseModel):
    """
    Exactly what a Loader extracted from a source -- no cleaning, no
    normalization, no hashing. Fields the loader could not directly
    observe (e.g. an author never embedded in the file) are left None
    rather than guessed.

    For HTML/website sources, `content` is the *raw* HTML markup --
    stripping/extraction happens later in the processing pipeline, not
    in the loader (loaders only extract; see html_loader.py / web_loader.py).
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    content: str
    source_uri: str
    source_type: SourceType
    knowledge_base_id: str

    # Facts observable directly from the source (filesystem stat, HTTP
    # headers, or trivially-exposed container metadata like an HTML
    # <title> or a MarkItDown-reported title). Never inferred/computed.
    title: str | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None

    loaded_at: datetime = Field(default_factory=_utc_now)
    extra: dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """
    Metadata attached to a ProcessedDocument, populated by the processing
    stage (metadata_extractor.py) from a RawDocument + its cleaned content.
    Fields the extractor cannot determine are left None rather than guessed.
    """

    source_uri: str
    source_type: SourceType
    title: str | None = None
    author: str | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    page_count: int | None = None
    language: str | None = None
    content_hash: str  # sha256 of normalized content; drives incremental indexing/caching
    knowledge_base_id: str
    extra: dict[str, Any] = Field(default_factory=dict)


class ProcessedDocument(BaseModel):
    """Cleaned + normalized content, ready for chunking. Immutable."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=_new_id)
    raw_document_id: str
    content: str  # cleaned/normalized text
    metadata: DocumentMetadata
    processed_at: datetime = Field(default_factory=_utc_now)


class Chunk(BaseModel):
    """A single retrievable unit produced by a Chunker from a ProcessedDocument."""

    id: str = Field(default_factory=_new_id)
    document_id: str  # ProcessedDocument.id
    content: str
    chunk_index: int  # position within the parent document
    metadata: DocumentMetadata  # inherited/augmented from the parent document
    token_count: int | None = None


class ChunkedDocument(BaseModel):
    """A ProcessedDocument plus the Chunks produced from it by a Chunker."""

    model_config = ConfigDict(frozen=True)

    processed_document: ProcessedDocument
    chunks: list[Chunk]


class EmbeddedChunk(BaseModel):
    """A Chunk plus its vector representation, ready for upsert into a VectorStore."""

    chunk: Chunk
    vector: list[float]
    model_id: str  # which embedding model produced this vector -- required for cache validity


class SearchResult(BaseModel):
    """A single hit returned from retrieval, before or after reranking."""

    chunk: Chunk
    score: float
    rerank_score: float | None = None


class Citation(BaseModel):
    """A source attribution attached to a generated answer."""

    chunk_id: str
    document_id: str
    source_uri: str
    title: str | None = None
    excerpt: str  # short quoted/paraphrased snippet shown to the user
    score: float


class KnowledgeBase(BaseModel):
    """
    A knowledge base is purely configuration: an id (== the underlying
    VectorStore collection name), display metadata, and the settings that
    govern how documents indexed into it are chunked and how queries
    against it are answered. "Portfolio Assistant" vs "QA Assistant" vs
    "Company Knowledge Base" are never separate code paths -- they are
    different KnowledgeBase records pointing at different collections.
    """

    id: str = Field(default_factory=_new_id)
    name: str
    description: str | None = None
    embedding_model_id: str
    embedding_dimensions: int
    system_prompt: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)


class GenerationChunk(BaseModel):
    """A single streamed token/segment from an LLM, plus any citations resolved so far."""

    text: str
    citations: list[Citation] = Field(default_factory=list)
    is_final: bool = False
