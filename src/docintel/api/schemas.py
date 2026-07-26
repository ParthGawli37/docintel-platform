"""Request/response schemas for the API layer -- kept separate from core/models.py,
which holds the internal pipeline data contracts, not wire formats."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreateKnowledgeBaseRequest(BaseModel):
    name: str
    description: str | None = None
    embedding_model_id: str | None = Field(
        default=None, description="Defaults to the server's configured NVIDIA embedding model."
    )
    embedding_dimensions: int | None = Field(
        default=None, description="Defaults to the server's configured embedding dimensions."
    )
    system_prompt: str | None = None


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str | None
    embedding_model_id: str
    embedding_dimensions: int
    system_prompt: str | None


class IngestUrlRequest(BaseModel):
    url: str


class IngestResultResponse(BaseModel):
    source: str
    skipped: bool
    chunk_count: int
    error: str | None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    source_uri: str
    title: str | None
    excerpt: str
    score: float
