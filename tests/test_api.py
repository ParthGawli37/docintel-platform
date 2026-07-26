import json
from datetime import UTC
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from qdrant_client import AsyncQdrantClient

from docintel.api.routers import ingestion, knowledge_bases, query
from docintel.core.models import Chunk, EmbeddedChunk, GenerationChunk
from docintel.indexing.indexer import IncrementalIndexer
from docintel.ingestion.chunking.recursive_chunker import RecursiveChunker
from docintel.ingestion.loaders import bootstrap_loaders
from docintel.ingestion.loaders.base import registry as loader_registry
from docintel.ingestion.pipeline import IngestionPipeline
from docintel.knowledge_base.manager import KnowledgeBaseManager
from docintel.retrieval.hybrid_retriever import HybridRetriever
from docintel.retrieval.reranker import LocalBM25Reranker
from docintel.retrieval.sparse_retriever import BM25SparseRetriever
from docintel.storage.hash_registry import SqliteHashRegistry
from docintel.storage.kb_store import SqliteKnowledgeBaseStore
from docintel.storage.raw_store import LocalRawFileStore
from docintel.vectorstore.qdrant_store import QdrantVectorStore

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeSettings:
    nvidia_embedding_model = "fake/embed-model"
    nvidia_embedding_dimensions = 4


class _FakeEmbedder:
    model_id = "fake/embed-model"
    dimensions = 4

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [EmbeddedChunk(chunk=c, vector=[1.0, 0.0, 0.0, 0.0], model_id=self.model_id) for c in chunks]

    async def embed_query(self, query_text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class _FakeLLM:
    model_id = "fake/llm"

    async def stream_generate(self, query, context, system_prompt=None):
        from docintel.citations.builder import DefaultCitationBuilder

        citations = DefaultCitationBuilder().build(context)
        yield GenerationChunk(text="Hello ", citations=[], is_final=False)
        yield GenerationChunk(text="world.", citations=citations, is_final=True)


class _TestContainer:
    """Hand-assembled container: real wiring everywhere testable, fakes
    only where a live NVIDIA connection would otherwise be required."""

    def __init__(self, tmp_path: Path, qdrant_client: AsyncQdrantClient) -> None:
        self.settings = _FakeSettings()
        bootstrap_loaders()
        self.loader_registry = loader_registry

        self.vector_store = QdrantVectorStore(qdrant_client)
        self.embedder = _FakeEmbedder()
        self.llm = _FakeLLM()

        self.hash_registry = SqliteHashRegistry(tmp_path / "hashes.sqlite")
        self.kb_store = SqliteKnowledgeBaseStore(tmp_path / "kb.sqlite")
        self.kb_manager = KnowledgeBaseManager(self.kb_store, self.vector_store)
        self.raw_file_store = LocalRawFileStore(tmp_path / "raw")

        self.sparse_retriever = BM25SparseRetriever(self.vector_store)
        self.reranker = LocalBM25Reranker()
        self.hybrid_retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            sparse_retriever=self.sparse_retriever,
            reranker=self.reranker,
            alpha=0.5,
        )
        self.pipeline = IngestionPipeline(
            chunker=RecursiveChunker(chunk_size_tokens=100, overlap_tokens=10)
        )

    def build_indexer(self, loader) -> IncrementalIndexer:
        indexer = IncrementalIndexer(
            loader=loader,
            pipeline=self.pipeline,
            embedder=self.embedder,
            vector_store=self.vector_store,
            hash_registry=self.hash_registry,
        )
        indexer.add_invalidate_callback(self.sparse_retriever.invalidate)
        return indexer


@pytest.fixture
async def client(tmp_path):
    qdrant_client = AsyncQdrantClient(location=":memory:")
    app = FastAPI()
    app.include_router(knowledge_bases.router)
    app.include_router(ingestion.router)
    app.include_router(query.router)
    app.state.container = _TestContainer(tmp_path, qdrant_client)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await qdrant_client.close()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Knowledge base CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_get_knowledge_base(client):
    resp = await client.post("/knowledge-bases", json={"name": "Test KB"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Test KB"
    assert body["embedding_dimensions"] == 4

    get_resp = await client.get(f"/knowledge-bases/{body['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == body["id"]


@pytest.mark.asyncio
async def test_get_nonexistent_knowledge_base_404(client):
    resp = await client.get("/knowledge-bases/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_knowledge_bases(client):
    await client.post("/knowledge-bases", json={"name": "KB1"})
    await client.post("/knowledge-bases", json={"name": "KB2"})
    resp = await client.get("/knowledge-bases")
    assert resp.status_code == 200
    names = {kb["name"] for kb in resp.json()}
    assert {"KB1", "KB2"}.issubset(names)


@pytest.mark.asyncio
async def test_delete_knowledge_base(client):
    create_resp = await client.post("/knowledge-bases", json={"name": "Temp"})
    kb_id = create_resp.json()["id"]

    delete_resp = await client.delete(f"/knowledge-bases/{kb_id}")
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/knowledge-bases/{kb_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_knowledge_base_404(client):
    resp = await client.delete("/knowledge-bases/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_file_into_knowledge_base(client):
    create_resp = await client.post("/knowledge-bases", json={"name": "Ingest KB"})
    kb_id = create_resp.json()["id"]

    file_content = (FIXTURES / "sample.txt").read_bytes()
    resp = await client.post(
        f"/knowledge-bases/{kb_id}/ingest/file",
        files={"file": ("sample.txt", file_content, "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["skipped"] is False
    assert body["chunk_count"] > 0


@pytest.mark.asyncio
async def test_ingest_file_into_nonexistent_kb_404(client):
    file_content = b"hello"
    resp = await client.post(
        "/knowledge-bases/nonexistent/ingest/file",
        files={"file": ("x.txt", file_content, "text/plain")},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_reingesting_same_file_is_skipped(client):
    create_resp = await client.post("/knowledge-bases", json={"name": "Ingest KB 2"})
    kb_id = create_resp.json()["id"]
    file_content = (FIXTURES / "sample.txt").read_bytes()

    first = await client.post(
        f"/knowledge-bases/{kb_id}/ingest/file",
        files={"file": ("sample.txt", file_content, "text/plain")},
    )
    second = await client.post(
        f"/knowledge-bases/{kb_id}/ingest/file",
        files={"file": ("sample.txt", file_content, "text/plain")},
    )
    assert first.json()["skipped"] is False
    assert second.json()["skipped"] is True


@pytest.mark.asyncio
async def test_ingest_url(client, monkeypatch):
    from datetime import datetime

    from docintel.ingestion.loaders.web_loader import FetchedPage, WebPageFetcher

    async def fake_fetch(self, url: str) -> FetchedPage:
        return FetchedPage(
            url=url,
            html="<html><head><title>T</title></head><body><p>Fetched content here.</p></body></html>",
            mime_type="text/html",
            content_length=100,
            last_modified=datetime(2026, 1, 1, tzinfo=UTC),
        )

    monkeypatch.setattr(WebPageFetcher, "fetch", fake_fetch)

    create_resp = await client.post("/knowledge-bases", json={"name": "URL KB"})
    kb_id = create_resp.json()["id"]

    resp = await client.post(
        f"/knowledge-bases/{kb_id}/ingest/url", json={"url": "https://example.com/page"}
    )
    assert resp.status_code == 200
    assert resp.json()["chunk_count"] > 0


# ---------------------------------------------------------------------------
# Query (streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_streams_sse_response_with_citations(client):
    create_resp = await client.post("/knowledge-bases", json={"name": "Query KB"})
    kb_id = create_resp.json()["id"]

    file_content = (FIXTURES / "sample.txt").read_bytes()
    await client.post(
        f"/knowledge-bases/{kb_id}/ingest/file",
        files={"file": ("sample.txt", file_content, "text/plain")},
    )

    resp = await client.post(f"/knowledge-bases/{kb_id}/query", json={"query": "plain text"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = [
        json.loads(line[len("data: "):])
        for line in resp.text.split("\n\n")
        if line.startswith("data: ")
    ]
    assert events[0]["text"] == "Hello "
    assert events[-1]["is_final"] is True
    assert len(events[-1]["citations"]) >= 1


@pytest.mark.asyncio
async def test_query_nonexistent_kb_404(client):
    resp = await client.post("/knowledge-bases/nonexistent/query", json={"query": "hi"})
    assert resp.status_code == 404
