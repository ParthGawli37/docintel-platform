# docintel

A modular, domain-agnostic **Document Intelligence / RAG platform**. One
ingestion → chunking → embedding → retrieval → generation pipeline powers
any number of independent **knowledge bases**, each defined purely by
configuration (a name, an embedding model, a system prompt) — never by
domain-specific code. The same platform can serve a Portfolio AI
Assistant, a QA Documentation Assistant, or a Company Knowledge Base
simply by creating a different knowledge base and pointing it at different
source documents.

## Architecture

```
Source (file/URL)
    │
    ▼
Loader (plugin, per-format)          ── extraction only, no cleaning/hashing
    │  produces: RawDocument
    ▼
IngestionPipeline
    │  Cleaner (Default or Html) → Normalizer → MetadataExtractor (computes content_hash)
    │  produces: ProcessedDocument
    ▼
Chunker (Recursive / Structural / Semantic)
    │  produces: ChunkedDocument (list[Chunk])
    ▼
Embedder (NVIDIA, cache-wrapped)      ── skips re-embedding unchanged content
    │  produces: list[EmbeddedChunk]
    ▼
VectorStore (Qdrant, per-collection == per-knowledge-base)
    │
    ▼
HybridRetriever (dense + BM25 sparse, fused, reranked)
    │
    ▼
LLM (NVIDIA Nemotron, streaming) + CitationBuilder
    │
    ▼
StreamingResponse (SSE) with inline citations
```

Every arrow above is a `Protocol` defined in `core/interfaces.py`.
Business logic (the pipeline, the indexer, the retriever) depends only
on these protocols — concrete providers (NVIDIA, Qdrant, Tesseract) are
wired together in exactly one place: `api/container.py` (the composition
root). Swapping any provider means writing one new class and changing
one line in that file.

### Directory layout

```
src/docintel/
├── core/            # config, logging, interfaces (Protocols), shared data models
├── storage/         # SQLite-backed embedding cache, hash registry, KB store; local raw-file store
├── ingestion/
│   ├── loaders/      # one plugin per format (pdf, docx, pptx, xlsx, csv, txt, md, html, image, web)
│   ├── ocr/           # OCRProvider abstraction (Tesseract is the default)
│   ├── processing/     # Cleaner, Normalizer, MetadataExtractor
│   ├── chunking/        # RecursiveChunker (default), StructuralChunker, SemanticChunker
│   └── pipeline.py       # orchestrates Raw -> Processed -> Chunked
├── embeddings/      # NvidiaEmbedder, CachedEmbedder (decorator)
├── vectorstore/     # QdrantVectorStore
├── retrieval/       # BM25SparseRetriever, LocalBM25Reranker, HybridRetriever
├── generation/      # NvidiaNemotronLLM (streaming)
├── citations/       # DefaultCitationBuilder
├── evaluation/      # deterministic retrieval/citation evaluation metrics
├── knowledge_base/  # KnowledgeBaseManager (collection lifecycle)
├── indexing/        # IncrementalIndexer (hash-based skip logic)
└── api/             # FastAPI app, composition root, routers
```

### Design principles applied throughout

- **Protocol-first**: every swappable component (loader, chunker, embedder,
  vector store, reranker, LLM, OCR provider) is a `typing.Protocol` in
  `core/interfaces.py`. Nothing outside `api/container.py` imports a
  concrete provider class.
- **Staged, immutable ingestion pipeline**: `RawDocument` (extraction-only,
  no hashing) → `ProcessedDocument` (cleaned/normalized, hash computed
  here) → `ChunkedDocument`. Loaders never hash or clean; the processing
  stage never re-extracts.
- **Collection-based knowledge bases**: a "knowledge base" is a
  `KnowledgeBase` config record (name, embedding model, system prompt) +
  one Qdrant collection. No code branches on "which assistant" — only on
  "which collection."
- **Plugin loaders**: adding a new source format means writing one new
  `LoaderPlugin` subclass and adding one import line to
  `ingestion/loaders/__init__.py`. The pipeline orchestrator never
  hardcodes a format list.
- **Caching**: embeddings are cached by `(content_hash, model_id)`
  (`CachedEmbedder` + `SqliteEmbeddingCache`); indexing is incremental via
  a `(knowledge_base_id, source_uri) -> content_hash` registry
  (`SqliteHashRegistry`), so unchanged sources are skipped entirely.
- **Never-invented metadata**: file size, mime type, and modification
  time come from real filesystem/HTTP facts; `created_at` is left `None`
  on POSIX (documented why in `_fs_facts.py`) rather than reporting a
  misleading value. NVIDIA model IDs are required, unset-by-default
  config fields — the platform never guesses one.

## Setup

```bash
cp .env.example .env
# Fill in the TODO(user) markers: NVIDIA_API_KEY, NVIDIA_GENERATION_MODEL,
# NVIDIA_EMBEDDING_MODEL, NVIDIA_EMBEDDING_DIMENSIONS.

pip install -e ".[dev]"
pytest                      # full test suite (no live NVIDIA/Qdrant server needed --
                             # Qdrant tests run against an in-memory instance,
                             # NVIDIA client tests run against a mocked transport)
```

### Running locally with Docker

```bash
docker compose up --build
```

This starts a local Qdrant instance and the API, wired together
automatically (`QDRANT_URL` is overridden to point at the compose
service). The API is then available at `http://localhost:8000` — see
`http://localhost:8000/docs` for interactive OpenAPI docs.

### Running without Docker

Requires a running Qdrant instance (local or cloud) reachable at
`QDRANT_URL`:

```bash
uvicorn docintel.api.main:app --host 0.0.0.0 --port 8000
```

## API overview

- `POST /knowledge-bases` — create a knowledge base
- `GET /knowledge-bases` / `GET /knowledge-bases/{id}` — list / fetch
- `DELETE /knowledge-bases/{id}` — delete (drops the underlying collection too)
- `POST /knowledge-bases/{id}/ingest/file` — upload + incrementally index a file
- `POST /knowledge-bases/{id}/ingest/url` — fetch + incrementally index a web page
- `POST /knowledge-bases/{id}/query` — hybrid search + streaming generation (SSE),
  citations attached to the final event
- `GET /health` — process-level liveness check
- `GET /ready` — dependency-aware readiness check (currently validates Qdrant)

## Reliability and evaluation (V1.5)

- **Request observability**: every API request gets an `X-Request-ID` correlation
  ID and `X-Process-Time-Ms` response header; lifecycle events are emitted through
  structured logging.
- **Safe error handling**: unexpected API failures return a stable error envelope
  with a correlation ID instead of leaking provider internals.
- **Bounded retries**: NVIDIA generation requests use exponential backoff with a
  maximum of three attempts. Embedding requests already use the same policy.
- **Contract tests**: core provider protocols have structural compatibility tests
  so replacement implementations can be checked independently of the concrete provider.
- **Integration/API coverage**: Qdrant runs in memory for real vector-store integration
  tests, while the API suite exercises knowledge-base CRUD, ingestion and streaming query paths.
- **Deterministic RAG evaluation**: `docintel.evaluation` provides Recall@K,
  Precision@K, MRR, citation precision/recall, per-case evaluation and dataset-level
  macro-averaged reports without requiring an LLM judge.

Example evaluation usage:

```python
from docintel.evaluation import RetrievalCase, evaluate_dataset

report = evaluate_dataset([
    RetrievalCase(
        query="How does authentication work?",
        results=retrieved_results,
        relevant_document_ids={"auth-design-doc"},
        k=5,
    )
])

print(report.mean_recall_at_k, report.mean_reciprocal_rank)
```

## Testing notes

This sandbox's network allowlist includes PyPI/npm/GitHub but not
`api.nvidia.com` or a live Qdrant server. Consequently:

- **Qdrant** is tested against `AsyncQdrantClient(location=":memory:")` —
  a real, fully-functional Qdrant instance with no server process
  required. These are genuine integration tests, not mocks.
- **NVIDIA** (`NvidiaEmbedder`, `NvidiaNemotronLLM`) client code is
  written against the confirmed real `openai` SDK request/response
  shapes, but tested by injecting a fake client that mimics those shapes
  — not a live API call. Before deploying, confirm connectivity with
  real credentials (`NVIDIA_API_KEY` + the model IDs in `.env`).
- Everything else (loaders, processing pipeline, chunkers, storage layer,
  retrieval, indexing, the full API) is tested end-to-end with real I/O
  against real fixture files.
- GitHub Actions runs **Ruff + the full pytest suite + strict mypy** on pull
  requests and pushes to `main`.

## Known placeholders requiring user configuration

See `.env.example` for the full list; the ones that block real NVIDIA/Qdrant
connectivity are:

- `NVIDIA_GENERATION_MODEL`, `NVIDIA_EMBEDDING_MODEL`, `NVIDIA_EMBEDDING_DIMENSIONS`
- `QDRANT_URL` / `QDRANT_API_KEY` if using Qdrant Cloud instead of the bundled
  Docker Compose instance
