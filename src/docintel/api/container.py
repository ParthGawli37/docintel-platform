"""
AppContainer: the composition root. Builds every concrete implementation
(NVIDIA, Qdrant, SQLite-backed stores, ...) exactly once from Settings,
and wires them together behind the Protocols defined in core/interfaces.py.

This is the ONLY module that imports concrete provider classes directly
everywhere else in the codebase depends on the Protocols. Swapping
Qdrant for another vector DB, or NVIDIA for another model provider, means
changing this file and nothing else.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from docintel.core.config import ChunkStrategy, Settings
from docintel.core.interfaces import Loader
from docintel.core.logging import configure_logging, get_logger
from docintel.embeddings.cached_embedder import CachedEmbedder
from docintel.embeddings.nvidia_embedder import NvidiaEmbedder
from docintel.generation.nvidia_nemotron import NvidiaNemotronLLM
from docintel.indexing.indexer import IncrementalIndexer
from docintel.ingestion.chunking.recursive_chunker import RecursiveChunker
from docintel.ingestion.chunking.structural_chunker import StructuralChunker
from docintel.ingestion.loaders import bootstrap_loaders
from docintel.ingestion.loaders.base import LoaderRegistry
from docintel.ingestion.loaders.base import registry as loader_registry
from docintel.ingestion.pipeline import IngestionPipeline
from docintel.knowledge_base.manager import KnowledgeBaseManager
from docintel.retrieval.hybrid_retriever import HybridRetriever
from docintel.retrieval.reranker import LocalBM25Reranker
from docintel.retrieval.sparse_retriever import BM25SparseRetriever
from docintel.storage.cache_store import SqliteEmbeddingCache
from docintel.storage.hash_registry import SqliteHashRegistry
from docintel.storage.kb_store import SqliteKnowledgeBaseStore
from docintel.storage.raw_store import LocalRawFileStore
from docintel.vectorstore.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


def _build_chunker(settings: Settings) -> RecursiveChunker | StructuralChunker:
    if settings.chunk_strategy is ChunkStrategy.STRUCTURAL:
        return StructuralChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    # SEMANTIC is intentionally not wired as a default here -- it needs an
    # embed_fn at construction time and costs an embedding call per
    # ingested document; a deployment that wants it opts in explicitly by
    # constructing AppContainer differently, rather than this composition
    # root silently making that cost/latency trade-off for everyone.
    return RecursiveChunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens)


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        configure_logging(settings)
        self.settings = settings

        bootstrap_loaders()
        self.loader_registry: LoaderRegistry = loader_registry

        self.qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self.vector_store = QdrantVectorStore(self.qdrant_client)

        self.embedding_cache = SqliteEmbeddingCache(settings.cache_dir / "embeddings.sqlite")
        self._raw_embedder = NvidiaEmbedder(settings)
        self.embedder = CachedEmbedder(self._raw_embedder, self.embedding_cache)

        self.llm = NvidiaNemotronLLM(settings)

        self.hash_registry = SqliteHashRegistry(settings.hash_registry_path)
        self.kb_store = SqliteKnowledgeBaseStore(settings.cache_dir / "knowledge_bases.sqlite")
        self.kb_manager = KnowledgeBaseManager(self.kb_store, self.vector_store)

        self.raw_file_store = LocalRawFileStore(settings.raw_files_dir)

        self.sparse_retriever = BM25SparseRetriever(self.vector_store)
        self.reranker = LocalBM25Reranker()
        self.hybrid_retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            sparse_retriever=self.sparse_retriever,
            reranker=self.reranker,
            alpha=settings.hybrid_search_alpha,
        )

        self.chunker = _build_chunker(settings)
        self.pipeline = IngestionPipeline(chunker=self.chunker)

        logger.info("app_container_initialized", app_env=settings.app_env.value)

    def build_indexer(self, loader: Loader) -> IncrementalIndexer:
        """
        A fresh IncrementalIndexer bound to a specific Loader -- loaders
        are per-source-type (see LoaderRegistry), so the indexer is
        constructed per ingest call after resolving the right loader for
        the given source, not held as a single container-wide singleton.
        """
        indexer = IncrementalIndexer(
            loader=loader,
            pipeline=self.pipeline,
            embedder=self.embedder,
            vector_store=self.vector_store,
            hash_registry=self.hash_registry,
        )
        indexer.add_invalidate_callback(self.sparse_retriever.invalidate)
        return indexer

    async def close(self) -> None:
        await self.qdrant_client.close()
        self.embedding_cache.close()
        self.hash_registry.close()
        self.kb_store.close()
        logger.info("app_container_closed")
