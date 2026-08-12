from pathlib import Path

from docintel.core.interfaces import CitationBuilder, Embedder, Loader, VectorStore
from docintel.core.models import Citation, Chunk, EmbeddedChunk, RawDocument, SearchResult


class _Loader:
    supported_extensions = (".txt",)

    def can_load(self, source: str | Path) -> bool:
        return str(source).endswith(".txt")

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        return []


class _Embedder:
    model_id = "contract-model"
    dimensions = 2

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return []

    async def embed_query(self, query: str) -> list[float]:
        return [0.0, 0.0]


class _VectorStore:
    async def ensure_collection(self, collection: str, dimensions: int) -> None:
        return None

    async def drop_collection(self, collection: str) -> None:
        return None

    async def upsert(self, collection: str, embedded_chunks: list[EmbeddedChunk]) -> None:
        return None

    async def delete_by_document_id(self, collection: str, document_id: str) -> None:
        return None

    async def delete_by_source_uri(self, collection: str, source_uri: str) -> int:
        return 0

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        return []

    async def get_existing_content_hashes(self, collection: str) -> set[str]:
        return set()

    async def get_all_chunks(self, collection: str) -> list[Chunk]:
        return []


class _CitationBuilder:
    def build(self, results: list[SearchResult]) -> list[Citation]:
        return []


def test_core_protocols_accept_structural_implementations():
    assert isinstance(_Loader(), Loader)
    assert isinstance(_Embedder(), Embedder)
    assert isinstance(_VectorStore(), VectorStore)
    assert isinstance(_CitationBuilder(), CitationBuilder)
