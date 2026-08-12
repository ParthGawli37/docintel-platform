"""
QdrantVectorStore: default VectorStore implementation.

Each `collection` argument corresponds 1:1 with a knowledge base id, per
the collection-based knowledge base design -- this class has no concept
of "which assistant" it's serving, only which collection.

Points are keyed by Chunk.id (a UUID string), with the full DocumentMetadata
plus chunk content stored in the payload so search results can be
reconstructed into a Chunk without a second lookup.
"""

from __future__ import annotations

from typing import cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from docintel.core.logging import get_logger
from docintel.core.models import Chunk, DocumentMetadata, EmbeddedChunk, SearchResult

logger = get_logger(__name__)

_CONTENT_HASH_SCROLL_LIMIT = 512


class QdrantVectorStore:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def ensure_collection(self, collection: str, dimensions: int) -> None:
        exists = await self._client.collection_exists(collection)
        if exists:
            return
        await self._client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
        )
        logger.info("qdrant_collection_created", collection=collection, dimensions=dimensions)

    async def drop_collection(self, collection: str) -> None:
        exists = await self._client.collection_exists(collection)
        if exists:
            await self._client.delete_collection(collection)
            logger.info("qdrant_collection_dropped", collection=collection)

    async def upsert(self, collection: str, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        points = [
            PointStruct(
                id=ec.chunk.id,
                vector=ec.vector,
                payload=_chunk_to_payload(ec.chunk),
            )
            for ec in embedded_chunks
        ]
        await self._client.upsert(collection_name=collection, points=points)
        logger.info("qdrant_upsert", collection=collection, point_count=len(points))

    async def delete_by_document_id(self, collection: str, document_id: str) -> None:
        await self._client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        logger.info("qdrant_delete_by_document", collection=collection, document_id=document_id)

    async def delete_by_source_uri(self, collection: str, source_uri: str) -> int:
        """Delete all vectors for one source using a Qdrant payload filter."""
        exists = await self._client.collection_exists(collection)
        if not exists:
            return 0

        source_filter = Filter(
            must=[FieldCondition(key="source_uri", match=MatchValue(value=source_uri))]
        )
        count = await self._client.count(
            collection_name=collection,
            count_filter=source_filter,
            exact=True,
        )
        if count.count == 0:
            return 0

        await self._client.delete(
            collection_name=collection,
            points_selector=source_filter,
        )
        logger.info(
            "qdrant_delete_by_source",
            collection=collection,
            source_uri=source_uri,
            point_count=count.count,
        )
        return count.count

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        query_filter = _build_filter(filters) if filters else None
        response = await self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [_point_to_search_result(point) for point in response.points]

    async def get_existing_content_hashes(self, collection: str) -> set[str]:
        chunks = await self.get_all_chunks(collection)
        return {c.metadata.content_hash for c in chunks}

    async def get_all_chunks(self, collection: str) -> list[Chunk]:
        exists = await self._client.collection_exists(collection)
        if not exists:
            return []

        chunks: list[Chunk] = []
        offset = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=collection,
                limit=_CONTENT_HASH_SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload:
                    chunks.append(_payload_to_chunk(str(record.id), dict(record.payload)))
            if offset is None:
                break
        return chunks


def _chunk_to_payload(chunk: Chunk) -> dict[str, object]:
    metadata = chunk.metadata
    return {
        "document_id": chunk.document_id,
        "content": chunk.content,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "source_uri": metadata.source_uri,
        "source_type": metadata.source_type.value,
        "title": metadata.title,
        "author": metadata.author,
        "content_hash": metadata.content_hash,
        "knowledge_base_id": metadata.knowledge_base_id,
        "mime_type": metadata.mime_type,
        "file_size_bytes": metadata.file_size_bytes,
        "page_count": metadata.page_count,
        "language": metadata.language,
        "extra": metadata.extra,
    }


def _payload_to_chunk(point_id: str, payload: dict[str, object]) -> Chunk:
    metadata = DocumentMetadata(
        source_uri=str(payload["source_uri"]),
        source_type=payload["source_type"],  # type: ignore[arg-type]  # StrEnum accepts the raw str value
        title=payload.get("title"),  # type: ignore[arg-type]
        author=payload.get("author"),  # type: ignore[arg-type]
        content_hash=str(payload["content_hash"]),
        knowledge_base_id=str(payload["knowledge_base_id"]),
        mime_type=payload.get("mime_type"),  # type: ignore[arg-type]
        file_size_bytes=payload.get("file_size_bytes"),  # type: ignore[arg-type]
        page_count=payload.get("page_count"),  # type: ignore[arg-type]
        language=payload.get("language"),  # type: ignore[arg-type]
        extra=payload.get("extra") or {},  # type: ignore[arg-type]
    )
    return Chunk(
        id=point_id,
        document_id=str(payload["document_id"]),
        content=str(payload["content"]),
        chunk_index=int(payload["chunk_index"]),  # type: ignore[call-overload]
        metadata=metadata,
        token_count=payload.get("token_count"),  # type: ignore[arg-type]
    )


def _point_to_search_result(point: ScoredPoint) -> SearchResult:
    chunk = _payload_to_chunk(str(point.id), dict(point.payload or {}))
    return SearchResult(chunk=chunk, score=float(point.score))


def _build_filter(filters: dict[str, object]) -> Filter:
    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=cast(bool | int | str, value)))
            for key, value in filters.items()
        ]
    )
