"""
KnowledgeBaseManager: business-logic layer for creating/listing/deleting
knowledge bases. Combines SqliteKnowledgeBaseStore (config persistence)
with VectorStore (collection provisioning), so callers never have to
remember to do both -- create_knowledge_base() always leaves the config
record AND the underlying collection in sync.
"""

from __future__ import annotations

from docintel.core.interfaces import VectorStore
from docintel.core.logging import get_logger
from docintel.core.models import KnowledgeBase
from docintel.storage.kb_store import SqliteKnowledgeBaseStore

logger = get_logger(__name__)


class KnowledgeBaseNotFoundError(Exception):
    def __init__(self, kb_id: str) -> None:
        super().__init__(f"Knowledge base not found: {kb_id}")
        self.kb_id = kb_id


class KnowledgeBaseManager:
    def __init__(self, store: SqliteKnowledgeBaseStore, vector_store: VectorStore) -> None:
        self._store = store
        self._vector_store = vector_store

    async def create(
        self,
        name: str,
        embedding_model_id: str,
        embedding_dimensions: int,
        description: str | None = None,
        system_prompt: str | None = None,
    ) -> KnowledgeBase:
        kb = KnowledgeBase(
            name=name,
            description=description,
            embedding_model_id=embedding_model_id,
            embedding_dimensions=embedding_dimensions,
            system_prompt=system_prompt,
        )
        await self._vector_store.ensure_collection(kb.id, embedding_dimensions)
        await self._store.save(kb)
        logger.info("knowledge_base_created", kb_id=kb.id, name=kb.name)
        return kb

    async def get(self, kb_id: str) -> KnowledgeBase:
        kb = await self._store.get(kb_id)
        if kb is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        return kb

    async def list_all(self) -> list[KnowledgeBase]:
        return await self._store.list_all()

    async def delete(self, kb_id: str) -> None:
        existing = await self._store.get(kb_id)
        if existing is None:
            raise KnowledgeBaseNotFoundError(kb_id)
        await self._vector_store.drop_collection(kb_id)
        await self._store.delete(kb_id)
        logger.info("knowledge_base_deleted", kb_id=kb_id)
