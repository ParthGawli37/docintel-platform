"""
SqliteKnowledgeBaseStore: persists KnowledgeBase records.

Kept in storage/ alongside the other SQLite-backed stores (cache_store,
hash_registry) for consistency -- knowledge_base/manager.py is the
business-logic layer on top of this that also provisions the underlying
VectorStore collection; this class only persists the config record.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from docintel.core.models import KnowledgeBase
from docintel.storage._sqlite_utils import get_connection

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    embedding_model_id TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    system_prompt TEXT,
    created_at TEXT NOT NULL
);
"""


_KBRow = tuple[str, str, str | None, str, int, str | None, str]


class SqliteKnowledgeBaseStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = get_connection(db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def _row_to_kb(self, row: _KBRow) -> KnowledgeBase:
        return KnowledgeBase(
            id=row[0],
            name=row[1],
            description=row[2],
            embedding_model_id=row[3],
            embedding_dimensions=row[4],
            system_prompt=row[5],
            created_at=datetime.fromisoformat(row[6]),
        )

    def _save_sync(self, kb: KnowledgeBase) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO knowledge_bases "
            "(id, name, description, embedding_model_id, embedding_dimensions, "
            "system_prompt, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                kb.id,
                kb.name,
                kb.description,
                kb.embedding_model_id,
                kb.embedding_dimensions,
                kb.system_prompt,
                kb.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def _get_sync(self, kb_id: str) -> KnowledgeBase | None:
        cursor = self._conn.execute(
            "SELECT id, name, description, embedding_model_id, embedding_dimensions, "
            "system_prompt, created_at FROM knowledge_bases WHERE id = ?",
            (kb_id,),
        )
        row = cursor.fetchone()
        return self._row_to_kb(row) if row else None

    def _list_sync(self) -> list[KnowledgeBase]:
        cursor = self._conn.execute(
            "SELECT id, name, description, embedding_model_id, embedding_dimensions, "
            "system_prompt, created_at FROM knowledge_bases ORDER BY created_at"
        )
        return [self._row_to_kb(row) for row in cursor.fetchall()]

    def _delete_sync(self, kb_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    async def save(self, kb: KnowledgeBase) -> None:
        await asyncio.to_thread(self._save_sync, kb)

    async def get(self, kb_id: str) -> KnowledgeBase | None:
        return await asyncio.to_thread(self._get_sync, kb_id)

    async def list_all(self) -> list[KnowledgeBase]:
        return await asyncio.to_thread(self._list_sync)

    async def delete(self, kb_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, kb_id)

    def close(self) -> None:
        self._conn.close()
