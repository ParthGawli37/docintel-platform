"""
SqliteHashRegistry: default HashRegistry implementation.

Tracks the last-indexed content_hash per (knowledge_base_id, source_uri),
so the incremental indexer can skip re-processing a source whose content
hasn't changed since the last run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from docintel.storage._sqlite_utils import get_connection

_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_hashes (
    knowledge_base_id TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (knowledge_base_id, source_uri)
);
"""


class SqliteHashRegistry:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = get_connection(db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def _get_hash_sync(self, knowledge_base_id: str, source_uri: str) -> str | None:
        cursor = self._conn.execute(
            "SELECT content_hash FROM document_hashes WHERE knowledge_base_id = ? AND source_uri = ?",
            (knowledge_base_id, source_uri),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _set_hash_sync(self, knowledge_base_id: str, source_uri: str, content_hash: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO document_hashes "
            "(knowledge_base_id, source_uri, content_hash, updated_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (knowledge_base_id, source_uri, content_hash),
        )
        self._conn.commit()

    async def get_hash(self, knowledge_base_id: str, source_uri: str) -> str | None:
        return await asyncio.to_thread(self._get_hash_sync, knowledge_base_id, source_uri)

    async def set_hash(self, knowledge_base_id: str, source_uri: str, content_hash: str) -> None:
        await asyncio.to_thread(self._set_hash_sync, knowledge_base_id, source_uri, content_hash)

    async def has_changed(
        self, knowledge_base_id: str, source_uri: str, current_hash: str
    ) -> bool:
        existing = await self.get_hash(knowledge_base_id, source_uri)
        return existing != current_hash

    def close(self) -> None:
        self._conn.close()
