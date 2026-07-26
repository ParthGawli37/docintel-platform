"""
SqliteEmbeddingCache: default EmbeddingCache implementation.

Keyed by (content_hash, model_id) -- a cache hit means "this exact
normalized content has already been embedded by this exact model", which
is the only condition under which reusing a stored vector is correct.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from docintel.storage._sqlite_utils import get_connection

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (content_hash, model_id)
);
"""


class SqliteEmbeddingCache:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = get_connection(db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def _get_sync(self, content_hash: str, model_id: str) -> list[float] | None:
        cursor = self._conn.execute(
            "SELECT vector_json FROM embedding_cache WHERE content_hash = ? AND model_id = ?",
            (content_hash, model_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        result: list[float] = json.loads(row[0])
        return result

    def _set_sync(self, content_hash: str, model_id: str, vector: list[float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (content_hash, model_id, vector_json) "
            "VALUES (?, ?, ?)",
            (content_hash, model_id, json.dumps(vector)),
        )
        self._conn.commit()

    async def get(self, content_hash: str, model_id: str) -> list[float] | None:
        return await asyncio.to_thread(self._get_sync, content_hash, model_id)

    async def set(self, content_hash: str, model_id: str, vector: list[float]) -> None:
        await asyncio.to_thread(self._set_sync, content_hash, model_id, vector)

    def close(self) -> None:
        self._conn.close()
