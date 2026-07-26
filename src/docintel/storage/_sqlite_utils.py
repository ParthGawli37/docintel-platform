"""
Shared SQLite connection helper for the storage layer (cache_store,
hash_registry). SQLite is a deliberate choice here: both stores are
simple local key-value tables with no need for a network round-trip,
and WAL mode gives safe concurrent reads/writes from async code without
pulling in a separate service dependency.

All actual DB calls are synchronous sqlite3 calls run via
asyncio.to_thread, consistent with how the rest of the codebase (loaders)
handles blocking I/O.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn
