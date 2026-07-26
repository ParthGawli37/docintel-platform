"""
LocalRawFileStore: persists uploaded/ingested source files on the local
filesystem, organized per knowledge base.

Only the local backend is implemented -- STORAGE_BACKEND=s3 is a
configured-but-unimplemented placeholder (see .env.example) until an S3
implementation is actually requested; this class is intentionally the
only thing indexing/ingestion code depends on directly (never a bare
filesystem path), so adding an S3RawFileStore later means writing one new
class, not touching callers.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path


class LocalRawFileStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _kb_dir(self, knowledge_base_id: str) -> Path:
        kb_dir = self._base_dir / knowledge_base_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        return kb_dir

    def _save_sync(self, knowledge_base_id: str, filename: str, content: bytes) -> Path:
        dest = self._kb_dir(knowledge_base_id) / filename
        dest.write_bytes(content)
        return dest

    def _copy_sync(self, knowledge_base_id: str, source_path: Path) -> Path:
        dest = self._kb_dir(knowledge_base_id) / source_path.name
        shutil.copy2(source_path, dest)
        return dest

    async def save_bytes(self, knowledge_base_id: str, filename: str, content: bytes) -> Path:
        return await asyncio.to_thread(self._save_sync, knowledge_base_id, filename, content)

    async def copy_from_path(self, knowledge_base_id: str, source_path: Path) -> Path:
        return await asyncio.to_thread(self._copy_sync, knowledge_base_id, source_path)

    async def list_files(self, knowledge_base_id: str) -> list[Path]:
        def _list() -> list[Path]:
            kb_dir = self._kb_dir(knowledge_base_id)
            return sorted(p for p in kb_dir.iterdir() if p.is_file())

        return await asyncio.to_thread(_list)

    async def delete(self, knowledge_base_id: str, filename: str) -> bool:
        def _delete() -> bool:
            path = self._kb_dir(knowledge_base_id) / filename
            if path.exists():
                path.unlink()
                return True
            return False

        return await asyncio.to_thread(_delete)
