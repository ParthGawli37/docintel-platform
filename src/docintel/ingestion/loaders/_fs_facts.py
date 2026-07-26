"""
Shared helpers for gathering real, observable filesystem facts (mime
type, size, timestamps) about a local file. Used by every local-file
loader so metadata capture is consistent and never duplicated/invented
per-loader.

Only reports what the OS actually exposes:
- file_size_bytes: exact, from stat().st_size
- modified_at: from stat().st_mtime (reliable across platforms)
- created_at: intentionally left None on POSIX systems, where st_ctime is
  metadata-change time, not creation time, and reporting it as "created"
  would be misleading/incorrect rather than merely approximate.
"""

from __future__ import annotations

import mimetypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def guess_mime_type(path: Path) -> str | None:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type


def gather_fs_facts(path: Path) -> dict[str, Any]:
    """
    Return a dict of fields matching RawDocument's fs-derived attributes:
    file_size_bytes, modified_at. Safe to ** -unpack into RawDocument(...).
    """
    try:
        stat = path.stat()
    except OSError:
        return {"file_size_bytes": None, "modified_at": None}

    return {
        "file_size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC),
    }
