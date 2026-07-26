"""Deterministic content hashing, used for change detection and caching."""

from __future__ import annotations

import hashlib


def compute_content_hash(content: str) -> str:
    """
    Return a stable sha256 hex digest of the given text.

    Used by loaders (provisional hash of raw extracted content) and by
    the processing stage (final hash of cleaned/normalized content) to
    drive incremental indexing and embedding caching.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
