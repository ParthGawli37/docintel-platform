"""
Normalizer: collapses whitespace and blank-line runs into a consistent
form so identical semantic content produces identical text (and thus a
stable content_hash) regardless of incidental formatting differences
from the source format.
"""

from __future__ import annotations

import re

_TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


class WhitespaceNormalizer:
    """Default normalizer: whitespace/blank-line collapsing only -- no semantic changes."""

    def normalize(self, content: str) -> str:
        text = _TRAILING_WHITESPACE_RE.sub("\n", content)
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
        return text.strip()
