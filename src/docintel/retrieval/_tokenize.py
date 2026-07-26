"""Shared BM25 tokenization, used by both the sparse retriever and the local reranker."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+")


def _light_stem(word: str) -> str:
    """
    Minimal, well-known suffix stripping (not a full stemmer like Porter)
    so simple plural/verb-form variants (cat/cats, box/boxes) share a
    token for BM25 matching. Deliberately conservative to avoid
    over-merging distinct words.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("xes", "ches", "shes", "ses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [_light_stem(w) for w in _TOKEN_RE.findall(text.lower())]
