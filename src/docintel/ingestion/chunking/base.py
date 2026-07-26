"""
Shared helpers for chunker implementations.

Token counting here is a word-based approximation (not a real tokenizer
for any specific model) -- no exact tokenizer was specified for the
NVIDIA models in use, and guessing one would risk silently mismatching
the actual model's tokenization. This approximation is only used for
sizing chunks consistently; if precise token counts become necessary
later (e.g. to hit an exact context-window budget), swap in the real
tokenizer for whichever model is configured, at that call site only.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\S+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_token_count(text: str) -> int:
    """Rough approximation: ~0.75 tokens per word for English-like text."""
    word_count = len(_WORD_RE.findall(text))
    return max(1, round(word_count / 0.75)) if word_count else 0


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
