"""
Cleaners: strip noise from raw content before normalization/chunking.

Different source types need different cleaning strategies -- HTML needs
tag/script/style stripping, everything else needs lighter control-
character/boilerplate cleanup. The pipeline (ingestion/pipeline.py)
selects which Cleaner to use per RawDocument.source_type; neither cleaner
needs to know about the other or about source_type itself, keeping each
one single-purpose.
"""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

# Tags that are never meaningful document content -- stripped before extraction.
_HTML_NOISE_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "svg", "form")

# Control characters (excluding \t \n \r) that occasionally leak in from
# poorly-encoded sources and add no semantic value.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class DefaultCleaner:
    """General-purpose cleaner for plain text / markdown / office-extracted content."""

    def clean(self, content: str) -> str:
        text = unicodedata.normalize("NFKC", content)
        text = _CONTROL_CHAR_RE.sub("", text)
        return text


class HtmlCleaner:
    """
    Strips non-content HTML tags and extracts readable text. Used for
    both HTML and WEBSITE source types, since a fetched web page and a
    local .html file share the same cleaning needs.
    """

    def clean(self, content: str) -> str:
        soup = BeautifulSoup(content, "html.parser")
        for tag_name in _HTML_NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = _CONTROL_CHAR_RE.sub("", text)
        return text
