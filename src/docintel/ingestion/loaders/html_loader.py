"""
HTML loader for local .html/.htm files.

Extraction-only: this loader reads the raw HTML markup as-is. Stripping
noise tags (script/style/nav/footer) and extracting readable text is a
cleaning concern, not a loading concern -- see
ingestion/processing/cleaner.py:HtmlCleaner, which both this loader's
output and web_loader.py's output are cleaned by in the processing stage.

The only exception is `title`: an HTML <title> is a directly observable
fact of the source (like a filename), not a derived/cleaned value, so
loaders are allowed to read it -- consistent with MetadataExtractor's
role of *deriving* metadata versus a loader's role of *reading* trivially
exposed source facts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from bs4 import BeautifulSoup

from docintel.core.logging import get_logger
from docintel.core.models import RawDocument, SourceType
from docintel.ingestion.loaders._fs_facts import gather_fs_facts
from docintel.ingestion.loaders.base import LoaderPlugin, register_loader

logger = get_logger(__name__)


def _read_html_and_title_sync(path: Path) -> tuple[str, str | None]:
    html = path.read_text(encoding="utf-8", errors="replace")
    # Peeking at <title> is a cheap, directly-observable read -- not parsing
    # or cleaning the body content, so it stays consistent with "loaders
    # only extract what's trivially exposed by the source".
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    return html, title


@register_loader
class HtmlLoader(LoaderPlugin):
    supported_extensions = (".html", ".htm")

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        path = Path(source)
        raw_html, title = await asyncio.to_thread(_read_html_and_title_sync, path)
        fs_facts = gather_fs_facts(path)

        document = RawDocument(
            content=raw_html,
            source_uri=str(path),
            source_type=SourceType.HTML,
            knowledge_base_id=knowledge_base_id,
            title=title,
            mime_type="text/html",
            **fs_facts,
        )
        return [document]
