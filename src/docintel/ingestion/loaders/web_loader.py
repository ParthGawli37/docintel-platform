"""
Web loader for single-page URL ingestion.

Split into two pieces on purpose:

  WebPageFetcher -- single responsibility: fetch one URL, return raw HTML
                    + response headers. No knowledge of RawDocument/loader
                    contracts at all.
  WebLoader       -- LoaderPlugin that calls WebPageFetcher and packages
                    the result into a RawDocument (extraction-only, no
                    HTML cleaning -- see html_loader.py's docstring).

This separation is what makes future extension straightforward without
redesigning the loader architecture: a recursive crawler, robots.txt
compliance, rate limiting, canonical-URL resolution, duplicate detection,
and depth limits are all *orchestration* concerns that would sit in a new
module (e.g. ingestion/loaders/web_crawler.py) built on top of
WebPageFetcher -- they would call fetch() repeatedly with their own
policy, and only ever hand the final URL list to WebLoader/RawDocument
construction. None of that orchestration exists yet (nothing here should
be mistaken for it); this module intentionally fetches exactly one page,
today.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from docintel.core.logging import get_logger
from docintel.core.models import RawDocument, SourceType
from docintel.ingestion.loaders.base import LoaderPlugin, register_loader

logger = get_logger(__name__)


@dataclass(frozen=True)
class FetchedPage:
    """Raw result of fetching a single URL -- no RawDocument concepts here."""

    url: str
    html: str
    mime_type: str | None
    content_length: int | None
    last_modified: datetime | None


class WebPageFetcher:
    """
    Fetches a single URL over HTTP(S) and returns a FetchedPage.

    Deliberately has no concept of crawling, depth, or dedup -- those are
    orchestration concerns for a future caller, not this class's job.
    """

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self._timeout_seconds = timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def fetch(self, url: str) -> FetchedPage:
        async with httpx.AsyncClient(follow_redirects=True, timeout=self._timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type")
        mime_type = content_type.split(";")[0].strip() if content_type else None

        content_length_header = response.headers.get("content-length")
        content_length = (
            int(content_length_header)
            if content_length_header and content_length_header.isdigit()
            else len(response.content)
        )

        last_modified_header = response.headers.get("last-modified")
        last_modified: datetime | None = None
        if last_modified_header:
            try:
                last_modified = parsedate_to_datetime(last_modified_header)
            except (TypeError, ValueError):
                last_modified = None  # malformed header -- don't guess a date

        return FetchedPage(
            url=str(response.url),
            html=response.text,
            mime_type=mime_type,
            content_length=content_length,
            last_modified=last_modified,
        )


@register_loader
class WebLoader(LoaderPlugin):
    supported_extensions = ()
    handles_urls = True

    def __init__(self, fetcher: WebPageFetcher | None = None) -> None:
        self._fetcher = fetcher or WebPageFetcher()

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        url = str(source)
        logger.info("web_fetch_start", url=url)

        page = await self._fetcher.fetch(url)

        logger.info("web_fetch_complete", url=page.url, content_length=page.content_length)

        document = RawDocument(
            content=page.html,
            source_uri=page.url,
            source_type=SourceType.WEBSITE,
            knowledge_base_id=knowledge_base_id,
            mime_type=page.mime_type,
            file_size_bytes=page.content_length,
            modified_at=page.last_modified,
        )
        return [document]
