"""
Plain-text-family loaders: CSV, TXT, Markdown.

These formats don't need MarkItDown's binary parsing -- CSV is rendered
into a Markdown table (so it chunks/reads consistently with everything
else the platform indexes), TXT and Markdown are read as-is. No hashing
here -- the processing stage hashes cleaned/normalized content.
"""

from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path

from docintel.core.logging import get_logger
from docintel.core.models import RawDocument, SourceType
from docintel.ingestion.loaders._fs_facts import gather_fs_facts, guess_mime_type
from docintel.ingestion.loaders.base import LoaderPlugin, register_loader

logger = get_logger(__name__)


def _read_text_sync(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _csv_to_markdown_table(raw: str) -> str:
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return ""

    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        # Guard against ragged rows rather than silently misaligning columns.
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def _build_raw_document(
    content: str, source: str | Path, source_type: SourceType, knowledge_base_id: str
) -> RawDocument:
    path = Path(source)
    fs_facts = gather_fs_facts(path)
    return RawDocument(
        content=content,
        source_uri=str(path),
        source_type=source_type,
        knowledge_base_id=knowledge_base_id,
        mime_type=guess_mime_type(path),
        **fs_facts,
    )


@register_loader
class TxtLoader(LoaderPlugin):
    supported_extensions = (".txt",)

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        path = Path(source)
        content = await asyncio.to_thread(_read_text_sync, path)
        return [_build_raw_document(content, path, SourceType.TXT, knowledge_base_id)]


@register_loader
class MarkdownLoader(LoaderPlugin):
    supported_extensions = (".md", ".markdown")

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        path = Path(source)
        content = await asyncio.to_thread(_read_text_sync, path)
        return [_build_raw_document(content, path, SourceType.MARKDOWN, knowledge_base_id)]


@register_loader
class CsvLoader(LoaderPlugin):
    supported_extensions = (".csv",)

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        path = Path(source)
        raw = await asyncio.to_thread(_read_text_sync, path)
        content = _csv_to_markdown_table(raw)
        return [_build_raw_document(content, path, SourceType.CSV, knowledge_base_id)]
