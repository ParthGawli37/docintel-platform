"""
Office-format loaders: PDF, DOCX, PPTX, XLSX.

All four share the same MarkItDown-based conversion path (see
_markitdown_common.py) -- this is the direct descendant of the original
convert_to_md.py script's core idea, now expressed as four independently
registered, swappable plugins instead of one hardcoded function.
"""

from __future__ import annotations

from pathlib import Path

from docintel.core.models import RawDocument, SourceType
from docintel.ingestion.loaders._markitdown_common import convert_with_markitdown
from docintel.ingestion.loaders.base import LoaderPlugin, register_loader


@register_loader
class PdfLoader(LoaderPlugin):
    supported_extensions = (".pdf",)

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        return await convert_with_markitdown(source, knowledge_base_id, SourceType.PDF)


@register_loader
class DocxLoader(LoaderPlugin):
    supported_extensions = (".docx",)

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        return await convert_with_markitdown(source, knowledge_base_id, SourceType.DOCX)


@register_loader
class PptxLoader(LoaderPlugin):
    supported_extensions = (".pptx",)

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        return await convert_with_markitdown(source, knowledge_base_id, SourceType.PPTX)


@register_loader
class XlsxLoader(LoaderPlugin):
    supported_extensions = (".xlsx",)

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        return await convert_with_markitdown(source, knowledge_base_id, SourceType.XLSX)
