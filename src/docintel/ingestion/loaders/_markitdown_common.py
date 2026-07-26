"""
Shared conversion logic for loaders backed by MarkItDown (PDF, DOCX,
PPTX, XLSX). Retained from the original converter script's approach --
MarkItDown handles the binary parsing; this module just wraps it into
the Loader/RawDocument contract.

Not itself a loader (no LoaderPlugin subclass here) -- office_loaders.py's
PdfLoader/DocxLoader/PptxLoader/XlsxLoader each call
`convert_with_markitdown` and supply their own SourceType.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from markitdown import MarkItDown

from docintel.core.logging import get_logger
from docintel.core.models import RawDocument, SourceType
from docintel.ingestion.loaders._fs_facts import gather_fs_facts, guess_mime_type

logger = get_logger(__name__)

_markitdown = MarkItDown()


def _convert_sync(path: Path) -> str:
    result = _markitdown.convert(str(path))
    return str(result.text_content)


async def convert_with_markitdown(
    source: str | Path,
    knowledge_base_id: str,
    source_type: SourceType,
) -> list[RawDocument]:
    """
    Convert a single office file to a RawDocument via MarkItDown.

    MarkItDown's conversion is synchronous/CPU-bound, so it's offloaded
    to a thread to keep the loader interface (`async def load`) non-blocking.
    No hashing happens here -- that's the processing stage's job, applied
    to cleaned/normalized content.
    """
    path = Path(source)
    logger.info("markitdown_convert_start", path=str(path), source_type=source_type.value)

    try:
        text_content = await asyncio.to_thread(_convert_sync, path)
    except Exception as exc:
        logger.error("markitdown_convert_failed", path=str(path), error=str(exc))
        raise

    fs_facts = gather_fs_facts(path)
    document = RawDocument(
        content=text_content,
        source_uri=str(path),
        source_type=source_type,
        knowledge_base_id=knowledge_base_id,
        mime_type=guess_mime_type(path),
        **fs_facts,
    )
    logger.info("markitdown_convert_complete", path=str(path), content_length=len(text_content))
    return [document]
