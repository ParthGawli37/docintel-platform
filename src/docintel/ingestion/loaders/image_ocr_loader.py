"""
Image loader -- delegates text extraction to an injected OCRProvider
(Tesseract by default; see ingestion/ocr/). The loader itself only knows
how to gather filesystem facts and hand the image to the provider --
swapping OCR backends never requires touching this file.
"""

from __future__ import annotations

from pathlib import Path

from docintel.core.logging import get_logger
from docintel.core.models import RawDocument, SourceType
from docintel.ingestion.loaders._fs_facts import gather_fs_facts, guess_mime_type
from docintel.ingestion.loaders.base import LoaderPlugin, register_loader
from docintel.ingestion.ocr.tesseract_provider import TesseractOCRProvider

logger = get_logger(__name__)


@register_loader
class ImageOcrLoader(LoaderPlugin):
    supported_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")

    def __init__(self, ocr_provider: TesseractOCRProvider | None = None) -> None:
        # Default kept concrete (not the Protocol) here only because a
        # LoaderPlugin needs *some* zero-arg-constructible default for the
        # @register_loader decorator; callers that want a different
        # OCRProvider construct ImageOcrLoader(ocr_provider=...) directly
        # and register it themselves instead of relying on the decorator.
        self.ocr_provider = ocr_provider or TesseractOCRProvider()

    async def load(self, source: str | Path, knowledge_base_id: str) -> list[RawDocument]:
        path = Path(source)
        text = await self.ocr_provider.extract_text(path)
        fs_facts = gather_fs_facts(path)

        document = RawDocument(
            content=text,
            source_uri=str(path),
            source_type=SourceType.IMAGE,
            knowledge_base_id=knowledge_base_id,
            mime_type=guess_mime_type(path),
            extra={"ocr_provider": self.ocr_provider.provider_name},
            **fs_facts,
        )
        return [document]
