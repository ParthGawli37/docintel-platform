"""
Default OCRProvider implementation, backed by Tesseract via pytesseract.

Satisfies core.interfaces.OCRProvider structurally (no inheritance
required). To add a different backend (NVIDIA OCR, Azure OCR, Google
Vision, ...), write a new class with the same shape in a sibling module --
ImageOcrLoader depends only on the OCRProvider protocol, never on this
class directly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytesseract
from PIL import Image

from docintel.core.logging import get_logger

logger = get_logger(__name__)


def _ocr_sync(path: Path) -> str:
    with Image.open(path) as img:
        return str(pytesseract.image_to_string(img))


class TesseractOCRProvider:
    provider_name: str = "tesseract"

    async def extract_text(self, image_path: Path) -> str:
        logger.info("ocr_start", provider=self.provider_name, path=str(image_path))
        try:
            text = await asyncio.to_thread(_ocr_sync, image_path)
        except Exception as exc:
            logger.error("ocr_failed", provider=self.provider_name, path=str(image_path), error=str(exc))
            raise
        logger.info("ocr_complete", provider=self.provider_name, extracted_chars=len(text))
        return text
