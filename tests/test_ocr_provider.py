from pathlib import Path

import pytest

from docintel.core.interfaces import OCRProvider
from docintel.ingestion.loaders.image_ocr_loader import ImageOcrLoader
from docintel.ingestion.ocr.tesseract_provider import TesseractOCRProvider

FIXTURES = Path(__file__).parent / "fixtures"


def test_tesseract_provider_satisfies_ocr_protocol():
    assert isinstance(TesseractOCRProvider(), OCRProvider)


@pytest.mark.asyncio
async def test_tesseract_provider_extracts_real_text():
    provider = TesseractOCRProvider()
    text = await provider.extract_text(FIXTURES / "sample.png")
    assert "Test" in text


class FakeOCRProvider:
    provider_name = "fake"

    async def extract_text(self, image_path: Path) -> str:
        return "fake extracted text"


def test_fake_ocr_provider_satisfies_protocol():
    assert isinstance(FakeOCRProvider(), OCRProvider)


@pytest.mark.asyncio
async def test_image_loader_uses_injected_ocr_provider():
    """Proves ImageOcrLoader depends on the OCRProvider abstraction, not
    Tesseract directly -- swapping providers requires no loader changes."""
    loader = ImageOcrLoader(ocr_provider=FakeOCRProvider())
    docs = await loader.load(FIXTURES / "sample.png", knowledge_base_id="kb-test")
    assert docs[0].content == "fake extracted text"
    assert docs[0].extra["ocr_provider"] == "fake"
