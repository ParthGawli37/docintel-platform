"""
End-to-end loader tests against real fixture files in tests/fixtures/.
No mocking of MarkItDown/BeautifulSoup/pytesseract -- these exercise the
actual conversion paths. Loaders now return RawDocument (extraction-only,
no hashing, no HTML cleaning) per the staged pipeline design.
"""

from pathlib import Path

import pytest

from docintel.core.models import SourceType
from docintel.ingestion.loaders import bootstrap_loaders
from docintel.ingestion.loaders.base import registry

FIXTURES = Path(__file__).parent / "fixtures"
KB_ID = "kb-test"


@pytest.fixture(autouse=True, scope="module")
def _bootstrap():
    bootstrap_loaders()


def test_all_expected_loaders_registered():
    registered_extensions = {
        ext for loader in registry.all() for ext in loader.supported_extensions
    }
    expected = {
        ".pdf", ".docx", ".pptx", ".xlsx",
        ".txt", ".md", ".markdown", ".csv",
        ".html", ".htm",
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",
    }
    assert expected.issubset(registered_extensions)
    assert any(loader.handles_urls for loader in registry.all())


@pytest.mark.asyncio
async def test_txt_loader_extracts_raw_content_and_fs_facts():
    loader = registry.resolve(FIXTURES / "sample.txt")
    docs = await loader.load(FIXTURES / "sample.txt", KB_ID)
    assert len(docs) == 1
    doc = docs[0]
    assert "plain text fixture" in doc.content
    assert doc.source_type is SourceType.TXT
    assert doc.knowledge_base_id == KB_ID
    assert doc.file_size_bytes is not None and doc.file_size_bytes > 0
    assert doc.modified_at is not None
    assert doc.mime_type == "text/plain"


@pytest.mark.asyncio
async def test_markdown_loader():
    loader = registry.resolve(FIXTURES / "sample.md")
    docs = await loader.load(FIXTURES / "sample.md", KB_ID)
    assert "Sample Markdown" in docs[0].content
    assert docs[0].source_type is SourceType.MARKDOWN


@pytest.mark.asyncio
async def test_csv_loader_renders_markdown_table():
    loader = registry.resolve(FIXTURES / "sample.csv")
    docs = await loader.load(FIXTURES / "sample.csv", KB_ID)
    content = docs[0].content
    assert content.startswith("| name | role | team |")
    assert "Alice" in content
    assert docs[0].source_type is SourceType.CSV


@pytest.mark.asyncio
async def test_html_loader_returns_raw_markup_unstripped():
    """The loader must NOT clean HTML -- that's the processing stage's job."""
    loader = registry.resolve(FIXTURES / "sample.html")
    docs = await loader.load(FIXTURES / "sample.html", KB_ID)
    content = docs[0].content
    assert "<script>" in content       # raw markup preserved
    assert "<footer>" in content
    assert docs[0].title == "Sample Page"  # title is a directly-observable fact
    assert docs[0].source_type is SourceType.HTML


@pytest.mark.asyncio
async def test_docx_loader():
    loader = registry.resolve(FIXTURES / "sample.docx")
    docs = await loader.load(FIXTURES / "sample.docx", KB_ID)
    assert "Sample DOCX" in docs[0].content
    assert docs[0].source_type is SourceType.DOCX
    assert docs[0].file_size_bytes is not None


@pytest.mark.asyncio
async def test_pptx_loader():
    loader = registry.resolve(FIXTURES / "sample.pptx")
    docs = await loader.load(FIXTURES / "sample.pptx", KB_ID)
    assert "Sample PPTX" in docs[0].content
    assert docs[0].source_type is SourceType.PPTX


@pytest.mark.asyncio
async def test_xlsx_loader():
    loader = registry.resolve(FIXTURES / "sample.xlsx")
    docs = await loader.load(FIXTURES / "sample.xlsx", KB_ID)
    assert "Alice" in docs[0].content
    assert docs[0].source_type is SourceType.XLSX


@pytest.mark.asyncio
async def test_pdf_loader():
    loader = registry.resolve(FIXTURES / "sample.pdf")
    docs = await loader.load(FIXTURES / "sample.pdf", KB_ID)
    assert "Sample PDF fixture" in docs[0].content
    assert docs[0].source_type is SourceType.PDF


@pytest.mark.asyncio
async def test_image_ocr_loader_uses_default_tesseract_provider():
    loader = registry.resolve(FIXTURES / "sample.png")
    docs = await loader.load(FIXTURES / "sample.png", KB_ID)
    assert "Test" in docs[0].content
    assert docs[0].source_type is SourceType.IMAGE
    assert docs[0].extra["ocr_provider"] == "tesseract"


@pytest.mark.asyncio
async def test_web_loader_resolves_for_url():
    loader = registry.resolve("https://example.com")
    assert loader.handles_urls is True


def test_unregistered_extension_raises():
    with pytest.raises(ValueError, match="No registered loader"):
        registry.resolve(FIXTURES / "sample.exotic")
