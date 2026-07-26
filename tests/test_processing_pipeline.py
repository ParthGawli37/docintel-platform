from datetime import UTC

import pytest

from docintel.core.hashing import compute_content_hash
from docintel.core.models import RawDocument, SourceType
from docintel.ingestion.chunking.recursive_chunker import RecursiveChunker
from docintel.ingestion.pipeline import IngestionPipeline
from docintel.ingestion.processing.cleaner import DefaultCleaner, HtmlCleaner
from docintel.ingestion.processing.metadata_extractor import DefaultMetadataExtractor
from docintel.ingestion.processing.normalizer import WhitespaceNormalizer


def _pipeline() -> IngestionPipeline:
    return IngestionPipeline(chunker=RecursiveChunker(chunk_size_tokens=100, overlap_tokens=10))


def test_default_cleaner_strips_control_chars():
    cleaned = DefaultCleaner().clean("hello\x00world\x1f!")
    assert cleaned == "helloworld!"


def test_html_cleaner_strips_noise_tags_and_extracts_title_text():
    html = "<html><head><style>.x{}</style></head><body><script>bad()</script><p>Real content</p></body></html>"
    cleaned = HtmlCleaner().clean(html)
    assert "Real content" in cleaned
    assert "bad()" not in cleaned
    assert ".x{}" not in cleaned


def test_normalizer_collapses_whitespace():
    normalized = WhitespaceNormalizer().normalize("hello   world\n\n\n\nfoo")
    assert normalized == "hello world\n\nfoo"


def test_metadata_extractor_computes_hash_from_cleaned_content_not_raw():
    raw = RawDocument(
        content="RAW <script>noise</script> content",
        source_uri="x.html",
        source_type=SourceType.HTML,
        knowledge_base_id="kb-1",
    )
    cleaned_content = "cleaned content"
    metadata = DefaultMetadataExtractor().extract(raw, cleaned_content)

    assert metadata.content_hash == compute_content_hash(cleaned_content)
    assert metadata.content_hash != compute_content_hash(raw.content)


def test_pipeline_selects_html_cleaner_for_html_source_type():
    raw = RawDocument(
        content="<html><body><script>bad()</script><p>Hello world</p></body></html>",
        source_uri="page.html",
        source_type=SourceType.HTML,
        knowledge_base_id="kb-1",
    )
    processed = _pipeline().process(raw)
    assert "Hello world" in processed.content
    assert "bad()" not in processed.content


def test_pipeline_selects_default_cleaner_for_txt_source_type():
    raw = RawDocument(
        content="Plain   text\x00 with noise",
        source_uri="x.txt",
        source_type=SourceType.TXT,
        knowledge_base_id="kb-1",
    )
    processed = _pipeline().process(raw)
    assert "\x00" not in processed.content
    assert "Plain text with noise" == processed.content


def test_pipeline_hash_is_stable_for_identical_cleaned_content():
    raw_a = RawDocument(
        content="Same content.",
        source_uri="a.txt",
        source_type=SourceType.TXT,
        knowledge_base_id="kb-1",
    )
    raw_b = RawDocument(
        content="Same content.",
        source_uri="b.txt",  # different source, same content
        source_type=SourceType.TXT,
        knowledge_base_id="kb-1",
    )
    processed_a = _pipeline().process(raw_a)
    processed_b = _pipeline().process(raw_b)
    assert processed_a.metadata.content_hash == processed_b.metadata.content_hash


def test_pipeline_preserves_fs_facts_into_metadata():
    from datetime import datetime

    ts = datetime(2026, 1, 1, tzinfo=UTC)
    raw = RawDocument(
        content="hello",
        source_uri="x.txt",
        source_type=SourceType.TXT,
        knowledge_base_id="kb-1",
        file_size_bytes=1234,
        modified_at=ts,
        mime_type="text/plain",
    )
    processed = _pipeline().process(raw)
    assert processed.metadata.file_size_bytes == 1234
    assert processed.metadata.modified_at == ts
    assert processed.metadata.mime_type == "text/plain"


@pytest.mark.asyncio
async def test_process_and_chunk_end_to_end():
    raw = RawDocument(
        content="Paragraph one is here.\n\nParagraph two is here as well.",
        source_uri="x.txt",
        source_type=SourceType.TXT,
        knowledge_base_id="kb-1",
    )
    chunked = await _pipeline().process_and_chunk(raw)
    assert len(chunked.chunks) >= 1
    assert all(c.document_id == chunked.processed_document.id for c in chunked.chunks)
