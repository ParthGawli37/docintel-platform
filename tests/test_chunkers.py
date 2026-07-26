import pytest

from docintel.core.models import DocumentMetadata, ProcessedDocument, SourceType
from docintel.ingestion.chunking.recursive_chunker import RecursiveChunker
from docintel.ingestion.chunking.semantic_chunker import SemanticChunker
from docintel.ingestion.chunking.structural_chunker import StructuralChunker


def _processed_document(content: str) -> ProcessedDocument:
    return ProcessedDocument(
        raw_document_id="raw-1",
        content=content,
        metadata=DocumentMetadata(
            source_uri="x.txt",
            source_type=SourceType.TXT,
            content_hash="abc123",
            knowledge_base_id="kb-1",
        ),
    )


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recursive_chunker_single_small_paragraph_is_one_chunk():
    doc = _processed_document("A short paragraph.")
    chunked = await RecursiveChunker(chunk_size_tokens=100, overlap_tokens=10).chunk(doc)
    assert len(chunked.chunks) == 1
    assert chunked.chunks[0].content == "A short paragraph."
    assert chunked.chunks[0].document_id == doc.id


@pytest.mark.asyncio
async def test_recursive_chunker_splits_multiple_paragraphs_when_over_budget():
    long_para = " ".join(["word"] * 200)
    doc = _processed_document(f"{long_para}\n\n{long_para}\n\n{long_para}")
    chunker = RecursiveChunker(chunk_size_tokens=50, overlap_tokens=5)
    chunked = await chunker.chunk(doc)
    assert len(chunked.chunks) > 1
    for chunk in chunked.chunks:
        assert chunk.token_count is not None


@pytest.mark.asyncio
async def test_recursive_chunker_never_exceeds_size_even_for_oversized_paragraph():
    huge_para = " ".join(["word"] * 5000)  # single paragraph, no natural break
    chunker = RecursiveChunker(chunk_size_tokens=50, overlap_tokens=5)
    doc = _processed_document(huge_para)
    chunked = await chunker.chunk(doc)
    assert len(chunked.chunks) > 1


@pytest.mark.asyncio
async def test_recursive_chunker_rejects_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size_tokens=10, overlap_tokens=10)


@pytest.mark.asyncio
async def test_recursive_chunker_empty_document_yields_no_chunks():
    doc = _processed_document("")
    chunked = await RecursiveChunker(chunk_size_tokens=100, overlap_tokens=10).chunk(doc)
    assert chunked.chunks == []


@pytest.mark.asyncio
async def test_recursive_chunker_chunk_indices_are_sequential():
    long_para = " ".join(["word"] * 200)
    doc = _processed_document(f"{long_para}\n\n{long_para}\n\n{long_para}")
    chunked = await RecursiveChunker(chunk_size_tokens=50, overlap_tokens=5).chunk(doc)
    indices = [c.chunk_index for c in chunked.chunks]
    assert indices == list(range(len(indices)))


# ---------------------------------------------------------------------------
# StructuralChunker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structural_chunker_splits_on_markdown_headers():
    content = "# Intro\nIntro text.\n\n# Details\nDetails text."
    doc = _processed_document(content)
    chunked = await StructuralChunker(chunk_size_tokens=100, overlap_tokens=10).chunk(doc)
    assert len(chunked.chunks) == 2
    assert "Intro" in chunked.chunks[0].content
    assert "Details" in chunked.chunks[1].content


@pytest.mark.asyncio
async def test_structural_chunker_falls_back_to_recursive_within_large_section():
    long_body = " ".join(["word"] * 200)
    content = f"# Big Section\n{long_body}"
    doc = _processed_document(content)
    chunked = await StructuralChunker(chunk_size_tokens=50, overlap_tokens=5).chunk(doc)
    assert len(chunked.chunks) > 1


@pytest.mark.asyncio
async def test_structural_chunker_handles_no_headers():
    content = "Just plain text with no markdown headers at all."
    doc = _processed_document(content)
    chunked = await StructuralChunker(chunk_size_tokens=100, overlap_tokens=10).chunk(doc)
    assert len(chunked.chunks) == 1
    assert chunked.chunks[0].content == content


@pytest.mark.asyncio
async def test_structural_chunker_preserves_preamble_before_first_header():
    content = "Preamble text.\n\n# Header\nBody text."
    doc = _processed_document(content)
    chunked = await StructuralChunker(chunk_size_tokens=100, overlap_tokens=10).chunk(doc)
    assert any("Preamble text." in c.content for c in chunked.chunks)


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


async def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """
    Deterministic fake embeddings: sentences containing 'cat' cluster near
    [1,0], sentences containing 'stock' cluster near [0,1] -- lets us
    assert the chunker actually groups by topic similarity.
    """
    vectors = []
    for t in texts:
        if "cat" in t.lower():
            vectors.append([1.0, 0.01])
        elif "stock" in t.lower():
            vectors.append([0.01, 1.0])
        else:
            vectors.append([0.5, 0.5])
    return vectors


@pytest.mark.asyncio
async def test_semantic_chunker_groups_similar_sentences_and_splits_on_topic_shift():
    content = (
        "The cat sat on the mat. The cat likes to sleep. "
        "The stock market rose today. The stock price increased."
    )
    doc = _processed_document(content)
    chunker = SemanticChunker(embed_fn=_fake_embed_fn, similarity_threshold=0.5)
    chunked = await chunker.chunk(doc)
    assert len(chunked.chunks) == 2
    assert "cat" in chunked.chunks[0].content.lower()
    assert "stock" in chunked.chunks[1].content.lower()


@pytest.mark.asyncio
async def test_semantic_chunker_single_sentence_document():
    doc = _processed_document("Just one sentence here.")
    chunker = SemanticChunker(embed_fn=_fake_embed_fn)
    chunked = await chunker.chunk(doc)
    assert len(chunked.chunks) == 1


@pytest.mark.asyncio
async def test_semantic_chunker_empty_document():
    doc = _processed_document("")
    chunker = SemanticChunker(embed_fn=_fake_embed_fn)
    chunked = await chunker.chunk(doc)
    assert chunked.chunks == []
