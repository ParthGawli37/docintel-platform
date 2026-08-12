from docintel.core.models import Citation, Chunk, DocumentMetadata, SearchResult, SourceType
from docintel.evaluation import CitationEvaluation, evaluate_retrieval


def _result(document_id: str, source_uri: str, score: float) -> SearchResult:
    metadata = DocumentMetadata(
        source_uri=source_uri,
        source_type=SourceType.TXT,
        content_hash=f"hash-{document_id}",
        knowledge_base_id="kb",
    )
    return SearchResult(
        chunk=Chunk(
            document_id=document_id,
            content=f"content {document_id}",
            chunk_index=0,
            metadata=metadata,
        ),
        score=score,
    )


def _citation(document_id: str, source_uri: str) -> Citation:
    return Citation(
        chunk_id=f"chunk-{document_id}",
        document_id=document_id,
        source_uri=source_uri,
        excerpt="excerpt",
        score=1.0,
    )


def test_retrieval_metrics():
    results = [
        _result("doc-a", "a.txt", 0.9),
        _result("doc-x", "x.txt", 0.8),
        _result("doc-b", "b.txt", 0.7),
    ]

    evaluation = evaluate_retrieval(results, {"doc-a", "doc-b"}, k=3)

    assert evaluation.recall_at_k == 1.0
    assert evaluation.precision_at_k == 2 / 3
    assert evaluation.reciprocal_rank == 1.0


def test_retrieval_metrics_return_zero_for_no_relevant_documents():
    evaluation = evaluate_retrieval([_result("doc-a", "a.txt", 1.0)], set(), k=5)
    assert evaluation.recall_at_k == 0.0
    assert evaluation.precision_at_k == 0.0
    assert evaluation.reciprocal_rank == 0.0


def test_citation_metrics():
    citations = [_citation("doc-a", "a.txt"), _citation("doc-x", "x.txt")]
    evaluation = CitationEvaluation.from_citations(citations, {"a.txt", "b.txt"})

    assert evaluation.precision == 0.5
    assert evaluation.recall == 0.5
