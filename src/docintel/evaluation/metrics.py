"""Provider-agnostic, deterministic RAG evaluation metrics.

These metrics intentionally operate on source/document identifiers rather than
model-generated text. They can therefore be used in CI with a fixed evaluation
set without requiring an LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from docintel.core.models import Citation, SearchResult


def _normalize_ids(values: set[str] | list[str] | tuple[str, ...]) -> set[str]:
    return {value for value in values if value}


def recall_at_k(results: list[SearchResult], relevant_document_ids: set[str], k: int) -> float:
    """Fraction of relevant documents retrieved in the first *k* results."""
    relevant = _normalize_ids(relevant_document_ids)
    if not relevant or k <= 0:
        return 0.0
    retrieved = {result.chunk.document_id for result in results[:k]}
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(results: list[SearchResult], relevant_document_ids: set[str], k: int) -> float:
    """Fraction of the first *k* results that are relevant."""
    if k <= 0:
        return 0.0
    relevant = _normalize_ids(relevant_document_ids)
    if not relevant:
        return 0.0
    top = results[:k]
    if not top:
        return 0.0
    return sum(result.chunk.document_id in relevant for result in top) / len(top)


def mean_reciprocal_rank(results: list[SearchResult], relevant_document_ids: set[str]) -> float:
    """Return reciprocal rank of the first relevant result, or zero."""
    relevant = _normalize_ids(relevant_document_ids)
    if not relevant:
        return 0.0
    for rank, result in enumerate(results, start=1):
        if result.chunk.document_id in relevant:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Aggregate deterministic retrieval metrics for one evaluation case."""

    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalCase:
    """Expected retrieval targets for one evaluation query."""

    query: str
    results: list[SearchResult]
    relevant_document_ids: set[str]
    k: int = 5

    def evaluate(self) -> RetrievalEvaluation:
        return evaluate_retrieval(self.results, self.relevant_document_ids, self.k)


@dataclass(frozen=True)
class EvaluationReport:
    """Macro-averaged retrieval report over a fixed evaluation dataset."""

    case_count: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_reciprocal_rank: float


def evaluate_retrieval(
    results: list[SearchResult], relevant_document_ids: set[str], k: int
) -> RetrievalEvaluation:
    return RetrievalEvaluation(
        recall_at_k=recall_at_k(results, relevant_document_ids, k),
        precision_at_k=precision_at_k(results, relevant_document_ids, k),
        reciprocal_rank=mean_reciprocal_rank(results, relevant_document_ids),
    )


def evaluate_dataset(cases: list[RetrievalCase]) -> EvaluationReport:
    """Evaluate a fixed retrieval dataset and macro-average its metrics."""
    if not cases:
        return EvaluationReport(0, 0.0, 0.0, 0.0)

    evaluations = [case.evaluate() for case in cases]
    count = len(evaluations)
    return EvaluationReport(
        case_count=count,
        mean_recall_at_k=sum(item.recall_at_k for item in evaluations) / count,
        mean_precision_at_k=sum(item.precision_at_k for item in evaluations) / count,
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in evaluations) / count,
    )


def citation_precision(citations: list[Citation], relevant_source_uris: set[str]) -> float:
    """Fraction of emitted citations whose source is expected/relevant."""
    relevant = _normalize_ids(relevant_source_uris)
    if not citations or not relevant:
        return 0.0
    return sum(citation.source_uri in relevant for citation in citations) / len(citations)


def citation_recall(citations: list[Citation], relevant_source_uris: set[str]) -> float:
    """Fraction of expected sources represented by at least one citation."""
    relevant = _normalize_ids(relevant_source_uris)
    if not relevant:
        return 0.0
    cited = {citation.source_uri for citation in citations}
    return len(cited & relevant) / len(relevant)


@dataclass(frozen=True)
class CitationEvaluation:
    precision: float
    recall: float

    @classmethod
    def from_citations(
        cls, citations: list[Citation], relevant_source_uris: set[str]
    ) -> "CitationEvaluation":
        return cls(
            precision=citation_precision(citations, relevant_source_uris),
            recall=citation_recall(citations, relevant_source_uris),
        )
