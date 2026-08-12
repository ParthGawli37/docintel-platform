"""Deterministic evaluation primitives for retrieval and citation quality."""

from docintel.evaluation.metrics import (
    CitationEvaluation,
    RetrievalEvaluation,
    citation_precision,
    citation_recall,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    evaluate_retrieval,
)

__all__ = [
    "CitationEvaluation",
    "RetrievalEvaluation",
    "citation_precision",
    "citation_recall",
    "evaluate_retrieval",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]
