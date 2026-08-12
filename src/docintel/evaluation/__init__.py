"""Deterministic evaluation primitives for retrieval and citation quality."""

from docintel.evaluation.metrics import (
    CitationEvaluation,
    EvaluationReport,
    RetrievalCase,
    RetrievalEvaluation,
    citation_precision,
    citation_recall,
    evaluate_dataset,
    evaluate_retrieval,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "CitationEvaluation",
    "EvaluationReport",
    "RetrievalCase",
    "RetrievalEvaluation",
    "citation_precision",
    "citation_recall",
    "evaluate_dataset",
    "evaluate_retrieval",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]
