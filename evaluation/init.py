"""
Evaluation package API.

This file re-exports the most commonly used evaluation utilities so that
notebooks and downstream code can import from `evaluation` directly.
"""

from .precision_at_k import parse_relevance_list, precision_at_k, add_precision_at_k
from .cohens_kappa import pairwise_cohens_kappa
from .interpretability_stats import (
    aggregate_expert_scores,
    compute_expert_confidence_majority,
    compute_uncertainty_alignment,
    build_report_table,
    compute_summary_stats,
    export_outputs,
)

__all__ = [
    "parse_relevance_list",
    "precision_at_k",
    "add_precision_at_k",
    "pairwise_cohens_kappa",
    "aggregate_expert_scores",
    "compute_expert_confidence_majority",
    "compute_uncertainty_alignment",
    "build_report_table",
    "compute_summary_stats",
    "export_outputs",
]
