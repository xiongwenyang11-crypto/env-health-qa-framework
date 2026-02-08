"""
Evaluation package API.

This file re-exports commonly used evaluation utilities so that notebooks and
downstream code can import from `evaluation` directly.

Notes (manuscript-aligned):
- Expert panel size is arbitrary (>=2), with the manuscript using 10 external experts.
- Factual consistency uses a 0–2 ordinal scale; confidence uses Low/Medium/High.
- Uncertainty alignment in the manuscript is assessed using weighted Cohen's κ (linear weights).
"""

from .precision_at_k import precision_at_k, add_precision_at_k

# Backward-compatible import (only needed if you still use relevance_list like "1|1|0")
# If you migrate to document-level relevance labels, this can be removed.
try:
    from .precision_at_k import parse_relevance_list  # type: ignore
    _HAS_PARSE_RELEVANCE_LIST = True
except Exception:  # pragma: no cover
    parse_relevance_list = None  # type: ignore
    _HAS_PARSE_RELEVANCE_LIST = False

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
    # Retrieval metrics
    "precision_at_k",
    "add_precision_at_k",

    # Agreement
    "pairwise_cohens_kappa",

    # Expert aggregation + reporting
    "aggregate_expert_scores",
    "compute_expert_confidence_majority",
    "compute_uncertainty_alignment",
    "build_report_table",
    "compute_summary_stats",
    "export_outputs",
]

# Keep parse_relevance_list only if available (backward compatibility).
if _HAS_PARSE_RELEVANCE_LIST:
    __all__.insert(0, "parse_relevance_list")
