"""
uncertainty package API.

Re-exports the most commonly used uncertainty utilities for confidence calibration
and labeling, aligned with the manuscript’s methodology.

Modules:
- calibration.py: Min–max normalization, aggregation, and thresholds-based labeling
- labeling.py: Labeling scores based on evidence support and thresholds
"""

from .calibration import minmax_normalize, aggregate_evidence_support, map_support_to_confidence
from .labeling import (
    ConfidenceConfig,
    score_to_label,
    label_scores,
    annotate_retrieval_df,
    compute_overall_confidence_from_retrieval,
)

__all__ = [
    "minmax_normalize",
    "aggregate_evidence_support",
    "map_support_to_confidence",
    "ConfidenceConfig",
    "score_to_label",
    "label_scores",
    "annotate_retrieval_df",
    "compute_overall_confidence_from_retrieval",
]