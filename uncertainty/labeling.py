"""
uncertainty.labeling

Manuscript-aligned mapping from retrieval similarity support to qualitative labels:
- Low / Medium / High

In the manuscript, uncertainty is computed at the *query level*:
1) Take similarity scores from top-k retrieved documents
2) Min–max normalize to [0,1]
3) Aggregate with arithmetic mean -> evidence support indicator
4) Apply fixed thresholds (t1, t2) -> Low/Medium/High

This module provides:
- score_to_label: map support indicator to label (Low/Medium/High)
- annotate_retrieval_df: optional per-document normalized scores for inspection
- compute_overall_confidence_from_retrieval: primary query-level confidence computation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union, Optional, Literal, Tuple

import numpy as np
import pandas as pd

from .calibration import minmax_normalize, aggregate_evidence_support, map_support_to_confidence


ConfidenceLabel = Literal["Low", "Medium", "High"]


@dataclass(frozen=True)
class ConfidenceConfig:
    """
    Configuration for qualitative uncertainty labeling (manuscript-aligned).
    """
    normalize: bool = True
    t1: float = 0.33
    t2: float = 0.67


def score_to_label(support: float, *, t1: float, t2: float) -> ConfidenceLabel:
    """
    Convert an evidence-support indicator to a qualitative confidence label.
    """
    label = map_support_to_confidence(support, t1=t1, t2=t2)
    # map_support_to_confidence may return "" if NaN; default to Low for robustness
    if label == "":
        return "Low"
    return label  # type: ignore


def annotate_retrieval_df(
    retrieved_df: pd.DataFrame,
    *,
    similarity_col: str = "similarity",
    out_norm_col: str = "similarity_norm",
    config: ConfidenceConfig = ConfidenceConfig(),
) -> pd.DataFrame:
    """
    Optional helper to annotate a retrieval output DataFrame with per-document
    normalized similarity scores (for inspection/visualization).

    This does NOT define the manuscript's query-level confidence by itself.

    Returns:
        Copy of DataFrame with an added normalized similarity column.
    """
    if similarity_col not in retrieved_df.columns:
        raise ValueError(f"Missing column '{similarity_col}' in retrieved_df.")

    df = retrieved_df.copy()
    raw = df[similarity_col].to_numpy(dtype=float)

    if config.normalize:
        df[out_norm_col] = minmax_normalize(raw)
    else:
        df[out_norm_col] = raw

    return df


def compute_support_indicator(
    retrieved_df: pd.DataFrame,
    *,
    similarity_col: str = "similarity",
    config: ConfidenceConfig = ConfidenceConfig(),
) -> float:
    """
    Compute the manuscript-aligned evidence-support indicator for a query
    from a retrieval table's similarity scores.

    Returns:
        support indicator in [0,1] (NaN if empty)
    """
    if similarity_col not in retrieved_df.columns:
        raise ValueError(f"Missing column '{similarity_col}' in retrieved_df.")

    scores = retrieved_df[similarity_col].to_numpy(dtype=float)
    return aggregate_evidence_support(scores, normalize=config.normalize)


def compute_overall_confidence_from_retrieval(
    retrieved_df: pd.DataFrame,
    *,
    similarity_col: str = "similarity",
    config: ConfidenceConfig = ConfidenceConfig(),
) -> Tuple[float, ConfidenceLabel]:
    """
    Primary manuscript-aligned function:
    retrieval_df -> (support_indicator, qualitative_confidence)

    Returns:
        (support, label)
    """
    support = compute_support_indicator(
        retrieved_df,
        similarity_col=similarity_col,
        config=config,
    )
    label = score_to_label(support, t1=config.t1, t2=config.t2)
    return support, label