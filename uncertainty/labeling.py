"""
uncertainty.labeling

Mapping calibrated confidence scores to qualitative labels:
- Low / Medium / High

Also provides helpers to annotate retrieved results tables.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Union, Optional, Literal

import numpy as np
import pandas as pd

from .calibration import calibrate_scores


ConfidenceLabel = Literal["Low", "Medium", "High"]


@dataclass(frozen=True)
class ConfidenceConfig:
    """
    Configuration for confidence calibration and labeling.
    """
    normalize: bool = True
    a: float = 10.0
    b: float = 0.5
    t_low: float = 0.33
    t_high: float = 0.67


def score_to_label(p: float, t_low: float = 0.33, t_high: float = 0.67) -> ConfidenceLabel:
    """
    Convert a calibrated score to a qualitative confidence label.
    """
    if p < t_low:
        return "Low"
    if p < t_high:
        return "Medium"
    return "High"


def label_scores(
    scores: Sequence[Union[int, float]],
    *,
    config: ConfidenceConfig = ConfidenceConfig(),
) -> np.ndarray:
    """
    Calibrate and label a list of raw scores.

    Returns:
        array of labels, same length as scores
    """
    calibrated = calibrate_scores(
        scores,
        normalize=config.normalize,
        a=config.a,
        b=config.b,
    )
    labels = np.array([score_to_label(float(p), config.t_low, config.t_high) for p in calibrated], dtype=object)
    return labels


def annotate_retrieval_df(
    retrieved_df: pd.DataFrame,
    *,
    similarity_col: str = "similarity",
    out_score_col: str = "calibrated_score",
    out_label_col: str = "confidence",
    config: ConfidenceConfig = ConfidenceConfig(),
) -> pd.DataFrame:
    """
    Given a retrieval output DataFrame with a similarity column,
    add calibrated_score and confidence label columns.

    Expected:
        retrieved_df[similarity_col] exists

    Returns:
        copy of DataFrame with extra columns.
    """
    if similarity_col not in retrieved_df.columns:
        raise ValueError(f"Missing column '{similarity_col}' in retrieved_df.")

    df = retrieved_df.copy()
    raw = df[similarity_col].to_numpy(dtype=float)

    calibrated = calibrate_scores(
        raw,
        normalize=config.normalize,
        a=config.a,
        b=config.b,
    )

    df[out_score_col] = calibrated
    df[out_label_col] = [score_to_label(float(p), config.t_low, config.t_high) for p in calibrated]
    return df


def overall_confidence(
    retrieved_df: pd.DataFrame,
    *,
    label_col: str = "confidence",
    rank_col: str = "rank",
    method: str = "top1",
) -> ConfidenceLabel:
    """
    Derive a single overall confidence label for the final answer.

    Args:
        retrieved_df: retrieval table that has a confidence label per row
        method:
            - "top1": use confidence label of the top-ranked evidence (default)
            - "majority": majority vote among evidence confidence labels

    Returns:
        One of "Low" / "Medium" / "High"
    """
    if label_col not in retrieved_df.columns:
        raise ValueError(f"Missing column '{label_col}' in retrieved_df.")

    if len(retrieved_df) == 0:
        return "Low"

    if method == "top1":
        if rank_col in retrieved_df.columns:
            top = retrieved_df.sort_values(rank_col).iloc[0][label_col]
        else:
            top = retrieved_df.iloc[0][label_col]
        return top  # type: ignore

    if method == "majority":
        vc = retrieved_df[label_col].value_counts()
        if vc.empty:
            return "Low"
        return vc.idxmax()  # type: ignore

    raise ValueError(f"Unknown method: {method}")
