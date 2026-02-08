"""
uncertainty.calibration

Manuscript-aligned uncertainty calibration utilities.

Primary (paper-consistent) workflow:
1) Min–max normalize retrieval similarity scores to [0, 1]
2) Aggregate normalized scores (arithmetic mean) to obtain an evidence-support indicator
3) Map the indicator to qualitative confidence levels using fixed thresholds (t1, t2)

Notes:
- This is an interpretable proxy for relative evidence support, not a probabilistic calibration.
- Functions are deterministic given fixed inputs and thresholds.

Optional utilities:
- Sigmoid transformation is provided for exploratory/demo purposes only and is NOT used
  in the manuscript-aligned calibration path.
"""

from __future__ import annotations

from typing import Sequence, Union, Tuple, Optional

import numpy as np


CONF_LEVELS = ("Low", "Medium", "High")


def minmax_normalize(x: Sequence[Union[int, float]], eps: float = 1e-9) -> np.ndarray:
    """
    Min–max normalize to [0, 1].

    If all values are equal (or length <= 1), returns zeros.

    Args:
        x: sequence of numeric values
        eps: numerical stability term (avoid division by zero)

    Returns:
        np.ndarray normalized to [0, 1]
    """
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0:
        return arr

    x_min = float(np.nanmin(arr))
    x_max = float(np.nanmax(arr))

    # If all scores are identical, there is no relative support signal
    if np.isclose(x_max, x_min):
        return np.zeros_like(arr, dtype=float)

    denom = (x_max - x_min) + eps
    return (arr - x_min) / denom


def aggregate_evidence_support(
    raw_scores: Sequence[Union[int, float]],
    *,
    normalize: bool = True,
) -> float:
    """
    Compute a single evidence-support indicator from raw similarity scores.

    Manuscript-aligned:
      normalized scores (min–max to [0,1]) -> arithmetic mean.

    Args:
        raw_scores: similarity scores for top-k retrieved documents (e.g., cosine similarity)
        normalize: if True, apply min–max normalization before aggregation

    Returns:
        float evidence-support indicator in [0,1] (NaN if empty)
    """
    arr = np.asarray(list(raw_scores), dtype=float)
    if arr.size == 0:
        return float("nan")

    if normalize:
        arr = minmax_normalize(arr)

    return float(np.nanmean(arr))


def map_support_to_confidence(
    support: float,
    *,
    t1: float,
    t2: float,
) -> str:
    """
    Map an evidence-support indicator to qualitative confidence (Low/Medium/High).

    Manuscript-aligned:
      support < t1  -> Low
      t1 <= support < t2 -> Medium
      support >= t2 -> High

    Args:
        support: evidence-support indicator (typically in [0,1])
        t1: lower threshold (fixed prior to evaluation)
        t2: upper threshold (fixed prior to evaluation)

    Returns:
        "Low" | "Medium" | "High"
    """
    if np.isnan(support):
        return ""

    if t2 < t1:
        raise ValueError("Thresholds must satisfy t1 <= t2.")

    if support < t1:
        return "Low"
    if support < t2:
        return "Medium"
    return "High"


def assign_confidence_from_scores(
    raw_scores: Sequence[Union[int, float]],
    *,
    t1: float,
    t2: float,
    normalize: bool = True,
) -> Tuple[float, str]:
    """
    Convenience function: raw similarity scores -> (support, qualitative confidence).

    Returns:
        (support_indicator, confidence_label)
    """
    support = aggregate_evidence_support(raw_scores, normalize=normalize)
    label = map_support_to_confidence(support, t1=t1, t2=t2)
    return support, label


# -------------------------
# Optional exploratory utility (NOT used in manuscript)
# -------------------------

def sigmoid(x: Sequence[Union[int, float]], a: float = 10.0, b: float = 0.5) -> np.ndarray:
    """
    Sigmoid transform: 1 / (1 + exp(-a*(x-b))).

    Provided for exploratory/demo purposes only. Not used in the manuscript-aligned path.
    """
    arr = np.asarray(list(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-a * (arr - b)))