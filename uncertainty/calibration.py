"""
uncertainty.calibration

Score calibration utilities:
- min-max normalization
- sigmoid transformation

Designed for demonstration / reproducibility.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union

import numpy as np


def minmax_normalize(x: Sequence[Union[int, float]], eps: float = 1e-9) -> np.ndarray:
    """
    Min-max normalize to [0, 1].

    Args:
        x: sequence of numeric values
        eps: numerical stability term

    Returns:
        np.ndarray in [0, 1] (if all values equal, returns zeros)
    """
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0:
        return arr

    x_min = float(arr.min())
    x_max = float(arr.max())

    denom = (x_max - x_min) + eps
    return (arr - x_min) / denom


def sigmoid(x: Sequence[Union[int, float]], a: float = 10.0, b: float = 0.5) -> np.ndarray:
    """
    Sigmoid transform: 1 / (1 + exp(-a*(x-b)))

    Args:
        x: values (typically normalized in [0, 1])
        a: steepness
        b: midpoint

    Returns:
        np.ndarray of same shape as x
    """
    arr = np.asarray(list(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-a * (arr - b)))


def calibrate_scores(
    raw_scores: Sequence[Union[int, float]],
    *,
    normalize: bool = True,
    a: float = 10.0,
    b: float = 0.5,
) -> np.ndarray:
    """
    Calibrate raw similarity scores into a smoother confidence score.

    Typical use:
        raw cosine similarity -> normalize -> sigmoid -> calibrated score

    Args:
        raw_scores: list/array of similarity scores (e.g., cosine similarity)
        normalize: whether to apply minmax normalization first
        a, b: sigmoid parameters

    Returns:
        calibrated scores in (0, 1)
    """
    if normalize:
        z = minmax_normalize(raw_scores)
    else:
        z = np.asarray(list(raw_scores), dtype=float)
    return sigmoid(z, a=a, b=b)
