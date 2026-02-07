"""
Citation precision@k utilities.

This module provides:
- parse_relevance_list: parse "1|0|1" into [1,0,1]
- add_precision_at_k: compute precision@k from relevance lists

Designed for transparency and reproducibility (no heavy dependencies).
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pandas as pd


def parse_relevance_list(s: Union[str, Sequence[int]]) -> List[int]:
    """
    Parse a relevance list stored as a string like "1|0|1" into a list of ints [1,0,1].

    If a list/tuple is already provided, it will be converted to a list of ints.

    Parameters
    ----------
    s:
        Either a pipe-separated string (e.g., "1|0|1") or a list-like of ints.

    Returns
    -------
    List[int]
    """
    if isinstance(s, (list, tuple, np.ndarray)):
        return [int(x) for x in s]

    if s is None:
        return []

    txt = str(s).strip().strip('"').strip("'")
    if txt == "":
        return []

    parts = [p for p in txt.split("|") if p != ""]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise ValueError(f"Invalid relevance_list token '{p}' in '{s}'. Expected 0/1.") from e
    return out


def precision_at_k(relevance_list: Sequence[int]) -> float:
    """
    Compute precision@k for a given relevance list.
    Assumes relevance_list contains 0/1 labels.

    Returns NaN if the list is empty.
    """
    rel = np.asarray(list(relevance_list), dtype=float)
    if rel.size == 0:
        return float("nan")
    return float(rel.mean())


def add_precision_at_k(
    df: pd.DataFrame,
    *,
    relevance_col: str = "relevance_list",
    out_col: str = "precision_at_k",
) -> pd.DataFrame:
    """
    Add a precision@k column computed from a relevance_list column.

    Parameters
    ----------
    df:
        DataFrame containing relevance lists (already parsed into list[int]).
    relevance_col:
        Column name that stores a list of 0/1 labels.
    out_col:
        Output column name.

    Returns
    -------
    pd.DataFrame
        Copy of df with a new precision@k column.
    """
    if relevance_col not in df.columns:
        raise KeyError(f"Missing column '{relevance_col}' in df.")

    out = df.copy()
    out[out_col] = out[relevance_col].apply(precision_at_k)
    return out
