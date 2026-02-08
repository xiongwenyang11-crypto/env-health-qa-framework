"""
Citation precision@k utilities.

Manuscript-aligned:
- Precision@k is defined as the proportion of expert-judged relevant studies among the top-k retrieved items.

This module supports two input schemas:
1) Recommended (long format):
   scenario_id, document_rank (1..k), relevance (0/1), [k optional]
2) Backward-compatible (list format):
   scenario_id, relevance_list ("1|0|1" or list[int]), [k optional]

Designed for transparency and reproducibility (no heavy dependencies).
"""

from __future__ import annotations

from typing import List, Sequence, Union, Optional

import numpy as np
import pandas as pd


# -------------------------
# Backward compatibility: "1|0|1" parsing
# -------------------------

def parse_relevance_list(s: Union[str, Sequence[int], None]) -> List[int]:
    """
    Parse a relevance list stored as a string like "1|0|1" into [1,0,1].

    If a list/tuple/ndarray is already provided, it will be converted to list[int].

    Returns empty list if input is None/empty.
    """
    if s is None:
        return []

    if isinstance(s, (list, tuple, np.ndarray)):
        return [int(x) for x in s]

    txt = str(s).strip().strip('"').strip("'")
    if txt == "":
        return []

    parts = [p for p in txt.split("|") if p != ""]
    out: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as e:
            raise ValueError(f"Invalid relevance_list token '{p}' in '{s}'. Expected 0/1.") from e
        if v not in (0, 1):
            raise ValueError(f"Invalid relevance_list value '{v}' in '{s}'. Expected 0/1.")
        out.append(v)
    return out


# -------------------------
# Core metric
# -------------------------

def precision_at_k(relevance: Sequence[Union[int, float]]) -> float:
    """
    Compute precision@k as mean relevance among top-k items.

    Returns NaN if empty.
    """
    rel = np.asarray(list(relevance), dtype=float)
    if rel.size == 0:
        return float("nan")
    return float(np.nanmean(rel))


# -------------------------
# List-format API (legacy)
# -------------------------

def add_precision_at_k_from_list(
    df: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    relevance_col: str = "relevance_list",
    k_col: str = "k",
    out_col: str = "precision_at_k",
    parse_if_needed: bool = True,
) -> pd.DataFrame:
    """
    Add precision@k computed from a relevance_list column.

    Expected:
      - scenario_id
      - relevance_list (either list[int] or "1|0|1")
      - k (optional)

    Returns:
      df with added precision@k column.
    """
    for c in (scenario_col, relevance_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in df.")

    out = df.copy()

    if parse_if_needed:
        out[relevance_col] = out[relevance_col].apply(parse_relevance_list)

    out[out_col] = out[relevance_col].apply(precision_at_k)

    # If k not provided, infer from list length (deterministic)
    if k_col not in out.columns:
        out[k_col] = out[relevance_col].apply(lambda x: int(len(x)) if x is not None else 0)

    return out


# -------------------------
# Long-format API (recommended)
# -------------------------

def compute_precision_at_k_from_long(
    relevance_long: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    rank_col: str = "document_rank",
    relevance_col: str = "relevance",
    k_col: str = "k",
    out_col: str = "precision_at_k",
) -> pd.DataFrame:
    """
    Compute precision@k from document-level relevance labels (recommended schema).

    Expected columns:
      - scenario_id
      - document_rank (1..k)
      - relevance (0/1)
    Optional:
      - k (if absent, inferred as max(document_rank) per scenario)

    Returns:
      One row per scenario:
        - scenario_id
        - k
        - precision_at_k
    """
    for c in (scenario_col, rank_col, relevance_col):
        if c not in relevance_long.columns:
            raise KeyError(f"Missing column '{c}' in relevance_long.")

    df = relevance_long.copy()

    # Ensure numeric relevance
    df[relevance_col] = pd.to_numeric(df[relevance_col], errors="coerce")

    if k_col not in df.columns:
        df[k_col] = df.groupby(scenario_col)[rank_col].transform("max")

    out = (
        df.groupby(scenario_col)
        .agg(
            k=(k_col, "max"),
            precision_at_k=(relevance_col, lambda x: float(np.nanmean(x.to_numpy(dtype=float)))),
        )
        .reset_index()
        .rename(columns={"precision_at_k": out_col})
    )
    return out


# -------------------------
# Convenience wrapper
# -------------------------

def add_precision_at_k(
    df: pd.DataFrame,
    *,
    mode: str = "auto",
    out_col: str = "precision_at_k",
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience wrapper.

    mode:
      - "auto": infer schema
      - "list": use relevance_list schema
      - "long": use long schema and return per-scenario table

    Notes:
      - In "long" mode, the output is *aggregated* (one row per scenario).
      - In "list" mode, the output keeps the original rows and adds precision column.
    """
    if mode not in ("auto", "list", "long"):
        raise ValueError("mode must be one of: 'auto', 'list', 'long'.")

    if mode == "auto":
        if "relevance_list" in df.columns:
            mode = "list"
        elif "document_rank" in df.columns and "relevance" in df.columns:
            mode = "long"
        else:
            raise ValueError(
                "Unable to infer schema. Provide mode='list' or mode='long', "
                "and ensure required columns exist."
            )

    if mode == "list":
        return add_precision_at_k_from_list(df, out_col=out_col, **kwargs)

    # mode == "long"
    return compute_precision_at_k_from_long(df, out_col=out_col, **kwargs)