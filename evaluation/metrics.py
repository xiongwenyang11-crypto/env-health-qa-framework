"""
evaluation.metrics

Core evaluation metrics for the demo evaluation workflow.

Design goals:
- Minimal dependencies
- Clear, reusable functions
- Works with synthetic/demo CSV and real evaluation tables
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union, Optional, Dict, Any
import numpy as np
import pandas as pd


def precision_at_k(relevance_list: Sequence[Union[int, float]]) -> float:
    """
    Compute citation precision@k as mean relevance among top-k items.

    Args:
        relevance_list: sequence like [1, 1, 0] where 1 = relevant, 0 = not relevant

    Returns:
        float precision@k (NaN if empty)
    """
    rel = np.asarray(list(relevance_list), dtype=float)
    if rel.size == 0:
        return float("nan")
    return float(rel.mean())


def compute_precision_table(
    retrieval_relevance: pd.DataFrame,
    relevance_col: str = "relevance_list",
    out_col: str = "precision_at_k",
) -> pd.DataFrame:
    """
    Add precision@k column to a retrieval relevance table.

    Expected columns:
        - scenario_id
        - k (optional)
        - relevance_col (default: relevance_list), each row contains list-like values

    Returns:
        DataFrame with added out_col
    """
    if relevance_col not in retrieval_relevance.columns:
        raise ValueError(f"Missing column '{relevance_col}' in retrieval_relevance table.")

    df = retrieval_relevance.copy()
    df[out_col] = df[relevance_col].apply(precision_at_k)
    return df


def aggregate_expert_ratings(
    expert_ratings: pd.DataFrame,
    scenario_col: str = "scenario_id",
    factuality_col: str = "factuality",
    interpretability_col: str = "interpretability",
) -> pd.DataFrame:
    """
    Aggregate expert ratings per scenario.

    Returns a table with:
        - scenario_id
        - factuality_mean
        - interpretability_mean
    """
    for c in [scenario_col, factuality_col, interpretability_col]:
        if c not in expert_ratings.columns:
            raise ValueError(f"Missing column '{c}' in expert_ratings table.")

    agg = (
        expert_ratings.groupby(scenario_col)
        .agg(
            factuality_mean=(factuality_col, "mean"),
            interpretability_mean=(interpretability_col, "mean"),
        )
        .reset_index()
    )
    return agg


def summarize_results(
    results: pd.DataFrame,
    cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Compute overall summary statistics (mean ± SD) for key metrics.

    By default expects these columns in `results`:
        - factuality_mean
        - precision_at_k
        - interpretability_mean
        - aligned  (0/1)

    You can override column names via `cols`.

    Returns:
        A single-row DataFrame of summary stats.
    """
    default_cols = {
        "factuality_mean": "factuality_mean",
        "precision_at_k": "precision_at_k",
        "interpretability_mean": "interpretability_mean",
        "aligned": "aligned",
    }
    if cols:
        default_cols.update(cols)

    fcol = default_cols["factuality_mean"]
    pcol = default_cols["precision_at_k"]
    icol = default_cols["interpretability_mean"]
    acol = default_cols["aligned"]

    for c in [fcol, pcol, icol, acol]:
        if c not in results.columns:
            raise ValueError(f"Missing column '{c}' in results table.")

    row = {
        "factuality_mean": float(results[fcol].mean()),
        "factuality_sd": float(results[fcol].std(ddof=1)),
        "precision_at_k_mean": float(results[pcol].mean()),
        "precision_at_k_sd": float(results[pcol].std(ddof=1)),
        "interpretability_mean": float(results[icol].mean()),
        "interpretability_sd": float(results[icol].std(ddof=1)),
        "uncertainty_alignment_rate": float(results[acol].mean()),
    }
    return pd.DataFrame([row])
