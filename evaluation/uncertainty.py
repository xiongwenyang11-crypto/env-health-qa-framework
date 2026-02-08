"""
evaluation.uncertainty

Utilities for uncertainty alignment between:
- system-assigned confidence labels (Low/Medium/High)
- expert confidence labels (Low/Medium/High), aggregated by majority vote

Manuscript alignment:
- Confidence alignment is quantified using weighted Cohen's κ (linear weights),
  treating Low/Medium/High as an ordinal scale.
- Side-by-side per-scenario comparison is retained for transparency.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


CONF_ORDER_DEFAULT = ["Low", "Medium", "High"]


def majority_vote(series: pd.Series):
    """
    Return the most frequent value in a series (deterministic).
    If ties occur, pandas value_counts() order decides.
    """
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    return vc.idxmax()


def compute_expert_confidence_majority(
    expert_ratings: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    expert_conf_col: str = "expert_confidence",
    out_col: str = "expert_confidence_majority",
) -> pd.DataFrame:
    """
    Compute per-scenario expert majority confidence label.

    Returns:
        DataFrame with:
            - scenario_id
            - expert_confidence_majority
    """
    for c in (scenario_col, expert_conf_col):
        if c not in expert_ratings.columns:
            raise ValueError(f"Missing column '{c}' in expert_ratings table.")

    maj = (
        expert_ratings.groupby(scenario_col)[expert_conf_col]
        .apply(majority_vote)
        .reset_index()
        .rename(columns={expert_conf_col: out_col})
    )
    return maj


def uncertainty_alignment_table(
    system_confidence: pd.DataFrame,
    expert_ratings: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    system_conf_col: str = "system_confidence",
    expert_conf_col: str = "expert_confidence",
    out_expert_majority_col: str = "expert_confidence_majority",
) -> pd.DataFrame:
    """
    Create a per-scenario uncertainty comparison table:
        scenario_id, system_confidence, expert_confidence_majority

    (No longer forces aligned(0/1) as the primary metric; κ is computed separately.)
    """
    for c in (scenario_col, system_conf_col):
        if c not in system_confidence.columns:
            raise ValueError(f"Missing column '{c}' in system_confidence table.")

    expert_majority = compute_expert_confidence_majority(
        expert_ratings,
        scenario_col=scenario_col,
        expert_conf_col=expert_conf_col,
        out_col=out_expert_majority_col,
    )

    ua = system_confidence.merge(expert_majority, on=scenario_col, how="left")
    return ua[[scenario_col, system_conf_col, out_expert_majority_col]]


def compute_uncertainty_alignment_kappa(
    ua_table: pd.DataFrame,
    *,
    system_conf_col: str = "system_confidence",
    expert_majority_col: str = "expert_confidence_majority",
    conf_order: Sequence[str] = CONF_ORDER_DEFAULT,
    weights: str = "linear",
) -> Tuple[float, int, float]:
    """
    Compute weighted Cohen's κ (linear by default) between system confidence and expert consensus confidence.

    Returns:
        (kappa_weighted, n_items, exact_match_rate)

    Notes:
        - κ is the manuscript-aligned primary metric.
        - exact_match_rate is an auxiliary descriptive statistic (optional to report).
    """
    for c in (system_conf_col, expert_majority_col):
        if c not in ua_table.columns:
            raise ValueError(f"Missing column '{c}' in ua_table.")

    mapping = {v: i for i, v in enumerate(conf_order)}
    sys_codes = ua_table[system_conf_col].map(mapping)
    exp_codes = ua_table[expert_majority_col].map(mapping)

    mask = (~pd.isna(sys_codes)) & (~pd.isna(exp_codes))
    n_items = int(mask.sum())
    if n_items == 0:
        return float("nan"), 0, float("nan")

    kappa = float(cohen_kappa_score(sys_codes[mask], exp_codes[mask], weights=weights))

    exact_match_rate = float((ua_table.loc[mask, system_conf_col] == ua_table.loc[mask, expert_majority_col]).mean())
    return kappa, n_items, exact_match_rate