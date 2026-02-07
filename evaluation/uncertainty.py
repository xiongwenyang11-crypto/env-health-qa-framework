"""
evaluation.uncertainty

Utilities for uncertainty alignment between:
- system-assigned confidence labels (Low/Medium/High)
- expert confidence labels (Low/Medium/High), aggregated by majority vote
"""

from __future__ import annotations

from typing import Optional
import pandas as pd


def majority_vote(series: pd.Series) -> str:
    """
    Return the most frequent value in a series.
    If ties occur, pandas value_counts() order will decide deterministically.
    """
    vc = series.value_counts()
    if vc.empty:
        return ""
    return vc.idxmax()


def compute_expert_confidence_majority(
    expert_ratings: pd.DataFrame,
    scenario_col: str = "scenario_id",
    expert_conf_col: str = "expert_confidence",
) -> pd.DataFrame:
    """
    Compute per-scenario expert majority confidence label.

    Returns:
        DataFrame with:
            - scenario_id
            - expert_confidence_majority
    """
    for c in [scenario_col, expert_conf_col]:
        if c not in expert_ratings.columns:
            raise ValueError(f"Missing column '{c}' in expert_ratings table.")

    maj = (
        expert_ratings.groupby(scenario_col)[expert_conf_col]
        .apply(majority_vote)
        .reset_index()
        .rename(columns={expert_conf_col: "expert_confidence_majority"})
    )
    return maj


def uncertainty_alignment_table(
    system_confidence: pd.DataFrame,
    expert_ratings: pd.DataFrame,
    scenario_col: str = "scenario_id",
    system_conf_col: str = "system_confidence",
    expert_conf_col: str = "expert_confidence",
) -> pd.DataFrame:
    """
    Create uncertainty alignment table:
        scenario_id, system_confidence, expert_confidence_majority, aligned (0/1)
    """
    for c in [scenario_col, system_conf_col]:
        if c not in system_confidence.columns:
            raise ValueError(f"Missing column '{c}' in system_confidence table.")

    expert_majority = compute_expert_confidence_majority(
        expert_ratings,
        scenario_col=scenario_col,
        expert_conf_col=expert_conf_col,
    )

    ua = system_confidence.merge(expert_majority, on=scenario_col, how="left")
    ua["aligned"] = (ua[system_conf_col] == ua["expert_confidence_majority"]).astype(int)
    return ua
