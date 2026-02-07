"""
Evaluation aggregation and reporting utilities.

This module provides:
- aggregate_expert_scores: per-scenario mean factuality and interpretability
- compute_expert_confidence_majority: majority vote for expert confidence labels
- compute_uncertainty_alignment: system vs expert majority alignment (0/1)
- build_report_table: Table-1-like report table
- compute_summary_stats: mean ± SD for key metrics + alignment rate + mean κ
- export_outputs: export CSVs into outputs/ folder

All functions are deterministic and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def aggregate_expert_scores(
    expert_ratings: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    factuality_col: str = "factuality",
    interpretability_col: str = "interpretability",
) -> pd.DataFrame:
    """
    Aggregate expert ratings per scenario (mean factuality and mean interpretability).

    Returns a DataFrame with:
    - scenario_id
    - factuality_mean
    - interpretability_mean
    """
    for c in (scenario_col, factuality_col, interpretability_col):
        if c not in expert_ratings.columns:
            raise KeyError(f"Missing column '{c}' in expert_ratings.")

    agg = (
        expert_ratings.groupby(scenario_col)
        .agg(
            factuality_mean=(factuality_col, "mean"),
            interpretability_mean=(interpretability_col, "mean"),
        )
        .reset_index()
    )
    return agg


def _majority_vote(series: pd.Series) -> str:
    """
    Majority vote for categorical labels.
    Ties are resolved by pandas' idxmax order (deterministic).
    """
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return ""
    return str(vc.idxmax())


def compute_expert_confidence_majority(
    expert_ratings: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    conf_col: str = "expert_confidence",
    out_col: str = "expert_confidence_majority",
) -> pd.DataFrame:
    """
    Compute majority-vote confidence label per scenario.

    Returns:
    - scenario_id
    - expert_confidence_majority
    """
    for c in (scenario_col, conf_col):
        if c not in expert_ratings.columns:
            raise KeyError(f"Missing column '{c}' in expert_ratings.")

    maj = (
        expert_ratings.groupby(scenario_col)[conf_col]
        .apply(_majority_vote)
        .reset_index()
        .rename(columns={conf_col: out_col})
    )
    return maj


def compute_uncertainty_alignment(
    *,
    system_confidence_df: pd.DataFrame,
    expert_majority_df: pd.DataFrame,
    scenario_col: str = "scenario_id",
    system_conf_col: str = "system_confidence",
    expert_conf_col: str = "expert_confidence_majority",
) -> pd.DataFrame:
    """
    Compute uncertainty alignment between system confidence and expert majority confidence.

    Returns:
    - scenario_id
    - system_confidence
    - expert_confidence_majority
    - aligned (0/1)
    """
    for c in (scenario_col, system_conf_col):
        if c not in system_confidence_df.columns:
            raise KeyError(f"Missing column '{c}' in system_confidence_df.")
    for c in (scenario_col, expert_conf_col):
        if c not in expert_majority_df.columns:
            raise KeyError(f"Missing column '{c}' in expert_majority_df.")

    merged = system_confidence_df.merge(expert_majority_df, on=scenario_col, how="left")
    merged["aligned"] = (merged[system_conf_col] == merged[expert_conf_col]).astype(int)

    return merged[[scenario_col, system_conf_col, expert_conf_col, "aligned"]]


def build_report_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Table-1-like report table from a merged results dataframe.

    Expected columns (minimum):
    - scenario_id, pollutant, outcome, query
    - k, precision_at_k
    - factuality_mean, interpretability_mean
    - system_confidence
    - expert_confidence_majority
    - aligned

    Returns a sorted report DataFrame.
    """
    required = [
        "scenario_id",
        "pollutant",
        "outcome",
        "query",
        "k",
        "precision_at_k",
        "factuality_mean",
        "interpretability_mean",
        "system_confidence",
        "expert_confidence_majority",
        "aligned",
    ]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in results_df: {missing}")

    report = results_df[required].copy()
    report = report.sort_values("scenario_id")
    return report


def compute_summary_stats(
    *,
    results: pd.DataFrame,
    kappa_factuality: Optional[pd.DataFrame] = None,
    kappa_interpretability: Optional[pd.DataFrame] = None,
    precision_col: str = "precision_at_k",
    factuality_col: str = "factuality_mean",
    interpretability_col: str = "interpretability_mean",
    alignment_col: str = "aligned",
) -> pd.DataFrame:
    """
    Compute summary statistics (mean ± SD) and optionally include mean kappa values.

    Returns a single-row DataFrame.
    """
    for c in (precision_col, factuality_col, interpretability_col, alignment_col):
        if c not in results.columns:
            raise KeyError(f"Missing column '{c}' in results.")

    def _mean_sd(x: pd.Series):
        return float(x.mean()), float(x.std(ddof=1))

    factuality_mean, factuality_sd = _mean_sd(results[factuality_col])
    precision_mean, precision_sd = _mean_sd(results[precision_col])
    interp_mean, interp_sd = _mean_sd(results[interpretability_col])
    alignment_rate = float(results[alignment_col].mean())

    k_f = float("nan")
    k_i = float("nan")
    if kappa_factuality is not None and "kappa" in kappa_factuality.columns:
        k_f = float(np.nanmean(kappa_factuality["kappa"].to_numpy()))
    if kappa_interpretability is not None and "kappa" in kappa_interpretability.columns:
        k_i = float(np.nanmean(kappa_interpretability["kappa"].to_numpy()))

    out = pd.DataFrame([{
        "factuality_mean": factuality_mean,
        "factuality_sd": factuality_sd,
        "precision_at_k_mean": precision_mean,
        "precision_at_k_sd": precision_sd,
        "interpretability_mean": interp_mean,
        "interpretability_sd": interp_sd,
        "uncertainty_alignment_rate": alignment_rate,
        "kappa_factuality_pairwise_mean": k_f,
        "kappa_interpretability_pairwise_mean": k_i,
    }])
    return out


def export_outputs(
    *,
    out_dir: Path,
    report_table: pd.DataFrame,
    summary_stats: pd.DataFrame,
    kappa_factuality: Optional[pd.DataFrame] = None,
    kappa_interpretability: Optional[pd.DataFrame] = None,
) -> None:
    """
    Export outputs as CSV files into out_dir.

    Creates out_dir if it does not exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_table.to_csv(out_dir / "demo_report_table.csv", index=False)
    summary_stats.to_csv(out_dir / "demo_summary_stats.csv", index=False)

    if kappa_factuality is not None:
        kappa_factuality.to_csv(out_dir / "demo_kappa_factuality.csv", index=False)
    if kappa_interpretability is not None:
        kappa_interpretability.to_csv(out_dir / "demo_kappa_interpretability.csv", index=False)
