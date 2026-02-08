"""
Evaluation aggregation and reporting utilities.

Manuscript-aligned utilities for:
- aggregate_expert_scores: per-scenario interpretability mean (SD) + optional factuality mean
- compute_expert_consensus_factuality: majority vote (0/1/2 ordinal) per scenario
- compute_expert_confidence_majority: majority vote (Low/Medium/High) per scenario
- compute_uncertainty_alignment_kappa: weighted Cohen's κ (linear weights) between system vs expert consensus confidence
- build_report_table: Table-2-like report table (query-level breakdown)
- compute_summary_stats: descriptive summaries (mean ± SD, κ for confidence alignment)
- export_outputs: export CSVs into outputs/ folder

All functions are deterministic and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


CONF_ORDER_DEFAULT = ["Low", "Medium", "High"]


def _majority_vote(series: pd.Series) -> Any:
    """
    Majority vote for labels (deterministic).
    Ties are resolved by pandas' idxmax order.
    """
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    return vc.idxmax()


def compute_expert_consensus_factuality(
    expert_ratings: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    factuality_col: str = "factuality",
    out_col: str = "factuality_consensus",
) -> pd.DataFrame:
    """
    Compute expert-consensus factuality label per scenario via majority vote.

    Manuscript alignment:
    - factuality uses an ordinal 0–2 scale:
        2 = fully consistent
        1 = broadly consistent but incomplete
        0 = incorrect/unsupported

    Returns:
    - scenario_id
    - factuality_consensus
    """
    for c in (scenario_col, factuality_col):
        if c not in expert_ratings.columns:
            raise KeyError(f"Missing column '{c}' in expert_ratings.")

    cons = (
        expert_ratings.groupby(scenario_col)[factuality_col]
        .apply(_majority_vote)
        .reset_index()
        .rename(columns={factuality_col: out_col})
    )
    return cons


def aggregate_expert_scores(
    expert_ratings: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    factuality_col: str = "factuality",
    interpretability_col: str = "interpretability",
) -> pd.DataFrame:
    """
    Aggregate expert ratings per scenario.

    Manuscript alignment:
    - Interpretability is summarized as mean (SD) across experts (5-point Likert).
    - Factuality is ordinal; mean is optional descriptive context, but consensus label
      should be used for query-level reporting.

    Returns:
    - scenario_id
    - interpretability_mean
    - interpretability_sd
    - factuality_mean (optional descriptive)
    - factuality_sd  (optional descriptive)
    - n_experts
    """
    for c in (scenario_col, factuality_col, interpretability_col):
        if c not in expert_ratings.columns:
            raise KeyError(f"Missing column '{c}' in expert_ratings.")

    agg = (
        expert_ratings.groupby(scenario_col)
        .agg(
            n_experts=(interpretability_col, "count"),
            interpretability_mean=(interpretability_col, "mean"),
            interpretability_sd=(interpretability_col, lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            factuality_mean=(factuality_col, "mean"),
            factuality_sd=(factuality_col, lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        )
        .reset_index()
    )
    return agg


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


def compute_uncertainty_alignment_kappa(
    *,
    system_confidence_df: pd.DataFrame,
    expert_majority_df: pd.DataFrame,
    scenario_col: str = "scenario_id",
    system_conf_col: str = "system_confidence",
    expert_conf_col: str = "expert_confidence_majority",
    conf_order: Sequence[str] = CONF_ORDER_DEFAULT,
    weights: str = "linear",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute uncertainty alignment between system confidence and expert consensus confidence
    using weighted Cohen's κ (linear weights), consistent with the manuscript.

    Returns:
    1) merged_table: per-scenario side-by-side comparison
       - scenario_id
       - system_confidence
       - expert_confidence_majority

    2) alignment_stats: single-row DataFrame with:
       - kappa_weighted
       - n_items

    Notes
    -----
    This κ captures ordinal agreement/interpretability alignment rather than probabilistic calibration.
    """
    for c in (scenario_col, system_conf_col):
        if c not in system_confidence_df.columns:
            raise KeyError(f"Missing column '{c}' in system_confidence_df.")
    for c in (scenario_col, expert_conf_col):
        if c not in expert_majority_df.columns:
            raise KeyError(f"Missing column '{c}' in expert_majority_df.")

    merged = system_confidence_df.merge(expert_majority_df, on=scenario_col, how="left")

    # Encode ordinal labels deterministically
    mapping = {v: i for i, v in enumerate(conf_order)}
    sys_codes = merged[system_conf_col].map(mapping)
    exp_codes = merged[expert_conf_col].map(mapping)

    mask = (~pd.isna(sys_codes)) & (~pd.isna(exp_codes))
    n_items = int(mask.sum())

    if n_items == 0:
        kappa = float("nan")
    else:
        kappa = float(cohen_kappa_score(sys_codes[mask], exp_codes[mask], weights=weights))

    stats = pd.DataFrame([{"kappa_weighted": kappa, "n_items": n_items, "weights": weights}])
    return merged[[scenario_col, system_conf_col, expert_conf_col]], stats


def build_report_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Table-2-like report table from a merged results dataframe.

    Expected columns (minimum):
    - query_instance_id OR scenario_id (at least one)
    - pollutant, outcome, query
    - k, precision_at_k
    - factuality_consensus
    - interpretability_mean, interpretability_sd
    - system_confidence
    - expert_confidence_majority

    Returns a sorted report DataFrame.
    """
    # Support either query_instance_id (18 instances) or scenario_id (6 topics demo).
    id_col = "query_instance_id" if "query_instance_id" in results_df.columns else "scenario_id"

    required = [
        id_col,
        "pollutant",
        "outcome",
        "query",
        "k",
        "precision_at_k",
        "factuality_consensus",
        "interpretability_mean",
        "interpretability_sd",
        "system_confidence",
        "expert_confidence_majority",
    ]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in results_df: {missing}")

    report = results_df[required].copy()
    report = report.sort_values(id_col)
    return report


def compute_summary_stats(
    *,
    results: pd.DataFrame,
    alignment_stats: Optional[pd.DataFrame] = None,
    precision_col: str = "precision_at_k",
    interpretability_mean_col: str = "interpretability_mean",
) -> pd.DataFrame:
    """
    Compute descriptive summary statistics.

    Manuscript alignment:
    - Precision@k: mean ± SD across query instances
    - Interpretability: mean ± SD across query instances (using per-instance mean ratings)
    - Uncertainty alignment: report weighted κ (linear weights) from alignment_stats

    Returns a single-row DataFrame.
    """
    for c in (precision_col, interpretability_mean_col):
        if c not in results.columns:
            raise KeyError(f"Missing column '{c}' in results.")

    def _mean_sd(x: pd.Series) -> Tuple[float, float]:
        return float(x.mean()), float(x.std(ddof=1))

    precision_mean, precision_sd = _mean_sd(results[precision_col])
    interp_mean, interp_sd = _mean_sd(results[interpretability_mean_col])

    kappa = float("nan")
    n_items = float("nan")
    weights = ""
    if alignment_stats is not None and len(alignment_stats) > 0:
        kappa = float(alignment_stats.loc[0, "kappa_weighted"])
        n_items = float(alignment_stats.loc[0, "n_items"])
        weights = str(alignment_stats.loc[0, "weights"])

    out = pd.DataFrame(
        [
            {
                "precision_at_k_mean": precision_mean,
                "precision_at_k_sd": precision_sd,
                "interpretability_mean": interp_mean,
                "interpretability_sd": interp_sd,
                "confidence_alignment_kappa_weighted": kappa,
                "confidence_alignment_n_items": n_items,
                "confidence_alignment_weights": weights,
            }
        ]
    )
    return out


def export_outputs(
    *,
    out_dir: Path,
    report_table: pd.DataFrame,
    summary_stats: pd.DataFrame,
    confidence_alignment_table: Optional[pd.DataFrame] = None,
) -> None:
    """
    Export outputs as CSV files into out_dir.

    Creates out_dir if it does not exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_table.to_csv(out_dir / "demo_report_table.csv", index=False)
    summary_stats.to_csv(out_dir / "demo_summary_stats.csv", index=False)

    if confidence_alignment_table is not None:
        confidence_alignment_table.to_csv(out_dir / "demo_confidence_alignment_table.csv", index=False)