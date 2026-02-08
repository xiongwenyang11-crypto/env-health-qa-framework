"""
evaluation.metrics

Core evaluation metrics for the demo evaluation workflow.

Manuscript-aligned design goals:
- Minimal dependencies and deterministic behavior
- Clear, reusable functions
- Compatible with synthetic/demo CSV schemas and real evaluation tables

Key manuscript alignments:
- Precision@k (k=3 in the paper): computed from expert-judged relevance of retrieved studies.
- Factual consistency: ordinal 0–2, summarized via expert consensus (majority vote).
- Interpretability: 5-point Likert, summarized as mean (SD) across experts.
- Uncertainty alignment: assessed using weighted Cohen's κ (linear weights) between
  system confidence and expert consensus confidence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


CONF_ORDER_DEFAULT = ["Low", "Medium", "High"]


# -------------------------
# Retrieval metric: Precision@k
# -------------------------

def precision_at_k_from_list(relevance_list: Sequence[Union[int, float]]) -> float:
    """
    Compute precision@k as mean relevance among top-k items.

    Args:
        relevance_list: sequence like [1, 1, 0] where 1 = relevant, 0 = not relevant

    Returns:
        float precision@k (NaN if empty)
    """
    rel = np.asarray(list(relevance_list), dtype=float)
    if rel.size == 0:
        return float("nan")
    return float(np.nanmean(rel))


def compute_precision_table_from_list_column(
    retrieval_relevance: pd.DataFrame,
    *,
    relevance_col: str = "relevance_list",
    out_col: str = "precision_at_k",
) -> pd.DataFrame:
    """
    Backward-compatible helper.

    Expected columns:
        - scenario_id
        - k (optional)
        - relevance_col, each row contains list-like values (already parsed)

    Returns:
        DataFrame with added out_col
    """
    if relevance_col not in retrieval_relevance.columns:
        raise ValueError(f"Missing column '{relevance_col}' in retrieval_relevance table.")

    df = retrieval_relevance.copy()
    df[out_col] = df[relevance_col].apply(precision_at_k_from_list)
    return df


def compute_precision_table_from_long_relevance(
    relevance_long: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    rank_col: str = "document_rank",
    relevance_col: str = "relevance",
    k_col: str = "k",
    out_col: str = "precision_at_k",
) -> pd.DataFrame:
    """
    Manuscript-aligned computation of Precision@k from document-level relevance labels.

    Expected long format:
        - scenario_id
        - document_rank (1..k)
        - relevance (0/1)
    Optional:
        - k (if absent, inferred as max(document_rank) per scenario)

    Returns:
        One row per scenario with:
        - scenario_id
        - k
        - precision_at_k
    """
    for c in (scenario_col, rank_col, relevance_col):
        if c not in relevance_long.columns:
            raise ValueError(f"Missing column '{c}' in relevance_long table.")

    df = relevance_long.copy()

    if k_col not in df.columns:
        df[k_col] = df.groupby(scenario_col)[rank_col].transform("max")

    # ensure numeric relevance
    df[relevance_col] = pd.to_numeric(df[relevance_col], errors="coerce")

    out = (
        df.groupby(scenario_col)
        .agg(
            k=(k_col, "max"),
            precision_at_k=(relevance_col, lambda x: float(np.nanmean(x.to_numpy(dtype=float)))),
        )
        .reset_index()
    )
    out.rename(columns={"precision_at_k": out_col}, inplace=True)
    return out


# -------------------------
# Expert aggregation (manuscript-aligned)
# -------------------------

def majority_vote(series: pd.Series) -> Any:
    """
    Deterministic majority vote for categorical/ordinal labels.
    Ties resolved by pandas' idxmax order.
    """
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    return vc.idxmax()


def aggregate_expert_ratings(
    expert_ratings: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    factuality_col: str = "factuality",
    interpretability_col: str = "interpretability",
    out_factuality_consensus: str = "factuality_consensus",
) -> pd.DataFrame:
    """
    Aggregate expert ratings per scenario (manuscript-aligned).

    Outputs:
        - scenario_id
        - n_experts
        - factuality_consensus   (majority vote; 0/1/2)
        - interpretability_mean  (mean)
        - interpretability_sd    (SD)
    Also includes factuality_mean/sd as optional descriptive context.
    """
    for c in (scenario_col, factuality_col, interpretability_col):
        if c not in expert_ratings.columns:
            raise ValueError(f"Missing column '{c}' in expert_ratings table.")

    df = expert_ratings.copy()
    df[factuality_col] = pd.to_numeric(df[factuality_col], errors="coerce")
    df[interpretability_col] = pd.to_numeric(df[interpretability_col], errors="coerce")

    agg = (
        df.groupby(scenario_col)
        .agg(
            n_experts=(interpretability_col, "count"),
            factuality_consensus=(factuality_col, majority_vote),
            factuality_mean=(factuality_col, "mean"),
            factuality_sd=(factuality_col, lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            interpretability_mean=(interpretability_col, "mean"),
            interpretability_sd=(interpretability_col, lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        )
        .reset_index()
    )

    agg.rename(columns={"factuality_consensus": out_factuality_consensus}, inplace=True)
    return agg


def compute_expert_confidence_majority(
    expert_ratings: pd.DataFrame,
    *,
    scenario_col: str = "scenario_id",
    conf_col: str = "expert_confidence",
    out_col: str = "expert_confidence_majority",
) -> pd.DataFrame:
    """
    Majority vote for expert confidence labels per scenario.

    Returns:
        - scenario_id
        - expert_confidence_majority
    """
    for c in (scenario_col, conf_col):
        if c not in expert_ratings.columns:
            raise ValueError(f"Missing column '{c}' in expert_ratings table.")

    maj = (
        expert_ratings.groupby(scenario_col)[conf_col]
        .apply(majority_vote)
        .reset_index()
        .rename(columns={conf_col: out_col})
    )
    return maj


# -------------------------
# Uncertainty alignment (manuscript-aligned): weighted kappa
# -------------------------

def weighted_kappa_confidence(
    system_vs_expert: pd.DataFrame,
    *,
    system_col: str = "system_confidence",
    expert_col: str = "expert_confidence_majority",
    conf_order: Sequence[str] = CONF_ORDER_DEFAULT,
    weights: str = "linear",
) -> Tuple[float, int]:
    """
    Compute weighted Cohen's κ (linear by default) for ordinal confidence labels.

    Returns:
        (kappa, n_items)

    Notes:
        This is an interpretability/alignment metric, not probabilistic calibration.
    """
    for c in (system_col, expert_col):
        if c not in system_vs_expert.columns:
            raise ValueError(f"Missing column '{c}' in system_vs_expert table.")

    mapping = {v: i for i, v in enumerate(conf_order)}
    s = system_vs_expert[system_col].map(mapping)
    e = system_vs_expert[expert_col].map(mapping)

    mask = (~pd.isna(s)) & (~pd.isna(e))
    n_items = int(mask.sum())
    if n_items == 0:
        return float("nan"), 0

    kappa = float(cohen_kappa_score(s[mask], e[mask], weights=weights))
    return kappa, n_items


# -------------------------
# Summary statistics (descriptive)
# -------------------------

def summarize_results(
    results: pd.DataFrame,
    *,
    precision_col: str = "precision_at_k",
    interpretability_mean_col: str = "interpretability_mean",
    kappa_value: Optional[float] = None,
    kappa_n_items: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute overall descriptive summary statistics (mean ± SD) for key metrics.

    Manuscript-aligned summary:
        - Precision@k: mean ± SD across query instances
        - Interpretability mean: mean ± SD across query instances
        - Confidence alignment: weighted κ (linear) reported separately (not an 'alignment rate')

    Returns:
        A single-row DataFrame of summary stats.
    """
    for c in (precision_col, interpretability_mean_col):
        if c not in results.columns:
            raise ValueError(f"Missing column '{c}' in results table.")

    row: Dict[str, Any] = {
        "precision_at_k_mean": float(results[precision_col].mean()),
        "precision_at_k_sd": float(results[precision_col].std(ddof=1)),
        "interpretability_mean": float(results[interpretability_mean_col].mean()),
        "interpretability_sd": float(results[interpretability_mean_col].std(ddof=1)),
    }

    if kappa_value is not None:
        row["confidence_alignment_kappa_weighted"] = float(kappa_value)
    if kappa_n_items is not None:
        row["confidence_alignment_n_items"] = int(kappa_n_items)

    return pd.DataFrame([row])