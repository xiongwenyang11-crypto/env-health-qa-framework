"""
Inter-rater agreement utilities (pairwise Cohen's kappa).

This module computes pairwise Cohen's κ across experts for a given label column.
It supports optional weighted κ for ordinal labels (e.g., confidence: Low/Medium/High),
consistent with the accompanying manuscript's use of linear-weighted κ for ordinal alignment.

Dependencies: scikit-learn only for cohen_kappa_score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# Canonical ordinal orders used in the manuscript/repo.
DEFAULT_ORDINAL_ORDERS: Dict[str, List[Any]] = {
    "expert_confidence": ["Low", "Medium", "High"],
    "system_confidence": ["Low", "Medium", "High"],
}


def _encode_ordinal(series: pd.Series, label_order: Sequence[Any]) -> pd.Series:
    """
    Encode an ordinal categorical series into integer codes according to label_order.
    Unknown labels become NA.
    """
    mapping = {v: i for i, v in enumerate(label_order)}
    return series.map(mapping)


def _maybe_encode(
    df: pd.DataFrame, label_col: str, label_order: Optional[Sequence[Any]]
) -> Tuple[pd.DataFrame, Optional[Sequence[Any]]]:
    """
    If label_order is provided (or inferred), encode df[label_col] accordingly.
    Returns encoded df and the order used; if no encoding applied, returns (df, None).
    """
    if label_order is None:
        label_order = DEFAULT_ORDINAL_ORDERS.get(label_col)

    if label_order is None:
        return df, None

    out = df.copy()
    out[label_col] = _encode_ordinal(out[label_col], label_order)
    return out, label_order


def pairwise_cohens_kappa(
    df: pd.DataFrame,
    *,
    label_col: str,
    scenario_col: str = "scenario_id",
    expert_col: str = "expert_id",
    weights: Optional[str] = None,
    label_order: Optional[Sequence[Any]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise Cohen's κ for a given label across experts.

    Parameters
    ----------
    df:
        Long-form DataFrame with columns: scenario_col, expert_col, label_col.
    label_col:
        Name of the column containing ratings (e.g., factuality, interpretability, expert_confidence).
    scenario_col:
        Scenario identifier column.
    expert_col:
        Expert identifier column (paper uses 10 experts; demo may use fewer).
    weights:
        Optional κ weighting scheme for ordinal labels: None, 'linear', or 'quadratic'.
        - Use 'linear' for ordinal scales such as confidence (Low/Medium/High), consistent with manuscript.
    label_order:
        Explicit ordinal order for string labels (e.g., ['Low','Medium','High']).
        If omitted, common orders for known columns may be inferred.
    dropna:
        If True, drop scenarios where either expert has missing values for the label.

    Returns
    -------
    pd.DataFrame
        Columns:
        - label: label_col
        - pair: "E1 vs E2"
        - kappa: Cohen's κ for that pair
        - n_items: number of scenarios used for that pair
        - weights: weighting scheme used
        - kappa_mean: mean κ across all pairs (repeated for convenience)
        - kappa_std: std κ across all pairs (repeated for convenience)
        - n_pairs: number of expert pairs (repeated for convenience)

    Notes
    -----
    - For ordinal labels, provide weights='linear' and a stable label_order to avoid inconsistent mappings.
    - This function is intended for descriptive agreement reporting (pairwise κ summaries).
    """
    for c in (scenario_col, expert_col, label_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in df.")

    df2, _used_order = _maybe_encode(df, label_col=label_col, label_order=label_order)

    wide = df2.pivot(index=scenario_col, columns=expert_col, values=label_col)
    experts: List[str] = [str(c) for c in wide.columns.tolist()]

    if len(experts) < 2:
        raise ValueError("Need at least two experts to compute pairwise Cohen's kappa.")

    rows: List[dict] = []
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            e1, e2 = experts[i], experts[j]
            a = wide[e1].to_numpy()
            b = wide[e2].to_numpy()

            if dropna:
                mask = (~pd.isna(a)) & (~pd.isna(b))
                a2 = a[mask]
                b2 = b[mask]
            else:
                a2 = a
                b2 = b

            n_items = int(len(a2))
            if n_items == 0:
                kappa = float("nan")
            else:
                kappa = float(cohen_kappa_score(a2, b2, weights=weights))

            rows.append(
                {
                    "label": label_col,
                    "pair": f"{e1} vs {e2}",
                    "kappa": kappa,
                    "n_items": n_items,
                    "weights": weights,
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        out["kappa_mean"] = float("nan")
        out["kappa_std"] = float("nan")
        out["n_pairs"] = 0
        return out

    kappas = out["kappa"].to_numpy(dtype=float)
    out["kappa_mean"] = float(np.nanmean(kappas))
    out["kappa_std"] = float(np.nanstd(kappas))
    out["n_pairs"] = int(len(out))
    return out