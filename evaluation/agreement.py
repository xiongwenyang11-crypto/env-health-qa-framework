"""
evaluation.agreement

Inter-rater agreement utilities.

This module computes inter-rater agreement among an arbitrary number of experts
(>=2). It supports pairwise Cohen's kappa and (optionally) weighted Cohen's kappa
for ordinal labels (e.g., confidence: Low/Medium/High).

Dependencies: scikit-learn only for cohen_kappa_score.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


DEFAULT_ORDINAL_ORDERS: Dict[str, List[Any]] = {
    # Common ordinal labels used in this project
    "expert_confidence": ["Low", "Medium", "High"],
    "system_confidence": ["Low", "Medium", "High"],
}


def _encode_ordinal(
    series: pd.Series,
    label_order: Sequence[Any],
) -> pd.Series:
    """
    Encode an ordinal categorical series into integer codes according to label_order.
    Unknown labels become NA.
    """
    mapping = {v: i for i, v in enumerate(label_order)}
    return series.map(mapping)


def _maybe_encode_labels(
    df: pd.DataFrame,
    label_col: str,
    label_order: Optional[Sequence[Any]] = None,
) -> Tuple[pd.DataFrame, Optional[Sequence[Any]]]:
    """
    If label_order is provided (or can be inferred), encode df[label_col] accordingly.
    Returns (df_encoded, used_label_order). If no encoding is applied, used_label_order is None.
    """
    if label_order is None:
        label_order = DEFAULT_ORDINAL_ORDERS.get(label_col)

    if label_order is None:
        return df, None

    out = df.copy()
    out[label_col] = _encode_ordinal(out[label_col], label_order)
    return out, label_order


def pairwise_cohens_kappa(
    expert_ratings: pd.DataFrame,
    label_col: str,
    scenario_col: str = "scenario_id",
    expert_col: str = "expert_id",
    *,
    weights: Optional[str] = None,
    label_order: Optional[Sequence[Any]] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise Cohen's kappa among experts for a given label column.

    Parameters
    ----------
    expert_ratings : pd.DataFrame
        Long-format expert rating table.
        Expected columns include:
          - scenario_col (default: 'scenario_id')
          - expert_col   (default: 'expert_id')
          - label_col    (e.g., 'factuality', 'interpretability', 'expert_confidence')
    label_col : str
        Column containing ratings to compute agreement on.
    scenario_col : str
        Scenario/query identifier column.
    expert_col : str
        Expert identifier column (paper uses 10 experts; demo may use fewer).
    weights : {None, 'linear', 'quadratic'}
        If provided, compute weighted Cohen's kappa (recommended for ordinal scales).
        Use 'linear' for confidence (Low/Medium/High) alignment, consistent with the manuscript.
    label_order : sequence, optional
        If label_col contains ordinal strings (e.g., Low/Medium/High), provide an explicit order.
        If omitted, common orders for known columns may be inferred.
    dropna : bool
        If True, drop scenario rows where either expert has missing label.

    Returns
    -------
    pd.DataFrame
        Columns:
          - label: label_col
          - pair: "<expertA> vs <expertB>"
          - kappa: Cohen's kappa value
          - n_items: number of scenario items used for this pair
          - weights: weights used (None/'linear'/'quadratic')
          - kappa_mean: mean of pairwise kappas (repeated for convenience)
          - kappa_std: std of pairwise kappas (repeated for convenience)
          - n_pairs: number of expert pairs (repeated for convenience)

    Notes
    -----
    - For ordinal labels (e.g., confidence), use weights='linear' and an explicit label_order
      (e.g., ['Low','Medium','High']) to ensure consistent encoding.
    - For interpretability (Likert 1-5) and factuality (0-2), weights='linear' is optional but reasonable.
    """
    for c in [scenario_col, expert_col, label_col]:
        if c not in expert_ratings.columns:
            raise ValueError(f"Missing column '{c}' in expert_ratings table.")

    # Encode ordinal strings if needed (e.g., Low/Medium/High)
    ratings, used_order = _maybe_encode_labels(expert_ratings, label_col, label_order=label_order)

    wide = ratings.pivot(index=scenario_col, columns=expert_col, values=label_col)
    experts = list(wide.columns)

    if len(experts) < 2:
        raise ValueError("Need at least two experts to compute pairwise kappa.")

    rows: List[Dict[str, Any]] = []
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            a = wide[experts[i]].to_numpy()
            b = wide[experts[j]].to_numpy()

            if dropna:
                mask = ~pd.isna(a) & ~pd.isna(b)
                a2 = a[mask]
                b2 = b[mask]
            else:
                a2 = a
                b2 = b

            n_items = int(len(a2))
            if n_items == 0:
                kappa = float("nan")
            else:
                # sklearn supports weights for cohen_kappa_score on ordinal labels
                kappa = float(cohen_kappa_score(a2, b2, weights=weights))

            rows.append(
                {
                    "label": label_col,
                    "pair": f"{experts[i]} vs {experts[j]}",
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

    out["kappa_mean"] = float(out["kappa"].mean())
    out["kappa_std"] = float(out["kappa"].std(ddof=0)) if len(out) > 1 else 0.0
    out["n_pairs"] = int(len(out))
    return out