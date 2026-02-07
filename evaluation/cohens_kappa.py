"""
Inter-rater agreement utilities (pairwise Cohen's kappa).

This module computes pairwise Cohen's κ across experts for a given label column.
It is intentionally lightweight and reproducible.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def pairwise_cohens_kappa(
    df: pd.DataFrame,
    *,
    label_col: str,
    scenario_col: str = "scenario_id",
    expert_col: str = "expert",
) -> pd.DataFrame:
    """
    Compute pairwise Cohen's κ for a given label across experts.

    Parameters
    ----------
    df:
        Long-form DataFrame with columns: scenario_col, expert_col, label_col.
    label_col:
        Name of the column containing ratings (e.g., factuality, interpretability).
    scenario_col:
        Scenario identifier column.
    expert_col:
        Expert identifier column.

    Returns
    -------
    pd.DataFrame
        Columns:
        - label: label_col
        - pair: "E1 vs E2"
        - kappa: Cohen's κ for that pair
        - kappa_mean: mean κ across all pairs (repeated for convenience)
    """
    for c in (scenario_col, expert_col, label_col):
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in df.")

    wide = df.pivot(index=scenario_col, columns=expert_col, values=label_col)

    experts: List[str] = [str(c) for c in wide.columns.tolist()]
    if len(experts) < 2:
        raise ValueError("Need at least two experts to compute pairwise Cohen's kappa.")

    rows = []
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            e1, e2 = experts[i], experts[j]
            a = wide[e1].to_numpy()
            b = wide[e2].to_numpy()

            # Remove scenarios where either is missing (NaN)
            mask = (~pd.isna(a)) & (~pd.isna(b))
            if mask.sum() == 0:
                kappa = float("nan")
            else:
                kappa = float(cohen_kappa_score(a[mask], b[mask]))

            rows.append({"label": label_col, "pair": f"{e1} vs {e2}", "kappa": kappa})

    out = pd.DataFrame(rows)
    out["kappa_mean"] = float(np.nanmean(out["kappa"].to_numpy())) if len(out) else float("nan")
    return out
