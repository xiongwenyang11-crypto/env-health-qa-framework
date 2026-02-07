"""
evaluation.agreement

Inter-rater agreement utilities.

For a 3-expert setup, this module provides pairwise Cohen's kappa
(E1 vs E2, E1 vs E3, E2 vs E3) and reports the mean.

Dependencies: scikit-learn only for cohen_kappa_score.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def pairwise_cohens_kappa(
    expert_ratings: pd.DataFrame,
    label_col: str,
    scenario_col: str = "scenario_id",
    expert_col: str = "expert",
) -> pd.DataFrame:
    """
    Compute pairwise Cohen's kappa among experts for a given label column.

    Expected expert_ratings columns:
        - scenario_id
        - expert (e.g., E1, E2, E3)
        - label_col (e.g., factuality or interpretability)

    Returns:
        DataFrame with columns:
            - label
            - pair
            - kappa
        plus:
            - kappa_mean (same value repeated; convenient for display/export)
    """
    for c in [scenario_col, expert_col, label_col]:
        if c not in expert_ratings.columns:
            raise ValueError(f"Missing column '{c}' in expert_ratings table.")

    wide = expert_ratings.pivot(index=scenario_col, columns=expert_col, values=label_col)
    experts = list(wide.columns)

    if len(experts) < 2:
        raise ValueError("Need at least two experts to compute pairwise kappa.")

    rows = []
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            a = wide[experts[i]].to_numpy()
            b = wide[experts[j]].to_numpy()

            # Remove cases where either expert is missing
            mask = ~pd.isna(a) & ~pd.isna(b)
            a2 = a[mask]
            b2 = b[mask]

            if len(a2) == 0:
                kappa = float("nan")
            else:
                kappa = float(cohen_kappa_score(a2, b2))

            rows.append({"label": label_col, "pair": f"{experts[i]} vs {experts[j]}", "kappa": kappa})

    out = pd.DataFrame(rows)
    out["kappa_mean"] = float(out["kappa"].mean()) if len(out) else float("nan")
    return out
