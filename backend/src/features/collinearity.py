"""Variance-inflation-factor predictor screening (computed once per bioclim version, globally)."""

from __future__ import annotations

import numpy as np
import pandas as pd

VIF_THRESHOLD = 5.0


def variance_inflation_factors(df: pd.DataFrame) -> pd.Series:
    """VIF_i = 1 / (1 - R²_i), regressing each predictor on all others.

    Uses least squares rather than inverting the correlation matrix, so exact linear
    dependencies (e.g. BIO7 = BIO5 - BIO6) yield VIF = inf instead of a meaningless
    pseudo-inverse value.
    """
    x = df.to_numpy(dtype="float64")
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    n, p = x.shape
    out = np.empty(p)
    for j in range(p):
        others = np.delete(x, j, axis=1)
        design = np.c_[np.ones(n), others]
        coef, *_ = np.linalg.lstsq(design, x[:, j], rcond=None)
        resid = x[:, j] - design @ coef
        r2 = 1.0 - (resid @ resid) / (x[:, j] @ x[:, j])
        out[j] = np.inf if r2 >= 1.0 - 1e-10 else 1.0 / (1.0 - r2)
    return pd.Series(out, index=df.columns, dtype="float64")


def select_low_collinearity(
    df: pd.DataFrame, threshold: float = VIF_THRESHOLD, keep_first: tuple[str, ...] = ()
) -> tuple[list[str], dict[str, float]]:
    """Iteratively drop the highest-VIF predictor until all VIF < threshold.

    `keep_first` predictors are never dropped (e.g. an expert-chosen core set).
    Returns (retained predictors, final VIF per retained predictor).
    """
    cols = [c for c in df.columns if df[c].std() > 0]
    while len(cols) > 2:
        vif = variance_inflation_factors(df[cols])
        droppable = vif.drop(labels=[c for c in keep_first if c in vif.index])
        if droppable.empty or droppable.max() < threshold:
            return cols, {k: round(float(v), 3) for k, v in vif.items()}
        cols.remove(str(droppable.idxmax()))
    vif = variance_inflation_factors(df[cols])
    return cols, {k: round(float(v), 3) for k, v in vif.items()}
