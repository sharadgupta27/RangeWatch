"""Response curves (partial dependence) for any fitted model (U4; Elith et al. 2005, 2010).

For each predictor the curve is the mean prediction over a fixed sample of background points
with that predictor set to each grid value — the marginal effect, averaged over the observed
combinations of the other predictors. Works for MaxEnt, ensembles, integrated models and
bias-conditioned models alike, because it only calls `model.predict` on climate predictors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

RESPONSE_METHOD = (
    "partial dependence: mean prediction over {n} background points with one predictor set "
    "to each value (others as observed)"
)


def response_curves(
    model: Any,
    presence_x: pd.DataFrame,
    background_x: pd.DataFrame,
    predictors: Sequence[str],
    n_points: int = 40,
    n_rows: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bg = background_x[list(predictors)].reset_index(drop=True)
    rows = bg.iloc[rng.choice(len(bg), size=min(n_rows, len(bg)), replace=False)]
    rows = rows.reset_index(drop=True)
    both = pd.concat([presence_x[list(predictors)], bg], ignore_index=True)
    curves = []
    for p in predictors:
        lo, hi = np.nanpercentile(both[p], [1, 99])
        grid = np.linspace(lo, hi, n_points)
        frame = pd.concat([rows] * n_points, ignore_index=True)
        frame[p] = np.repeat(grid, len(rows))
        pred = np.asarray(model.predict(frame), dtype="float64").reshape(n_points, len(rows))
        p5, p95 = np.nanpercentile(presence_x[p], [5, 95])
        curves.append(
            {
                "predictor": p,
                "x": [round(float(v), 4) for v in grid],
                "y": [round(float(v), 5) for v in pred.mean(axis=1)],
                "presence_range": [round(float(p5), 4), round(float(p95), 4)],
            }
        )
    return {"method": RESPONSE_METHOD.format(n=len(rows)), "curves": curves}
