"""Climate-data cross-check: does a model's picture depend on the climate dataset?

A stored model version is refitted on an independent climate source (e.g. CHELSA, which is
better in complex terrain) using the *exact* training set of that version: the same
presences and background locations (from `training_data.parquet`), the same feature classes,
regularisation and CV seed. Only the predictor values change. Reported:

* per-predictor agreement between the two sources at the training locations;
* spatial block CV metrics (AUC/CBI/TSS) of both fits;
* rank correlation of the two suitability predictions and agreement of the thresholded maps.

Low agreement means the projection is sensitive to the climate data — a source of
uncertainty to report next to MESS, not a reason to prefer one dataset automatically.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.features.raster_sampler import BioclimStack
from src.modeling.evaluation import binary_agreement, predictor_agreement, rank_correlation
from src.modeling.maxent_trainer import TrainingConfig, TrainingData, train_maxent

CONSISTENT_RHO = 0.8
CONSISTENT_KAPPA = 0.6
DIVERGENT_RHO = 0.5
# Median magnitudes this far apart indicate different units, not different climates.
UNITS_SUSPECT_RATIO = (0.5, 2.0)


def climate_crosscheck(
    training: pd.DataFrame,
    primary_model: Any,
    primary_threshold: float,
    primary_metrics: dict[str, Any],
    model_type: str,
    alt_stack: BioclimStack,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    predictors = [p for p in primary_metrics["predictors"]]
    alt_x = alt_stack.sample(training["longitude"], training["latitude"], predictors)
    valid = (
        ~alt_x.isna().any(axis=1).to_numpy() & ~training[predictors].isna().any(axis=1).to_numpy()
    )
    is_pres = (training["role"] == "presence").to_numpy()

    agreement = []
    for p in predictors:
        stats = predictor_agreement(training[p].to_numpy()[valid], alt_x[p].to_numpy()[valid])
        lo, hi = UNITS_SUSPECT_RATIO
        ratio = stats["scale_ratio"]
        agreement.append(
            {
                "predictor": p,
                **stats,
                "units_suspect": bool(np.isfinite(ratio) and not lo <= ratio <= hi),
            }
        )

    pres, bg = valid & is_pres, valid & ~is_pres
    data = TrainingData(
        predictors=predictors,
        presence_lon=training["longitude"].to_numpy()[pres],
        presence_lat=training["latitude"].to_numpy()[pres],
        presence_x=alt_x[pres].reset_index(drop=True),
        background_lon=training["longitude"].to_numpy()[bg],
        background_lat=training["latitude"].to_numpy()[bg],
        background_x=alt_x[bg].reset_index(drop=True),
        background_method="crosscheck (training-set background)",
    )
    alt = train_maxent(data, model_type, cfg)  # type: ignore[arg-type]

    x_primary = training.loc[valid, predictors].reset_index(drop=True)
    x_alt = alt_x[valid].reset_index(drop=True)
    pred_primary = np.asarray(primary_model.predict(x_primary), dtype="float64").ravel()
    pred_alt = np.asarray(alt.model.predict(x_alt), dtype="float64").ravel()
    rho = rank_correlation(pred_primary, pred_alt)
    binary = binary_agreement(pred_primary >= primary_threshold, pred_alt >= alt.threshold)

    if rho >= CONSISTENT_RHO and binary["kappa"] >= CONSISTENT_KAPPA:
        verdict = "consistent"
        text = (
            "Both climate datasets give a similar suitability ranking and suitable area: the "
            "projection is robust to the choice of climate data."
        )
    elif not np.isfinite(rho) or rho < DIVERGENT_RHO:
        verdict = "divergent"
        text = (
            "The two climate datasets lead to substantially different suitability patterns: "
            "treat the projection as climate-data-sensitive, especially in mountainous regions."
        )
    else:
        verdict = "moderate"
        text = (
            "Suitability rankings broadly agree but the thresholded suitable area differs: "
            "range edges are sensitive to the climate data."
        )
    if any(a["units_suspect"] for a in agreement):
        text += " Some predictors differ in magnitude between sources (check units)."

    return _finite_tree(
        {
            "alt_bioclim_version": alt_stack.version,
            "alt_source": alt_stack.metadata.source,
            "n_presence": int(pres.sum()),
            "n_background": int(bg.sum()),
            "n_dropped": int((~valid).sum()),
            "feature_classes": cfg.feature_classes,
            "beta_multiplier": cfg.beta_multiplier,
            "predictors": agreement,
            "primary": {k: primary_metrics.get(k) for k in ("auc_mean", "cbi_mean", "tss_mean")},
            "alternative": {
                **{k: alt.cv_summary.get(k) for k in ("auc_mean", "cbi_mean", "tss_mean")},
                "threshold": alt.threshold,
            },
            "suitability_rank_correlation": rho,
            "classification_agreement": binary["agreement"],
            "classification_kappa": binary["kappa"],
            "verdict": verdict,
            "interpretation": text,
        }
    )


def _finite_tree(v: Any) -> Any:
    """NaN/inf → None, recursively (JSON / jsonb cannot store them)."""
    if isinstance(v, dict):
        return {k: _finite_tree(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_finite_tree(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v
