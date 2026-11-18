"""Predictor-set comparison: which environmental variables explain a species best?

Which climate dimension limits a species differs between species (rainfall for some,
temperature for others, humidity or wind for others, or a combination), and is rarely known
in advance. This module refits a model version's *exact* training locations (presences +
background from its snapshot) with several candidate predictor sets and compares them by
spatial block cross-validation (CBI first, then AUC and TSS; see `evaluation.py`):

* each set is VIF-screened (VIF < 5) on the species' own background, so sets of different
  size are compared without collinearity inflating them;
* the same locations, CV folds, seeds and tuning grid are used for every set — only the
  predictors change;
* sets with present-only predictors (e.g. CHELSA relative humidity, wind) are marked: such a
  model cannot be projected onto future scenarios.

This is a relative comparison on the same (biased) data, plain MaxEnt with the version's
background (presence weights and bias covariates are not applied). Confirm a choice on
independent data (uploaded field records) before relying on it.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.domain import PREDICTOR_GROUP_LABELS, predictor_group
from src.features.collinearity import select_low_collinearity
from src.features.raster_sampler import BioclimMetadata
from src.modeling.maxent_trainer import TrainingConfig, TrainingData, train_maxent

MIN_PRESENCES = 10
# CBI difference below which two sets are reported as performing alike.
CBI_TIE = 0.02


def predictors_by_group(meta: BioclimMetadata) -> dict[str, list[str]]:
    """The version's predictors grouped (temperature, precipitation, humidity, …)."""
    out: dict[str, list[str]] = {}
    for name in meta.predictor_bands:
        out.setdefault(predictor_group(name), []).append(name)
    return out


def default_predictor_sets(
    meta: BioclimMetadata, current: list[str] | None = None
) -> list[dict[str, Any]]:
    """Candidate sets built from the groups the version offers (each VIF-screened later)."""
    groups = predictors_by_group(meta)
    temp, prec = groups.get("temperature", []), groups.get("precipitation", [])
    atmos = [*groups.get("humidity", []), *groups.get("wind", [])]
    water = groups.get("water_balance", [])
    season = groups.get("growing_season", [])
    soil = groups.get("soil", [])
    sets: list[dict[str, Any]] = []

    def add(name: str, label: str, predictors: list[str]) -> None:
        if len(predictors) >= 2 and all(predictors != s["predictors"] for s in sets):
            sets.append({"name": name, "label": label, "predictors": predictors})

    if current:
        add("current", "Current model", list(current))
    add("temperature", "Temperature", temp)
    add("precipitation", "Precipitation", prec)
    add("temperature_precipitation", "Temperature + precipitation", [*temp, *prec])
    if atmos:
        add("humidity_wind", "Humidity + wind", atmos)
        add(
            "climate_humidity_wind",
            "Temperature + precipitation + humidity/wind",
            [*temp, *prec, *atmos],
        )
    if water:
        add(
            "climate_water_balance",
            "Temperature + precipitation + water balance",
            [*temp, *prec, *water],
        )
    if season:
        add(
            "climate_growing_season",
            "Temperature + precipitation + growing season/frost",
            [*temp, *prec, *season],
        )
    if soil:
        add("climate_soil", "Temperature + precipitation + soil", [*temp, *prec, *soil])
    everything = [*temp, *prec, *atmos, *water, *season, *soil]
    if len(groups) > 2:
        add("all", "All variables", everything)
    return sets


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def compare_predictor_sets(
    training: pd.DataFrame,
    sets: list[dict[str, Any]],
    model_type: str,
    cfg: TrainingConfig,
    meta: BioclimMetadata,
    vif_screen: bool = True,
) -> dict[str, Any]:
    """Fit MaxEnt once per set on the same presence/background rows of `training` (columns
    longitude, latitude, role and every predictor used by any set) and rank the sets."""
    is_pres = (training["role"] == "presence").to_numpy()
    present_only = set(meta.present_only_layers()) | {
        n for n, e in meta.climate_layers.items() if not e.get("scenarios")
    }
    results = []
    for spec in sets:
        wanted = [p for p in dict.fromkeys(spec["predictors"]) if p in training.columns]
        row: dict[str, Any] = {
            "name": spec["name"],
            "label": spec.get("label", spec["name"]),
            "requested": list(spec["predictors"]),
            "groups": sorted({PREDICTOR_GROUP_LABELS[predictor_group(p)] for p in wanted}),
        }
        valid = ~training[wanted].isna().any(axis=1).to_numpy() if wanted else np.zeros(0, bool)
        pres, bg = valid & is_pres, valid & ~is_pres
        if len(wanted) < 2 or pres.sum() < MIN_PRESENCES:
            results.append(
                {
                    **row,
                    "status": "skipped",
                    "reason": "fewer than 2 predictors or too few "
                    "presences with values for them",
                }
            )
            continue
        predictors, dropped = wanted, []
        if vif_screen and len(wanted) > 2:
            kept, _ = select_low_collinearity(training.loc[bg, wanted])
            predictors = [p for p in wanted if p in kept]
            dropped = [p for p in wanted if p not in kept]
        data = TrainingData(
            predictors=predictors,
            presence_lon=training["longitude"].to_numpy()[pres],
            presence_lat=training["latitude"].to_numpy()[pres],
            presence_x=training.loc[pres, predictors].reset_index(drop=True),
            background_lon=training["longitude"].to_numpy()[bg],
            background_lat=training["latitude"].to_numpy()[bg],
            background_x=training.loc[bg, predictors].reset_index(drop=True),
            background_method="predictor-set comparison (training-set background)",
        )
        fit = train_maxent(data, model_type, replace(cfg, algorithm="maxent"))  # type: ignore[arg-type]
        cv = fit.cv_summary
        results.append(
            {
                **row,
                "status": "fitted",
                "predictors": predictors,
                "dropped_by_vif": dropped,
                "present_only": [p for p in predictors if p in present_only],
                "n_presence": int(pres.sum()),
                "n_background": int(bg.sum()),
                "feature_classes": fit.feature_classes,
                "beta_multiplier": _finite(fit.beta_multiplier),
                **{
                    k: _finite(cv.get(k))
                    for k in ("cbi_mean", "cbi_std", "auc_mean", "auc_std", "tss_mean", "tss_std")
                },
                "importance": {k: _finite(v) for k, v in fit.importance.items()},
            }
        )
    fitted = [r for r in results if r["status"] == "fitted"]
    fitted.sort(
        key=lambda r: (
            -(r["cbi_mean"] if r["cbi_mean"] is not None else -np.inf),
            -(r["auc_mean"] if r["auc_mean"] is not None else -np.inf),
        )
    )
    for rank, r in enumerate(fitted, start=1):
        r["rank"] = rank
    best = fitted[0] if fitted else None
    ties = (
        [
            r["name"]
            for r in fitted[1:]
            if best["cbi_mean"] is not None
            and r["cbi_mean"] is not None
            and best["cbi_mean"] - r["cbi_mean"] < CBI_TIE
        ]
        if best
        else []
    )
    return {
        "sets": fitted + [r for r in results if r["status"] != "fitted"],
        "best": best["name"] if best else None,
        "similar_to_best": ties,
        "ranking_metric": "spatial block CV CBI (then AUC)",
        "vif_screen": vif_screen,
        "note": (
            "Relative comparison on the same training data (plain MaxEnt, version background; "
            "presence weights and bias covariates not applied). Sets within "
            f"{CBI_TIE} CBI of the best perform alike — prefer the ecologically plausible or "
            "simpler one, and confirm on independent field data. Sets with present-only "
            "predictors cannot be projected onto future climate."
        ),
    }
