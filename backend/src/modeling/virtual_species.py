"""Virtual-species benchmark of sampling-bias corrections (§4.6; Leroy et al. 2016).

Bias corrections can look better under cross-validation without being better, because CV
re-uses the same biased data. With a *virtual* species the true niche is known:

1. a Gaussian response to two climate predictors defines the true suitability on the real
   climate grid;
2. presences are sampled with probability ∝ true suitability × observer effort, where effort
   is either synthetic (decaying from a few random "cities") or the real GBIF effort of a
   target group (map API, through the GBIF connector);
3. each correction is fitted with the production MaxEnt trainer and compared with the truth
   (Spearman correlation, Schoener's D, AUC against a realisation of true occupancy), next to
   what spatial CV on the biased data reports.

Variants: no_correction (uniform background), target_group (background drawn with the same
effort), bias_covariate (log effort as a covariate, held at its maximum for prediction) and
density_weights (presences down-weighted where clustered).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from src.features.raster_sampler import BioclimStack
from src.features.sampling_bias import effort_values, presence_weights
from src.modeling.evaluation import auc, rank_correlation, schoeners_d
from src.modeling.maxent_trainer import TrainingConfig, TrainingData, train_maxent

VARIANT_DESCRIPTIONS = {
    "no_correction": "uniform background in the study area; bias ignored",
    "target_group": "background drawn with the same observer effort as the presences",
    "bias_covariate": "log effort as a covariate, held at its maximum for prediction",
    "density_weights": "presences weighted by 1 / local density (B7), uniform background",
}
BIAS_COLUMN = "bias_log_effort"
BENCHMARK_TRAINING = TrainingConfig(
    tune=False, feature_classes="LQ", n_folds=3, importance_repeats=1, min_presences=20
)


def _landscape(
    stack: BioclimStack, predictors: list[str], region: list[float] | None, max_width: int
) -> pd.DataFrame:
    """Cell centres and predictor values of the (decimated) climate grid."""
    arr, t = stack.read_overview(predictors, max_width=max_width)
    nb, h, w = arr.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    lon = t.c + (cols.ravel() + 0.5) * t.a
    lat = t.f + (rows.ravel() + 0.5) * t.e
    x = arr.reshape(nb, -1).T
    ok = ~np.isnan(x).any(axis=1)
    if region is not None:
        west, south, east, north = region
        ok &= (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    out = pd.DataFrame(x[ok], columns=predictors)
    out.insert(0, "latitude", lat[ok])
    out.insert(0, "longitude", lon[ok])
    return out.reset_index(drop=True)


def _synthetic_effort(lon: np.ndarray, lat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Effort decaying exponentially (scale 300 km) from five random 'cities', floor 1%."""
    centres = rng.choice(len(lon), size=min(5, len(lon)), replace=False)
    e = np.zeros(len(lon))
    for c in centres:
        d = 111.32 * np.hypot((lon - lon[c]) * np.cos(np.deg2rad(lat)), lat - lat[c])
        e += np.exp(-d / 300.0)
    return 0.01 + e / e.max()


def run_benchmark(
    stack: BioclimStack,
    request: dict[str, Any],
    fetch_effort: Callable[[int, tuple[float, float, float, float], float], Any] | None = None,
    progress: Callable[[float, str], None] | None = None,
    max_width: int = 720,
) -> dict[str, Any]:
    report = progress or (lambda f, m: None)
    seed = int(request.get("seed", 42))
    rng = np.random.default_rng(seed)
    predictors = list(stack.metadata.selected_predictors or stack.metadata.bands)
    land = _landscape(stack, predictors, request.get("region"), max_width)
    if len(land) < 500:
        raise ValueError(f"Only {len(land)} land cells in the benchmark region")
    lon, lat = land["longitude"].to_numpy(), land["latitude"].to_numpy()

    # 1. virtual species: Gaussian response on the first two selected predictors
    pair = predictors[:2]
    mu = {p: float(np.quantile(land[p], rng.uniform(0.35, 0.65))) for p in pair}
    sd = {p: float(land[p].std() * 0.35) for p in pair}
    truth = np.exp(-0.5 * sum(((land[p] - mu[p]) / sd[p]) ** 2 for p in pair)).to_numpy()
    occupancy = rng.uniform(size=len(truth)) < truth

    # 2. observer effort
    strength = float(request.get("bias_strength", 3.0))
    if request.get("bias") == "gbif_effort":
        if fetch_effort is None:
            raise ValueError("GBIF effort needs the GBIF connector")
        bbox = (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
        eff = fetch_effort(int(request["target_group_key"]), bbox, abs(stack.res[0]))
        raw = np.expm1(effort_values(eff.lon, eff.lat, eff.count, eff.pixel_deg, lon, lat))
        effort = 0.01 + np.log1p(raw) / max(np.log1p(raw).max(), 1e-9)
        bias_info = {
            "source": "gbif_effort",
            "target_group_key": request["target_group_key"],
            "query": eff.query,
            "n_records": int(eff.n_records),
        }
    else:
        effort = _synthetic_effort(lon, lat, rng)
        bias_info = {"source": "synthetic", "rule": "exp(−distance / 300 km) from 5 cities"}
    bias = effort**strength
    bias_info.update(strength=strength, formula="P(record) ∝ true suitability × effort^strength")

    # 3. biased presences
    n = min(int(request.get("n_presences", 300)), int((truth > 0.01).sum()))
    w = truth * bias
    pres_idx = rng.choice(len(land), size=n, replace=False, p=w / w.sum())
    n_bg = min(5000, len(land))
    uniform_bg = rng.choice(len(land), size=n_bg, replace=False)
    biased_bg = rng.choice(len(land), size=n_bg, replace=False, p=bias / bias.sum())
    log_effort = np.log(effort)

    variants = []
    for i, variant in enumerate(request.get("variants") or list(VARIANT_DESCRIPTIONS)):
        report(0.1 + 0.85 * i / 4, f"Fitting {variant}")
        bg_idx = biased_bg if variant == "target_group" else uniform_bg
        pres_x = land.loc[pres_idx, predictors].reset_index(drop=True)
        bg_x = land.loc[bg_idx, predictors].reset_index(drop=True)
        bias_cols: dict[str, float] = {}
        weights = None
        if variant == "bias_covariate":
            pres_x[BIAS_COLUMN] = log_effort[pres_idx]
            bg_x[BIAS_COLUMN] = log_effort[bg_idx]
            bias_cols = {BIAS_COLUMN: float(log_effort.max())}
        if variant == "density_weights":
            weights, _ = presence_weights(lon[pres_idx], lat[pres_idx], None, "density", {}, 5)
        data = TrainingData(
            predictors=predictors,
            presence_lon=lon[pres_idx],
            presence_lat=lat[pres_idx],
            presence_x=pres_x,
            background_lon=lon[bg_idx],
            background_lat=lat[bg_idx],
            background_x=bg_x,
            background_method=variant,
            presence_weight=weights,
            bias_covariates=bias_cols,
        )
        entry: dict[str, Any] = {
            "variant": variant,
            "description": VARIANT_DESCRIPTIONS[variant],
        }
        try:
            res = train_maxent(data, "A", replace(BENCHMARK_TRAINING, seed=seed))
            pred = np.asarray(res.model.predict(land[predictors]), "float64").ravel()
            entry.update(
                spearman_vs_truth=round(rank_correlation(pred, truth), 4),
                schoeners_d=round(schoeners_d(pred, truth), 4),
                auc_vs_true_occupancy=(
                    round(auc(pred[occupancy], pred[~occupancy]), 4)
                    if occupancy.any() and (~occupancy).any()
                    else None
                ),
                cv_cbi_mean=_round(res.cv_summary.get("cbi_mean")),
                cv_auc_mean=_round(res.cv_summary.get("auc_mean")),
            )
        except Exception as exc:  # one failing variant is reported, not fatal
            entry["error"] = f"{type(exc).__name__}: {exc}"
        variants.append(entry)

    scored = [v for v in variants if v.get("spearman_vs_truth") is not None]
    best = max(scored, key=lambda v: v["spearman_vs_truth"]) if scored else None
    cv_best = max(
        (v for v in scored if v.get("cv_cbi_mean") is not None),
        key=lambda v: v["cv_cbi_mean"],
        default=None,
    )
    if best is None:
        text = "No variant could be fitted."
    else:
        text = (
            f"{best['variant']} recovers the true niche best (Spearman "
            f"{best['spearman_vs_truth']:.2f})."
        )
        if cv_best is not None and cv_best["variant"] != best["variant"]:
            text += (
                f" Spatial CV on the biased data would have picked {cv_best['variant']} "
                "instead — CV alone can mislead about bias corrections."
            )
        elif cv_best is not None:
            text += " Spatial CV on the biased data ranks it first as well."
    return {
        "virtual_species": {
            "response": "Gaussian on two predictors",
            "predictors": pair,
            "optimum": {p: round(v, 3) for p, v in mu.items()},
            "tolerance_sd": {p: round(v, 3) for p, v in sd.items()},
            "prevalence": round(float(occupancy.mean()), 4),
            "reference": "Leroy et al. 2016 (virtualspecies)",
        },
        "bias": bias_info,
        "n_presences": n,
        "n_evaluation_cells": len(land),
        "variants": variants,
        "best_variant": best["variant"] if best else None,
        "interpretation": text,
    }


def _round(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if np.isfinite(f) else None
