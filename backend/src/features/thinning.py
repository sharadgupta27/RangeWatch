"""Presence thinning beyond a fixed distance (improvements.md B3, B4).

* ``distance`` (default): greedy random-order thinning at the configured `thin_km` — the
  reviewed behaviour, unchanged.
* ``autocorrelation`` (B3; Valavi et al. 2019, blockCV ``cv_spatial_autocor``): the distance is
  the median effective range of exponential variograms fitted to the standardised predictors
  over the study region, clamped to [`thin_km`, `max_km`].
* ``environmental`` (B4; Varela et al. 2014): one presence per cell of a grid in predictor
  space (the first principal axes of the standardised predictors, each cut into `n_bins`
  equal bins), which reduces the over-representation of heavily sampled climates.

`reps` > 1 turns the geographic methods into maximal-retention thinning (spThin; Aiello-
Lammens et al. 2015): the greedy pass is repeated with seeds `seed … seed + reps − 1` and the
run keeping most records wins. elapid 1.0.4 has no thinning helper, so this is small numpy /
scikit-learn code. Every choice and its result is returned for the lineage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import haversine_distances

from src.features.background_sampler import EARTH_RADIUS_KM, distance_thin

THIN_METHODS = ("distance", "autocorrelation", "environmental")
VARIOGRAM_MAX_POINTS = 1500
VARIOGRAM_BINS = 20


def max_retention_thin(
    lon: Sequence[float], lat: Sequence[float], min_distance_km: float, seed: int, reps: int = 1
) -> tuple[np.ndarray, dict[str, Any]]:
    """Best of `reps` greedy random-order passes (most records kept; first wins ties)."""
    best = distance_thin(lon, lat, min_distance_km, seed=seed)
    best_seed, sizes = seed, [len(best)]
    for r in range(1, max(reps, 1)):
        keep = distance_thin(lon, lat, min_distance_km, seed=seed + r)
        sizes.append(len(keep))
        if len(keep) > len(best):
            best, best_seed = keep, seed + r
    info: dict[str, Any] = {"reps": max(reps, 1), "selected_seed": best_seed}
    if reps > 1:
        info.update(
            reference="maximal-retention thinning (spThin; Aiello-Lammens et al. 2015)",
            n_kept_min=int(min(sizes)),
            n_kept_max=int(max(sizes)),
        )
    return best, info


def _exponential(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    # Effective range `rng`: the model reaches 95 % of the sill there.
    return nugget + sill * (1.0 - np.exp(-3.0 * h / rng))


def autocorrelation_range_km(
    lon: Sequence[float],
    lat: Sequence[float],
    x: pd.DataFrame,
    seed: int,
    max_points: int = VARIOGRAM_MAX_POINTS,
    n_bins: int = VARIOGRAM_BINS,
) -> dict[str, Any]:
    """Effective range (km) of an exponential variogram per standardised predictor, fitted
    to the empirical semivariance of up to `max_points` points (lags up to half the largest
    distance), and their median — blockCV's `cv_spatial_autocor` recipe."""
    lon_a, lat_a = np.asarray(lon, "float64"), np.asarray(lat, "float64")
    idx = np.arange(len(lon_a))
    if len(idx) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(len(idx), max_points, replace=False))
    if len(idx) < 30:
        raise ValueError(f"only {len(idx)} reference points for the variogram (need ≥ 30)")
    coords = np.deg2rad(np.column_stack([lat_a[idx], lon_a[idx]]))
    d = haversine_distances(coords) * EARTH_RADIUS_KM
    iu = np.triu_indices(len(idx), k=1)
    dist = d[iu]
    max_lag = float(dist.max()) / 2.0
    edges = np.linspace(0.0, max_lag, n_bins + 1)
    in_range = dist <= max_lag
    bins = np.clip(np.digitize(dist[in_range], edges) - 1, 0, n_bins - 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    counts = np.bincount(bins, minlength=n_bins)
    per_predictor: dict[str, float | None] = {}
    for col in x.columns:
        v = x[col].to_numpy("float64")[idx]
        sd = np.nanstd(v)
        if not np.isfinite(sd) or sd == 0:
            per_predictor[col] = None
            continue
        z = (v - np.nanmean(v)) / sd
        sq = 0.5 * (z[iu[0]] - z[iu[1]]) ** 2
        sq = np.nan_to_num(sq[in_range])
        gamma = np.bincount(bins, weights=sq, minlength=n_bins) / np.maximum(counts, 1)
        ok = counts > 0
        try:
            (_, _, rng), _ = curve_fit(
                _exponential,
                centres[ok],
                gamma[ok],
                p0=(0.0, float(gamma[ok].max()), max_lag / 3.0),
                bounds=([0.0, 1e-9, 1e-3], [np.inf, np.inf, 10.0 * max_lag]),
                maxfev=5000,
            )
            per_predictor[col] = round(float(min(rng, max_lag)), 1)
        except (RuntimeError, ValueError):
            per_predictor[col] = None
    ranges = [r for r in per_predictor.values() if r is not None]
    if not ranges:
        raise ValueError("no predictor variogram could be fitted")
    return {
        "range_km": round(float(np.median(ranges)), 1),
        "range_km_by_predictor": per_predictor,
        "n_points": int(len(idx)),
        "max_lag_km": round(max_lag, 1),
        "model": "exponential variogram (effective range), median over predictors",
        "reference": "Valavi et al. 2019 (blockCV cv_spatial_autocor)",
    }


def environmental_thin(
    x: pd.DataFrame, n_bins: int, seed: int, n_axes: int = 2
) -> tuple[np.ndarray, dict[str, Any]]:
    """One record per occupied cell of an `n_bins`ⁿ grid over the first `n_axes` principal
    axes of the standardised predictors (Varela et al. 2014). Returns sorted indices."""
    arr = x.to_numpy("float64")
    n = len(arr)
    if n == 0:
        return np.arange(0), {}
    sd = arr.std(axis=0)
    sd[sd == 0] = 1.0
    z = (arr - arr.mean(axis=0)) / sd
    k = max(1, min(n_axes, z.shape[1], n))
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    scores = z @ vt[:k].T
    explained = (s**2) / max(float((s**2).sum()), 1e-12)
    lo, hi = scores.min(axis=0), scores.max(axis=0)
    width = np.where(hi > lo, (hi - lo) / n_bins, 1.0)
    cells = np.clip(((scores - lo) / width).astype(int), 0, n_bins - 1)
    key = pd.Series([tuple(c) for c in cells.tolist()])
    order = np.random.default_rng(seed).permutation(n)
    first = ~key.iloc[order].duplicated().to_numpy()
    keep = np.sort(order[first])
    return keep, {
        "reference": "environmental thinning (Varela et al. 2014)",
        "n_axes": int(k),
        "n_bins": int(n_bins),
        "variance_explained": [round(float(e), 4) for e in explained[:k]],
        "n_occupied_cells": int(len(keep)),
    }
