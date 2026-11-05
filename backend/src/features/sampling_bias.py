"""Explicit sampling-bias handling for presence-background training.

Two complements to the target-group background (background_sampler.py), both off by default
and logged per model version when used:

* **Observation weights** (presence_weights): presences are weighted by their data-quality
  tier (features/quality.py) and by how isolated they are — the mean great-circle distance to
  their k nearest neighbouring presences, relative to the median — so clusters of heavily
  recorded sites count less. This is the same idea as `elapid.distance_weights`, computed on
  the sphere instead of in degrees. Weights are rescaled to mean 1, so the total presence
  weight (and elapid's presence/background class weighting) is unchanged.

* **Bias covariates, then condition** (effort_covariate; Warton et al. 2013, Fithian et al.
  2015): the target group's GBIF record density (log1p records per effort pixel) enters the
  model as a predictor, so the environmental response is estimated net of observer effort.
  For prediction the covariate is held at one constant value (the maximum effort seen in the
  training data by default): suitability "as if sampled everywhere as intensively as the best
  sampled cell". Effort values come from the GBIF map API through the connector, not from a
  raster file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6_371.0088
PRESENCE_WEIGHTINGS = ("none", "quality_tier", "density", "quality_tier_density")
BIAS_COVARIATES = ("none", "target_group_effort")
BIAS_CONDITIONS = ("max", "p95")
EFFORT_COVARIATE = "bias_target_group_effort"
DEFAULT_TIER_WEIGHTS: dict[str, float] = {"A": 1.0, "B": 0.75, "C": 0.5}


def density_factor(lon: Sequence[float], lat: Sequence[float], k: int) -> np.ndarray:
    """Mean distance (km) to the k nearest other presences, divided by its median: > 1 for
    isolated presences, < 1 inside clusters."""
    lon_a, lat_a = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    n = len(lon_a)
    if n < 2:
        return np.ones(n)
    k = max(1, min(k, n - 1))
    pts = np.deg2rad(np.column_stack([lat_a, lon_a]))
    dist, _ = BallTree(pts, metric="haversine").query(pts, k=k + 1)
    mean_km = dist[:, 1:].mean(axis=1) * EARTH_RADIUS_KM
    med = float(np.median(mean_km))
    if med <= 0:
        return np.ones(n)
    return np.clip(mean_km / med, 0.1, 10.0)


def presence_weights(
    lon: Sequence[float],
    lat: Sequence[float],
    tiers: Sequence[str] | None,
    method: str,
    tier_weights: Mapping[str, float] = DEFAULT_TIER_WEIGHTS,
    k: int = 5,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """(weights or None, record for the lineage). None means unweighted training."""
    if method not in PRESENCE_WEIGHTINGS:
        raise ValueError(f"Unknown presence weighting {method!r}; use one of {PRESENCE_WEIGHTINGS}")
    info: dict[str, Any] = {"method": method}
    if method == "none":
        return None, info
    n = len(lon)
    w = np.ones(n, dtype="float64")
    if "quality_tier" in method:
        if tiers is None:
            raise ValueError("quality-tier weights need the records' quality tiers")
        w *= np.array([float(tier_weights.get(str(t), 1.0)) for t in tiers])
        info["tier_weights"] = dict(tier_weights)
        values, counts = np.unique(np.asarray(tiers, dtype=str), return_counts=True)
        info["n_by_tier"] = {str(t): int(c) for t, c in zip(values, counts, strict=True)}
    if "density" in method:
        w *= density_factor(lon, lat, k)
        info["density_neighbors"] = k
    mean = float(w.mean()) if n else 1.0
    if mean > 0:
        w /= mean
    info.update(
        {
            "n_presences": n,
            "rescaled_to_mean": 1.0,
            "min": round(float(w.min()), 4) if n else None,
            "median": round(float(np.median(w)), 4) if n else None,
            "max": round(float(w.max()), 4) if n else None,
        }
    )
    return w, info


def _pixel_keys(lon: np.ndarray, lat: np.ndarray, pixel_deg: float) -> list[tuple[int, int]]:
    cols = np.floor((lon + 180.0) / pixel_deg).astype(int)
    rows = np.floor((90.0 - lat) / pixel_deg).astype(int)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def effort_values(
    effort_lon: Sequence[float],
    effort_lat: Sequence[float],
    effort_count: Sequence[float],
    pixel_deg: float,
    lon: Sequence[float],
    lat: Sequence[float],
) -> np.ndarray:
    """log1p(target-group records) in the effort pixel holding each point; 0 where the group
    has no records. Effort pixels are the GBIF map API's (centre coordinates + counts)."""
    counts: dict[tuple[int, int], float] = {}
    keys = _pixel_keys(
        np.asarray(effort_lon, dtype="float64"), np.asarray(effort_lat, dtype="float64"), pixel_deg
    )
    for key, c in zip(keys, np.asarray(effort_count, dtype="float64").tolist(), strict=True):
        counts[key] = counts.get(key, 0.0) + c
    pts = _pixel_keys(np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64"), pixel_deg)
    return np.log1p(np.array([counts.get(k, 0.0) for k in pts], dtype="float64"))


def conditioning_value(values: np.ndarray, rule: str) -> float:
    """The constant a bias covariate is held at for prediction."""
    if rule not in BIAS_CONDITIONS:
        raise ValueError(f"Unknown bias conditioning {rule!r}; use one of {BIAS_CONDITIONS}")
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0
    return float(v.max() if rule == "max" else np.percentile(v, 95))
