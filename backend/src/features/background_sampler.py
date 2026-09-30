"""Background (pseudo-absence) sampling and spatial thinning.

All randomness takes an explicit seed so that it can be logged per model version
(CLAUDE.md reproducibility requirement). `elapid.sample_geoseries` is not used for the buffer
sampler because it does not accept a seed; elapid 1.0.x also has no distance-thinning helper,
so a small haversine thinning routine is provided here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from rasterio.transform import Affine, rowcol
from shapely.geometry import box
from sklearn.neighbors import BallTree

from src.features.raster_sampler import BioclimStack

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE = 111.32


def cell_dedupe(lon: Sequence[float], lat: Sequence[float], transform: Affine) -> np.ndarray:
    """Indices keeping one point per raster cell (first occurrence wins)."""
    rows, cols = rowcol(transform, np.asarray(lon), np.asarray(lat))
    keys = pd.Series(list(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist(), strict=True)))
    return np.flatnonzero(~keys.duplicated().to_numpy())


def distance_thin(
    lon: Sequence[float], lat: Sequence[float], min_distance_km: float, seed: int
) -> np.ndarray:
    """Greedy random-order thinning so no two kept points are closer than `min_distance_km`.

    Returns sorted indices of retained points.
    """
    lon_a, lat_a = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    n = lon_a.size
    if n == 0 or min_distance_km <= 0:
        return np.arange(n)
    coords = np.deg2rad(np.column_stack([lat_a, lon_a]))
    tree = BallTree(coords, metric="haversine")
    radius = min_distance_km / EARTH_RADIUS_KM
    order = np.random.default_rng(seed).permutation(n)
    removed = np.zeros(n, dtype=bool)
    kept: list[int] = []
    for i in order:
        if removed[i]:
            continue
        kept.append(int(i))
        removed[tree.query_radius(coords[i : i + 1], r=radius)[0]] = True
    return np.sort(np.asarray(kept, dtype=int))


@dataclass(frozen=True)
class BackgroundSample:
    lon: np.ndarray
    lat: np.ndarray
    features: pd.DataFrame
    method: str
    seed: int


def buffer_region(lon: Sequence[float], lat: Sequence[float], buffer_km: float) -> shapely.Geometry:
    """Union of circular-ish buffers around presences (degrees, clipped to the globe)."""
    pts = shapely.points(np.asarray(lon), np.asarray(lat))
    deg = buffer_km / KM_PER_DEGREE
    region = shapely.union_all(shapely.buffer(pts, deg, quad_segs=8))
    return shapely.intersection(region, box(-180, -90, 180, 90))


def sample_buffered_background(
    presence_lon: Sequence[float],
    presence_lat: Sequence[float],
    stack: BioclimStack,
    n: int,
    buffer_km: float,
    seed: int,
    bands: Sequence[str],
    max_rounds: int = 50,
) -> BackgroundSample:
    """Uniform (area-corrected) random background inside a buffer around presences, on land."""
    region = buffer_region(presence_lon, presence_lat, buffer_km)
    shapely.prepare(region)
    minx, miny, maxx, maxy = region.bounds
    rng = np.random.default_rng(seed)
    got_lon: list[np.ndarray] = []
    got_lat: list[np.ndarray] = []
    got_feat: list[pd.DataFrame] = []
    have = 0
    for _ in range(max_rounds):
        m = max(1000, 3 * (n - have))
        x = rng.uniform(minx, maxx, m)
        y = rng.uniform(miny, maxy, m)
        # Equal-area correction: accept with probability cos(lat).
        keep = rng.uniform(0, 1, m) < np.cos(np.deg2rad(y))
        x, y = x[keep], y[keep]
        inside = shapely.contains_xy(region, x, y)
        x, y = x[inside], y[inside]
        feats = stack.sample(x, y, bands)
        valid = ~feats.isna().any(axis=1).to_numpy()
        got_lon.append(x[valid])
        got_lat.append(y[valid])
        got_feat.append(feats[valid])
        have += int(valid.sum())
        if have >= n:
            break
    lon_a = np.concatenate(got_lon)[:n]
    lat_a = np.concatenate(got_lat)[:n]
    feats_df = pd.concat(got_feat, ignore_index=True).iloc[:n].reset_index(drop=True)
    if len(lon_a) == 0:
        raise ValueError("No valid background cells found inside the buffer region")
    return BackgroundSample(lon_a, lat_a, feats_df, f"buffer_{int(buffer_km)}km", seed)


def sample_target_group_background(
    bias_lon: Sequence[float],
    bias_lat: Sequence[float],
    stack: BioclimStack,
    n: int,
    seed: int,
    bands: Sequence[str],
) -> BackgroundSample:
    """Target-group background: draw from occurrences of related taxa sampled by the same
    observers/methods, which mirrors the sampling bias of the presences (preferred when a
    target group is available)."""
    idx = cell_dedupe(bias_lon, bias_lat, stack.transform)
    lon_a = np.asarray(bias_lon)[idx]
    lat_a = np.asarray(bias_lat)[idx]
    rng = np.random.default_rng(seed)
    take = rng.choice(len(lon_a), size=min(n, len(lon_a)), replace=False)
    feats = stack.sample(lon_a[take], lat_a[take], bands)
    valid = ~feats.isna().any(axis=1).to_numpy()
    return BackgroundSample(
        lon_a[take][valid],
        lat_a[take][valid],
        feats[valid].reset_index(drop=True),
        "target_group",
        seed,
    )
