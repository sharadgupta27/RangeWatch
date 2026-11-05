"""Background (pseudo-absence) sampling and spatial thinning.

All randomness takes an explicit seed so that it can be logged per model version
(CLAUDE.md reproducibility requirement). `elapid.sample_geoseries` is not used for the buffer
sampler because it does not accept a seed; elapid 1.0.x also has no distance-thinning helper,
so a small haversine thinning routine is provided here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    # Distinct bioclim cells in the sample (weighted target-group samples may repeat cells).
    n_cells: int | None = None


# How target-group cells are drawn: each cell with effort once ("unique_cells", Phillips et al.
# 2009), or with probability proportional to its record count — a bias file — or to its square
# root (a compromise when effort spans orders of magnitude).
TARGET_GROUP_WEIGHTINGS = ("unique_cells", "record_count", "sqrt_record_count")


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
    region: shapely.Geometry | None = None,
) -> BackgroundSample:
    """Uniform (area-corrected) random background inside a buffer around presences, on land.
    `region` overrides the buffer (e.g. the buffer clipped to a regional study area)."""
    region = region if region is not None else buffer_region(presence_lon, presence_lat, buffer_km)
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
    region: shapely.Geometry | None = None,
    method: str = "target_group",
    counts: Sequence[float] | None = None,
    weighting: str = "unique_cells",
) -> BackgroundSample:
    """Target-group background: draw from occurrences of related taxa sampled by the same
    observers/methods, which mirrors the sampling bias of the presences (preferred when a
    target group is available).

    Locations are de-duplicated to bioclim cells first. With `weighting="unique_cells"` every
    cell with recorded effort is equally likely (unique localities, as in Phillips et al.
    2009), so dense hotspots do not swamp the background. With "record_count" (or
    "sqrt_record_count") cells are drawn with replacement, with probability proportional to
    the target group's record count per cell (or its square root) from `counts` — the bias-
    file approach, which keeps the full effort gradient. `region` restricts the draw to the
    same area the buffered alternative would use.
    """
    if weighting not in TARGET_GROUP_WEIGHTINGS:
        raise ValueError(f"weighting must be one of {TARGET_GROUP_WEIGHTINGS}, got {weighting!r}")
    lon_in = np.asarray(bias_lon, dtype="float64")
    lat_in = np.asarray(bias_lat, dtype="float64")
    cnt_in = np.asarray(counts, dtype="float64") if counts is not None else np.ones(len(lon_in))
    if region is not None:
        inside = shapely.contains_xy(region, lon_in, lat_in)
        lon_in, lat_in, cnt_in = lon_in[inside], lat_in[inside], cnt_in[inside]
    idx = cell_dedupe(lon_in, lat_in, stack.transform)
    lon_a = lon_in[idx]
    lat_a = lat_in[idx]
    rng = np.random.default_rng(seed)
    if weighting == "unique_cells":
        take = rng.choice(len(lon_a), size=min(n, len(lon_a)), replace=False)
    else:
        # Effort per cell: sum of the pixel counts falling in it (first pixel gives the point).
        rows, cols = rowcol(stack.transform, lon_in, lat_in)
        codes, _ = pd.factorize(
            pd.Series(list(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist(), strict=True)))
        )
        per_cell = np.bincount(codes, weights=np.clip(cnt_in, 0, None), minlength=len(idx))
        w = np.sqrt(per_cell) if weighting == "sqrt_record_count" else per_cell
        if len(lon_a) == 0 or w.sum() <= 0:
            take = np.array([], dtype=int)
        else:
            take = rng.choice(len(lon_a), size=n, replace=True, p=w / w.sum())
    feats = stack.sample(lon_a[take], lat_a[take], bands)
    valid = ~feats.isna().any(axis=1).to_numpy()
    return BackgroundSample(
        lon_a[take][valid],
        lat_a[take][valid],
        feats[valid].reset_index(drop=True),
        method,
        seed,
        n_cells=len(np.unique(take[valid])),
    )


# ---------------------------------------------------------------------------
# Pseudo-absences for presence–absence algorithms (§3.3; Barbet-Massin et al. 2012)
# ---------------------------------------------------------------------------
# background: the MaxEnt background reused · disk: uniform in the buffer, at least
# `min_distance_km` from any presence · environmental_profile: outside the presences' climatic
# envelope (elapid.NicheEnvelopeModel, 2.5–97.5 percentiles of every predictor; Senay et al.
# 2013) · completeness: well-surveyed cells (target-group records ≥ `min_records`) without
# a presence nearby — inferred absences (Lobo et al. 2018), see `sample_completeness_absences`.
PSEUDO_ABSENCE_STRATEGIES = ("background", "disk", "environmental_profile", "completeness")


def distance_to_nearest_km(
    lon: Sequence[float], lat: Sequence[float], ref_lon: Sequence[float], ref_lat: Sequence[float]
) -> np.ndarray:
    """Great-circle distance from each point to the nearest reference point."""
    pts = np.deg2rad(np.column_stack([np.asarray(lat, "float64"), np.asarray(lon, "float64")]))
    ref = np.deg2rad(
        np.column_stack([np.asarray(ref_lat, "float64"), np.asarray(ref_lon, "float64")])
    )
    if len(pts) == 0:
        return np.zeros(0)
    if len(ref) == 0:
        return np.full(len(pts), np.inf)
    d, _ = BallTree(ref, metric="haversine").query(pts, k=1)
    return d[:, 0] * EARTH_RADIUS_KM


def _filtered_buffer_sample(
    presence_lon: Sequence[float],
    presence_lat: Sequence[float],
    stack: BioclimStack,
    n: int,
    buffer_km: float,
    seed: int,
    bands: Sequence[str],
    keep: Callable[[np.ndarray, np.ndarray, pd.DataFrame], np.ndarray],
    method: str,
    region: shapely.Geometry | None,
    max_rounds: int = 8,
) -> BackgroundSample:
    lon_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []
    feat_parts: list[pd.DataFrame] = []
    have = 0
    for r in range(max_rounds):
        cand = sample_buffered_background(
            presence_lon,
            presence_lat,
            stack,
            max(2 * n, 2000),
            buffer_km,
            seed + r,
            bands,
            region=region,
        )
        ok = keep(cand.lon, cand.lat, cand.features)
        lon_parts.append(cand.lon[ok])
        lat_parts.append(cand.lat[ok])
        feat_parts.append(cand.features[ok])
        have += int(ok.sum())
        if have >= n:
            break
    if have == 0:
        raise ValueError(f"No pseudo-absence candidates left for {method}")
    return BackgroundSample(
        np.concatenate(lon_parts)[:n],
        np.concatenate(lat_parts)[:n],
        pd.concat(feat_parts, ignore_index=True).iloc[:n].reset_index(drop=True),
        method,
        seed,
    )


def sample_disk_pseudo_absences(
    presence_lon: Sequence[float],
    presence_lat: Sequence[float],
    stack: BioclimStack,
    n: int,
    buffer_km: float,
    min_distance_km: float,
    seed: int,
    bands: Sequence[str],
    region: shapely.Geometry | None = None,
) -> BackgroundSample:
    """Uniform in the accessible area (buffer) but at least `min_distance_km` from every
    presence, so pseudo-absences do not fall in occupied habitat."""
    return _filtered_buffer_sample(
        presence_lon,
        presence_lat,
        stack,
        n,
        buffer_km,
        seed,
        bands,
        lambda lo, la, _: distance_to_nearest_km(lo, la, presence_lon, presence_lat)
        >= min_distance_km,
        f"disk_{int(min_distance_km)}-{int(buffer_km)}km",
        region,
    )


def sample_profile_pseudo_absences(
    presence_lon: Sequence[float],
    presence_lat: Sequence[float],
    presence_x: pd.DataFrame,
    stack: BioclimStack,
    n: int,
    buffer_km: float,
    seed: int,
    bands: Sequence[str],
    percentile_range: tuple[float, float] = (2.5, 97.5),
    region: shapely.Geometry | None = None,
) -> BackgroundSample:
    """Environmental profiling: candidates inside the presences' climatic envelope (every
    predictor within its presence percentile range) are dropped. Can inflate accuracy."""
    import elapid

    env = elapid.NicheEnvelopeModel(percentile_range=percentile_range, overlay="intersection")
    x = presence_x[list(bands)]
    env.fit(x, np.ones(len(x)))
    return _filtered_buffer_sample(
        presence_lon,
        presence_lat,
        stack,
        n,
        buffer_km,
        seed,
        bands,
        lambda lo, la, feats: np.asarray(env.predict(feats[list(bands)])).ravel() < 0.5,
        "environmental_profile",
        region,
    )


def sample_completeness_absences(
    effort_lon: Sequence[float],
    effort_lat: Sequence[float],
    effort_count: Sequence[float],
    presence_lon: Sequence[float],
    presence_lat: Sequence[float],
    stack: BioclimStack,
    n: int,
    seed: int,
    bands: Sequence[str],
    min_records: float,
    min_distance_km: float,
    region: shapely.Geometry | None = None,
) -> tuple[BackgroundSample, dict[str, int]]:
    """Inferred absences: effort pixels where the target group has ≥ `min_records` GBIF
    records (a completeness proxy: the GBIF map API gives record counts, not species lists)
    but no presence of the species lies within `min_distance_km`. Assumes detectability like
    the target group's; weaker than surveyed absences, stronger than random background."""
    lon = np.asarray(effort_lon, dtype="float64")
    lat = np.asarray(effort_lat, dtype="float64")
    cnt = np.asarray(effort_count, dtype="float64")
    counts = {"n_effort_pixels": len(lon)}
    if region is not None and len(lon):
        inside = shapely.contains_xy(region, lon, lat)
        lon, lat, cnt = lon[inside], lat[inside], cnt[inside]
    well = cnt >= min_records
    counts["n_well_sampled"] = int(well.sum())
    lon, lat = lon[well], lat[well]
    far = distance_to_nearest_km(lon, lat, presence_lon, presence_lat) >= min_distance_km
    lon, lat = lon[far], lat[far]
    counts["n_without_presence"] = len(lon)
    idx = cell_dedupe(lon, lat, stack.transform) if len(lon) else np.array([], dtype=int)
    lon, lat = lon[idx], lat[idx]
    rng = np.random.default_rng(seed)
    take = rng.choice(len(lon), size=min(n, len(lon)), replace=False) if len(lon) else []
    feats = stack.sample(lon[take], lat[take], bands)
    valid = ~feats.isna().any(axis=1).to_numpy()
    counts["n_sampled"] = int(valid.sum())
    sample = BackgroundSample(
        lon[take][valid],
        lat[take][valid],
        feats[valid].reset_index(drop=True),
        f"completeness_{int(min_records)}rec",
        seed,
        n_cells=int(valid.sum()),
    )
    return sample, counts
