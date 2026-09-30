"""Native-range handling.

CLAUDE.md constraint 6: the native range is never fully automated. This module only proposes
a *draft* polygon (clearly labelled with the heuristic used); training is gated on a user
confirming (and optionally editing) it in the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from sklearn.cluster import DBSCAN

from src.features.background_sampler import EARTH_RADIUS_KM, buffer_region

MIN_LABELLED_NATIVE = 5


@dataclass(frozen=True)
class NativeRangeProposal:
    geometry: dict[str, Any]
    source: str
    note: str
    n_points_used: int


def to_multipolygon(geom: shapely.Geometry) -> MultiPolygon:
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    if not polys:
        raise ValueError("Geometry contains no polygons")
    return MultiPolygon(polys)


def validate_geojson_polygon(geojson: dict[str, Any]) -> dict[str, Any]:
    """Validate + normalise a user-supplied (Multi)Polygon to a valid MultiPolygon GeoJSON."""
    geom = shape(geojson)
    if geom.is_empty:
        raise ValueError("Native-range geometry is empty")
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    minx, miny, maxx, maxy = geom.bounds
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
        raise ValueError("Native-range geometry must be in EPSG:4326 lon/lat degrees")
    return mapping(to_multipolygon(geom))


def _largest_cluster(lon: np.ndarray, lat: np.ndarray, eps_km: float) -> np.ndarray:
    # De-duplicate to a 0.5° grid first so DBSCAN stays cheap for large species.
    cells = pd.DataFrame({"x": np.round(lon * 2) / 2, "y": np.round(lat * 2) / 2})
    uniq = cells.drop_duplicates().to_numpy()
    labels = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM, min_samples=3, metric="haversine", algorithm="ball_tree"
    ).fit_predict(np.deg2rad(uniq[:, ::-1]))
    valid = labels[labels >= 0]
    if valid.size == 0:
        return uniq
    best = np.bincount(valid).argmax()
    return uniq[labels == best]


def propose_native_range(
    occ: pd.DataFrame, buffer_km: float = 150.0, cluster_eps_km: float = 750.0
) -> NativeRangeProposal:
    """Draft native-range polygon from occurrences (columns: longitude, latitude, range_label)."""
    if occ.empty:
        raise ValueError("No occurrences to derive a native-range draft from")
    native = occ[occ["range_label"] == "native"]
    if len(native) >= MIN_LABELLED_NATIVE:
        pts = native[["longitude", "latitude"]].to_numpy()
        source = "heuristic:gbif_establishment_means_native"
        note = (
            f"Buffered ({buffer_km:.0f} km) extent of {len(native)} records flagged native via "
            "GBIF establishmentMeans. These flags are sparse and may be incomplete — review "
            "and edit before confirming."
        )
    else:
        cand = occ[occ["range_label"] != "introduced"]
        if cand.empty:
            cand = occ
        pts = _largest_cluster(
            cand["longitude"].to_numpy(), cand["latitude"].to_numpy(), cluster_eps_km
        )
        source = "heuristic:largest_occurrence_cluster"
        note = (
            "No reliable native/introduced flags were available; this draft is the largest "
            f"spatial cluster of records (DBSCAN, eps={cluster_eps_km:.0f} km), buffered by "
            f"{buffer_km:.0f} km. The largest cluster is NOT necessarily the native range "
            "(heavily invaded regions are often better sampled) — consult GISD/POWO/literature "
            "and edit before confirming."
        )
    region = buffer_region(pts[:, 0], pts[:, 1], buffer_km).simplify(0.1, preserve_topology=True)
    return NativeRangeProposal(
        geometry=mapping(to_multipolygon(region)), source=source, note=note, n_points_used=len(pts)
    )


def effective_range_labels(occ: pd.DataFrame, native_geojson: dict[str, Any]) -> pd.Series:
    """Training-time labels, computed on the fly (occurrence rows are never mutated).

    Explicit GBIF/iNat labels win; unlabelled records are native inside the confirmed
    polygon and introduced outside it.
    """
    geom = shape(native_geojson)
    shapely.prepare(geom)
    inside = shapely.contains_xy(geom, occ["longitude"].to_numpy(), occ["latitude"].to_numpy())
    derived = np.where(inside, "native", "introduced")
    explicit = occ["range_label"].to_numpy()
    return pd.Series(
        np.where(explicit == "unknown", derived, explicit), index=occ.index, name="effective_label"
    )
