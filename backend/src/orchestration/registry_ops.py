"""Lightweight, synchronous registry operations used by the API (not long-running).

Kept separate from pipeline.py so API workers never import the modeling stack (elapid).
"""

from __future__ import annotations

from typing import Any

from shapely.geometry import shape

from src.domain import NativeRange, SpeciesStatus, utcnow
from src.modeling.native_range import validate_geojson_polygon
from src.modeling.severity_index import SeverityConfig, SeverityInputs, compute_severity
from src.persistence.repository import Repository


def update_native_range(
    repo: Repository,
    taxon_key: int,
    geometry: dict[str, Any],
    confirm: bool,
    note: str | None = None,
) -> NativeRange:
    species = repo.get_species(taxon_key)
    if species is None:
        raise KeyError(taxon_key)
    geom = validate_geojson_polygon(geometry)
    prev = repo.get_native_range(taxon_key)
    # Compare geometrically: PostGIS round-trips change float formatting / nesting.
    changed = prev is None or not shape(prev.geometry).equals_exact(shape(geom), tolerance=1e-7)
    if changed:
        source = "user_edit"
        default_note = f"User-edited from {prev.source}." if prev else "User-drawn polygon."
    else:
        # Confirming an unedited draft keeps its provenance (e.g. which heuristic proposed it).
        source = prev.source
        default_note = prev.note
    nr = repo.save_native_range(
        NativeRange(
            taxon_key=taxon_key,
            geometry=geom,
            status="confirmed" if confirm else "draft",
            source=source,
            note=note if note is not None else default_note,
            confirmed_ts=utcnow() if confirm else None,
        )
    )
    was_confirmed = prev is not None and prev.status == "confirmed"
    if confirm and species.model_version > 0 and (changed or not was_confirmed):
        repo.update_species(taxon_key, retrain_needed=True, status=SpeciesStatus.RETRAIN_PENDING)
    elif not confirm:
        repo.update_species(taxon_key, status=SpeciesStatus.AWAITING_NATIVE_RANGE_REVIEW)
    return nr


def recompute_severity(
    repo: Repository, taxon_key: int, cfg: SeverityConfig
) -> dict[str, Any] | None:
    """Re-weight the stored severity components for the current model (no retraining)."""
    species = repo.get_species(taxon_key)
    if species is None:
        raise KeyError(taxon_key)
    repo.update_species(taxon_key, severity_config=cfg.to_dict())
    if not species.model_version:
        return None
    mv = repo.get_model_version(taxon_key, species.model_version)
    if mv is None or not mv.severity:
        return None
    inputs = SeverityInputs(**mv.severity["inputs"])
    # spread window may change the rate only via retraining; weights/priors apply instantly
    severity = compute_severity(inputs, cfg)
    repo.update_model_severity(taxon_key, species.model_version, severity)
    return severity
