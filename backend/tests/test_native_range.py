"""Native-range tests: drafts only, explicit labels win, geometry validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import shape

from src.modeling.native_range import (
    effective_range_labels,
    propose_native_range,
    validate_geojson_polygon,
)


def _occ(n_native: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    eur = pd.DataFrame(
        {
            "longitude": rng.uniform(5, 30, 200),
            "latitude": rng.uniform(40, 55, 200),
            "range_label": "unknown",
        }
    )
    eur.loc[: n_native - 1, "range_label"] = "native"
    ame = pd.DataFrame(
        {
            "longitude": rng.uniform(-100, -85, 40),
            "latitude": rng.uniform(35, 45, 40),
            "range_label": "unknown",
        }
    )
    return pd.concat([eur, ame], ignore_index=True)


def test_proposal_uses_native_flags_when_available():
    p = propose_native_range(_occ(n_native=20))
    assert p.source == "heuristic:gbif_establishment_means_native"
    assert p.geometry["type"] == "MultiPolygon"
    assert "review" in p.note.lower()


def test_proposal_falls_back_to_largest_cluster():
    p = propose_native_range(_occ(n_native=0))
    assert p.source == "heuristic:largest_occurrence_cluster"
    g = shape(p.geometry)
    assert g.contains(shape({"type": "Point", "coordinates": [15, 47]}))
    assert not g.contains(shape({"type": "Point", "coordinates": [-92, 40]}))


def test_effective_labels_explicit_wins_then_polygon():
    occ = pd.DataFrame(
        {
            "longitude": [10, 10, -90, -90],
            "latitude": [45, 45, 40, 40],
            "range_label": ["unknown", "introduced", "unknown", "native"],
        }
    )
    poly = {"type": "Polygon", "coordinates": [[[0, 30], [40, 30], [40, 60], [0, 60], [0, 30]]]}
    labels = effective_range_labels(occ, poly)
    assert list(labels) == ["native", "introduced", "introduced", "native"]


def test_update_native_range_provenance_and_retrain_flag():
    from src.domain import NativeRange, SpeciesRecord, SpeciesStatus
    from src.orchestration.registry_ops import update_native_range
    from src.persistence.repository import InMemoryRepository

    repo = InMemoryRepository()
    repo.create_species(SpeciesRecord(taxon_key=1, scientific_name="X y", model_version=2))
    draft = validate_geojson_polygon(
        {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}
    )
    repo.save_native_range(
        NativeRange(1, draft, "draft", "heuristic:largest_occurrence_cluster", note="heuristic")
    )

    # Confirming the unedited draft keeps the heuristic's provenance.
    nr = update_native_range(repo, 1, draft, confirm=True)
    assert nr.source == "heuristic:largest_occurrence_cluster" and nr.note == "heuristic"
    assert repo.get_species(1).status == SpeciesStatus.RETRAIN_PENDING

    # Re-confirming the identical polygon does not flag another retrain.
    repo.update_species(1, retrain_needed=False, status=SpeciesStatus.UP_TO_DATE)
    update_native_range(repo, 1, draft, confirm=True)
    assert repo.get_species(1).retrain_needed is False

    # An actual edit replaces the stale heuristic note and flags a retrain.
    edited = {"type": "Polygon", "coordinates": [[[0, 0], [12, 0], [12, 12], [0, 12], [0, 0]]]}
    nr = update_native_range(repo, 1, edited, confirm=True)
    assert nr.source == "user_edit"
    assert nr.note == "User-edited from heuristic:largest_occurrence_cluster."
    assert repo.get_species(1).retrain_needed is True


def test_validate_geometry():
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    assert validate_geojson_polygon(poly)["type"] == "MultiPolygon"
    with pytest.raises(ValueError):
        validate_geojson_polygon(
            {"type": "Polygon", "coordinates": [[[0, 0], [500, 0], [500, 1], [0, 0]]]}
        )
    # self-intersecting bow-tie is repaired rather than rejected
    bowtie = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    assert shape(validate_geojson_polygon(bowtie)).is_valid
