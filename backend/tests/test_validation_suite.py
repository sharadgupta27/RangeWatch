"""Validation suite on the synthetic world (offline)."""

from __future__ import annotations

import json

import pytest

from src.orchestration import validation_suite as vs
from src.orchestration.validation_suite import ReferenceSpecies, Region
from tests.conftest import TAXON_KEY, FakeGbif, box_records
from tests.test_pipeline import make_pipeline


@pytest.fixture
def reference(monkeypatch, settings):
    sp = ReferenceSpecies(
        "Testus invasivus",
        TAXON_KEY,
        ((-2.0, 30.0), (45.0, 30.0), (45.0, 60.0), (-2.0, 60.0)),
        "synthetic",
        (
            Region("Invaded America", (-100, 35, -80, 50), "suitable", "synthetic"),
            Region("Equatorial control", (0, -10, 40, 5), "unsuitable", "synthetic"),
        ),
    )
    monkeypatch.setattr(vs, "REFERENCE_SPECIES", (sp,))
    yield sp
    # The data root is shared across the session: never leak a curated polygon to other tests.
    (settings.native_range_root / f"{TAXON_KEY}.geojson").unlink(missing_ok=True)


def test_suite_trains_evaluates_and_writes_report(settings, reference):
    recs = box_records(300, (0, 40), (35, 55), 1, seed=1) + box_records(
        60, (-100, -80), (35, 50), 10_000, label="introduced", seed=2
    )
    p = make_pipeline(settings, FakeGbif(initial=recs))
    report = vs.run_suite(p)

    sp = report["species"][0]
    assert sp["status"] == "evaluated" and sp["model_version"] == 1
    # Curated validation polygon was used (no manual review step needed for the suite).
    assert p.repo.get_native_range(TAXON_KEY).source.startswith("curated:validation-suite")
    regions = {r["name"]: r for r in sp["regions"]}
    assert regions["Invaded America"]["passed"]
    assert regions["Equatorial control"]["passed"]
    t = sp["transferability"]
    assert 0 <= t["auc"] <= 1 and t["n_introduced_test"] > 0
    assert report["summary"]["n_species"] == 1
    latest = p.artifacts.local_path("validation/latest.json")
    assert json.loads(latest.read_text())["summary"] == report["summary"]


def test_suite_never_overrides_a_user_edited_range(settings, reference):
    from src.orchestration.registry_ops import update_native_range
    from tests.conftest import native_box_geojson

    recs = box_records(300, (0, 40), (35, 55), 1, seed=1)
    p = make_pipeline(settings, FakeGbif(initial=recs))
    settings.native_range_root.mkdir(parents=True, exist_ok=True)
    p.run(TAXON_KEY)  # heuristic draft (no curated file written yet)
    update_native_range(p.repo, TAXON_KEY, native_box_geojson(), confirm=False)
    vs.install_reference_polygons(p, [reference])
    assert p.repo.get_native_range(TAXON_KEY).source == "user_edit"


def test_region_evaluation_handles_boxes_without_land(settings, reference):
    recs = box_records(300, (0, 40), (35, 55), 1, seed=1)
    p = make_pipeline(settings, FakeGbif(initial=recs))
    vs.run_suite(p)
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    out = vs.evaluate_regions(
        p.artifacts.local_path(mv.artifacts["suitability"]),
        0.5,
        (Region("Mid-Pacific", (-170, -10, -150, 10), "unsuitable", "ocean"),),
    )
    assert out[0]["passed"] is False  # no land → cannot be evaluated, never a silent pass
