"""Data-quality tests (features/quality.py), reference gazetteers and the independence check."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.features.quality import (
    DEFAULT_EXCLUDE,
    TESTS,
    QualityConfig,
    assess_quality,
    implied_uncertainty_m,
    overlap_with_occurrences,
)
from src.features.quality_refs import QualityReferences, write_references

CELL_10M = 1 / 6  # 10′ grid


def frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "source": "GBIF",
        "event_date": date(2015, 6, 1),
        "coordinate_uncertainty_m": 100.0,
        "basis_of_record": "HUMAN_OBSERVATION",
        "issues": [],
        "bio1": 15.0,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["record_id"] = np.arange(1, len(df) + 1)
    return df


def clean_points(n: int, seed: int = 0) -> list[dict]:
    """A dense, well-behaved cluster (no flags) in southern France."""
    rng = np.random.default_rng(seed)
    return [
        {"longitude": float(x), "latitude": float(y), "bio1": float(t)}
        for x, y, t in zip(
            rng.uniform(3.0, 6.0, n),
            rng.uniform(43.0, 45.0, n),
            rng.normal(14, 1, n),
            strict=True,
        )
    ]


@pytest.fixture
def refs(tmp_path) -> QualityReferences:
    write_references(
        tmp_path,
        "refs_test",
        {
            "capitals": pd.DataFrame(
                [{"name": "Paris", "country": "France", "longitude": 2.3522, "latitude": 48.8566}]
            ),
            "country_centroids": pd.DataFrame(
                [
                    {
                        "name": "Spain",
                        "iso_a3": "ESP",
                        "kind": "centroid",
                        "longitude": -3.65,
                        "latitude": 40.24,
                    }
                ]
            ),
            "province_centroids": pd.DataFrame(
                [
                    {
                        "name": "Corsica",
                        "iso_a3": "FRA",
                        "kind": "centroid",
                        "longitude": 9.10,
                        "latitude": 42.15,
                    }
                ]
            ),
            "institutions": pd.DataFrame(
                [
                    {
                        "key": "k",
                        "name": "Jardin des Plantes",
                        "types": "BOTANICAL_GARDEN",
                        "country": "FR",
                        "longitude": 3.8730,
                        "latitude": 43.6150,
                    }
                ]
            ),
        },
        sources={"capitals": "test"},
    )
    loaded = QualityReferences.load(tmp_path, "refs_test")
    assert loaded is not None
    return loaded


def flags_of(result, i: int) -> list[str]:
    return result.flags[i]


def test_error_tests_flag_and_exclude_bad_coordinates(refs):
    bad = [
        {"longitude": 25.0, "latitude": 25.0},  # equal coordinates
        {"longitude": 12.5790, "latitude": 55.6762},  # GBIF headquarters
        {"longitude": 2.355, "latitude": 48.858},  # 0.3 km from the capital point
        {"longitude": -3.6503, "latitude": 40.2402},  # country centroid (1 km buffer)
        {"longitude": 9.1001, "latitude": 42.1501},  # province centroid
        {"longitude": 3.87305, "latitude": 43.61502},  # botanical garden (100 m buffer)
        {"longitude": 4.123456, "latitude": 44.123456, "issues": ["COUNTRY_COORDINATE_MISMATCH"]},
    ]
    df = frame(bad + clean_points(60))
    r = assess_quality(df, ["bio1"], CELL_10M, refs)
    expected = [
        "equal_coordinates",
        "gbif_headquarters",
        "capital",
        "country_centroid",
        "province_centroid",
        "institution",
        "gbif_geospatial_issue",
    ]
    for i, test_id in enumerate(expected):
        assert test_id in flags_of(r, i), (i, flags_of(r, i))
        assert r.excluded[i] and r.tier[i] == "excluded"
    assert not r.excluded[len(bad) :].any()
    s = r.summary
    assert s["n_excluded"] == len(bad)
    by_id = {t["id"]: t for t in s["tests"]}
    assert all(by_id[t]["status"] == "run" for t in expected)
    assert s["reference"]["version"] == "refs_test"
    assert s["reference"]["tables"]["capitals"]["sha256"]


def test_records_near_but_not_at_a_capital_are_only_flagged_for_review(refs):
    # 3.5 km from the Paris point: inside CoordinateCleaner's 10 km buffer, but a precise
    # urban record, not a geocode to the capital.
    df = frame([{"longitude": 2.40, "latitude": 48.86}] + clean_points(30))
    r = assess_quality(df, ["bio1"], CELL_10M, refs)
    assert "near_capital" in flags_of(r, 0) and "capital" not in flags_of(r, 0)
    assert not r.excluded[0] and r.tier[0] == "C"


def test_reference_tests_are_skipped_not_passed_without_gazetteers():
    r = assess_quality(frame(clean_points(40)), ["bio1"], CELL_10M, refs=None)
    by_id = {t["id"]: t for t in r.summary["tests"]}
    for test_id in ("capital", "country_centroid", "province_centroid", "institution"):
        assert by_id[test_id]["status"] == "skipped"
        assert "quality_refs build" in by_id[test_id]["reason"]
    assert by_id["equal_coordinates"]["status"] == "run"
    assert r.summary["reference"] is None


def test_warning_tests_flag_but_keep_records():
    rows = [
        {"longitude": 4.5, "latitude": 44.1, "bio1": 40.0},  # environmental outlier
        {"longitude": 120.0, "latitude": -30.0},  # spatial isolate (> 1000 km from all)
        {"longitude": 4.2, "latitude": 44.2, "event_date": date(1890, 5, 1)},
        {"longitude": 4.3, "latitude": 44.3, "issues": ["TAXON_MATCH_FUZZY"]},
        {"longitude": 4.31, "latitude": 44.31, "issues": ["PRESUMED_NEGATED_LONGITUDE"]},
    ] + clean_points(80)
    r = assess_quality(frame(rows), ["bio1"], CELL_10M)
    assert "environmental_outlier" in flags_of(r, 0)
    assert "spatial_isolate" in flags_of(r, 1)
    assert "before_climate_baseline" in flags_of(r, 2)
    assert "gbif_taxon_match" in flags_of(r, 3)
    assert "gbif_coordinate_corrected" in flags_of(r, 4)
    assert not r.excluded[:5].any()
    assert (r.tier[:5] == "C").all()
    assert r.summary["n_flagged_kept"] >= 5
    assert r.summary["environmental_outlier_cutoff"]["chi2_df"] == 1


def test_low_precision_is_relative_to_the_model_grid():
    assert implied_uncertainty_m(34.9, 32.5) == pytest.approx(0.05 * 111_320)
    rows = [
        {"longitude": 35.0, "latitude": 32.0, "coordinate_uncertainty_m": None},  # ±55 km
        {"longitude": 34.9, "latitude": 32.5, "coordinate_uncertainty_m": None},  # ±5.6 km
        {"longitude": 34.91, "latitude": 32.51, "coordinate_uncertainty_m": 8_000.0},
    ] + clean_points(40)
    coarse = assess_quality(frame(rows), ["bio1"], CELL_10M)  # 10′ ≈ 18.6 km cells
    assert [("low_precision" in f) for f in coarse.flags[:3]] == [True, False, False]
    fine = assess_quality(frame(rows), ["bio1"], 2.5 / 60)  # 2.5′ ≈ 4.6 km cells
    assert [("low_precision" in f) for f in fine.flags[:3]] == [True, True, True]


def test_tiers_and_configurable_exclusions():
    rows = [
        {
            "longitude": 4.1,
            "latitude": 44.1,
            "basis_of_record": "PRESERVED_SPECIMEN",
            "coordinate_uncertainty_m": 30.0,
        },
        {"longitude": 4.2, "latitude": 44.2},  # human observation, stated uncertainty
        {"longitude": 4.3, "latitude": 44.3, "coordinate_uncertainty_m": None},
        {"longitude": 25.0, "latitude": 25.0},
    ] + clean_points(40)
    r = assess_quality(frame(rows), ["bio1"], CELL_10M)
    assert list(r.tier[:4]) == ["A", "B", "C", "excluded"]
    lenient = assess_quality(frame(rows), ["bio1"], CELL_10M, cfg=QualityConfig(exclude=()))
    assert not lenient.excluded.any() and lenient.tier[3] == "C"
    with pytest.raises(ValueError, match="Unknown quality tests"):
        QualityConfig(exclude=("no_such_test",))
    assert set(DEFAULT_EXCLUDE) == {t.id for t in TESTS.values() if t.severity == "error"}


def test_issue_tests_skip_when_issues_were_never_recorded():
    df = frame(clean_points(30))
    df["issues"] = None  # rows ingested before issue flags were kept
    by_id = {t["id"]: t for t in assess_quality(df, ["bio1"], CELL_10M).summary["tests"]}
    assert by_id["gbif_geospatial_issue"]["status"] == "skipped"
    assert "re-sync" in by_id["gbif_geospatial_issue"]["reason"]


def test_snapshot_frame_lists_every_record_sorted():
    df = frame([{"longitude": 25.0, "latitude": 25.0}] + clean_points(20))
    r = assess_quality(df, ["bio1"], CELL_10M)
    out = r.frame(df)
    assert len(out) == len(df)
    assert list(out.columns) == [
        "source",
        "record_id",
        "longitude",
        "latitude",
        "quality_flags",
        "quality_tier",
        "quality_excluded",
    ]
    first = out[out["record_id"] == 1].iloc[0]
    # (25, 25) is also > 1000 km from the cluster, so it is isolated as well.
    assert first["quality_flags"] == "equal_coordinates;spatial_isolate"
    assert bool(first["quality_excluded"])


def test_reference_versions_are_immutable(tmp_path, refs):
    with pytest.raises(FileExistsError):
        write_references(tmp_path, "refs_test", {}, {})
    assert QualityReferences.load(tmp_path, "missing") is None


def test_overlap_requires_distance_and_matching_dates():
    occ_lon, occ_lat = [34.900, 35.500], [32.500, 31.000]
    occ_dates = [date(2020, 3, 1), None]
    ref = overlap_with_occurrences(
        [34.9003, 34.9003, 34.95, 35.5004],
        [32.5002, 32.5002, 32.55, 31.0002],
        [date(2020, 3, 2), date(2021, 1, 1), date(2020, 3, 1), date(2019, 1, 1)],
        occ_lon,
        occ_lat,
        occ_dates,
        distance_m=100,
        max_days=1,
    )
    # same place + next day; same place, other year; 7 km away; near an undated record
    assert ref.tolist() == [True, False, False, True]
    assert overlap_with_occurrences([], [], [], occ_lon, occ_lat, occ_dates).size == 0


def test_cultivation_wording_is_flagged_for_review_not_excluded():
    pts = clean_points(4)
    texts = [
        {"locality": "Jardín Botánico de Valencia"},
        {"habitat": "planted along the road"},
        {"remarks": "cult. in pots"},
        {"locality": "Parc national des Cévennes"},  # a national park is a wild site
    ]
    occ = frame([{**p, "details": d} for p, d in zip(pts, texts, strict=True)])
    res = assess_quality(occ, ["bio1"], CELL_10M)
    assert [("cultivation_text" in f) for f in res.flags] == [True, True, True, False]
    assert not res.excluded.any() and list(res.tier[:3]) == ["C", "C", "C"]


def test_duplicate_specimens_keep_one_record_per_collector_number_date():
    pts = clean_points(6)
    spec = {"basis_of_record": "PRESERVED_SPECIMEN"}
    dup = {"recorded_by": "Smith, J.", "record_number": "1234"}
    rows = [
        {**pts[0], **spec, "details": dup},
        {**pts[1], **spec, "details": {"recorded_by": "J. Smith", "record_number": "1234"}},
        {**pts[2], **spec, "details": {"recorded_by": "smith j", "record_number": "1234"}},
        {**pts[3], **spec, "details": {"recorded_by": "Smith, J.", "record_number": "s.n."}},
        {**pts[4], **spec, "details": {"recorded_by": "Smith, J.", "record_number": "s.n."}},
        {**pts[5], "details": dup},  # a human observation, not a specimen
    ]
    res = assess_quality(frame(rows), ["bio1"], CELL_10M)
    dupes = ["duplicate_specimen" in f for f in res.flags]
    # "Smith, J." / "smith j" normalise alike; "J. Smith" differs; s.n. never groups
    assert dupes == [False, False, True, False, False, False]
    assert res.excluded[2] and "duplicate_specimen" in DEFAULT_EXCLUDE


def test_weak_inaturalist_identifications():
    pts = clean_points(4)
    inat = {"source": "iNaturalist"}
    rows = [
        {**pts[0], **inat, "details": {"id_agreements": 4, "id_disagreements": 1}},
        {**pts[1], **inat, "details": {"id_agreements": 2, "observer_id_from_vision": True}},
        {**pts[2], **inat, "details": {"id_agreements": 3, "observer_id_from_vision": True}},
        {**pts[3], **inat, "details": {"id_agreements": 2, "observer_id_from_vision": False}},
    ]
    res = assess_quality(frame(rows), ["bio1"], CELL_10M)
    assert ["weak_identification" in f for f in res.flags] == [True, True, False, False]
    by_id = {t["id"]: t for t in res.summary["tests"]}
    assert by_id["weak_identification"]["severity"] == "warning"


def test_detail_tests_skip_when_details_were_never_recorded(refs):
    res = assess_quality(frame(clean_points(5)), ["bio1"], CELL_10M, refs=refs)
    by_id = {t["id"]: t for t in res.summary["tests"]}
    for test_id in ("cultivation_text", "duplicate_specimen"):
        assert by_id[test_id]["status"] == "skipped" and "re-sync" in by_id[test_id]["reason"]
    assert by_id["weak_identification"]["reason"] == "no iNaturalist observations"
    assert res.summary["n_details_unknown"] == 5


def test_near_a_botanical_garden_is_a_warning(refs):
    garden = {"longitude": 3.8790, "latitude": 43.6150}  # ~0.5 km from the garden point
    at = {"longitude": 3.8730, "latitude": 43.6150}  # inside the institution buffer
    res = assess_quality(
        frame([garden, at, *clean_points(3, seed=5)]), ["bio1"], CELL_10M, refs=refs
    )
    assert "near_botanical_garden" in res.flags[0] and not res.excluded[0]
    # inside the 100 m institution buffer it is an error, not also a garden warning
    assert res.flags[1] == ["institution"] and res.excluded[1]
