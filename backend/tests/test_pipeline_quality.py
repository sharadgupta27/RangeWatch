"""Pipeline integration of the Priority-1 improvements: data-quality exclusions, explicit GBIF
absences, the sampling-effort layer, effort-weighted background and independent evaluation on
reference datasets (uploads, absences) across model versions."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from src.connectors.gbif_client import GBIF_ABSENCES_DATASET_ID
from src.connectors.upload_client import parse_upload
from src.domain import OccurrenceRecord, ReferenceDataset, ReferenceRecord
from src.orchestration.pipeline import Decision
from src.persistence.repository import InMemoryRepository
from tests.conftest import TAXON_KEY, FakeGbif, box_records
from tests.test_pipeline import confirm_native, initial_records, make_pipeline

GBIF_HQ = (12.5788, 55.6761)


def bad_records() -> list[OccurrenceRecord]:
    """Records the error tests must exclude: geocoded to GBIF's address, lon == lat, and a
    record GBIF flagged as outside its stated country."""
    hq = [
        OccurrenceRecord(
            TAXON_KEY,
            "GBIF",
            90_000 + i,
            GBIF_HQ[0],
            GBIF_HQ[1],
            event_date=date(2015, 1, 1),
            coordinate_uncertainty_m=10.0,
        )
        for i in range(4)
    ]
    equal = [
        OccurrenceRecord(
            TAXON_KEY,
            "GBIF",
            91_000 + i,
            30.0,
            30.0,
            event_date=date(2015, 1, 1),
            coordinate_uncertainty_m=10.0,
        )
        for i in range(3)
    ]
    mismatch = [
        replace(r, issues=("COUNTRY_COORDINATE_MISMATCH",))
        for r in box_records(2, (5, 10), (40, 45), start_id=92_000, seed=9)
    ]
    return hq + equal + mismatch


def absences(n: int = 25) -> list[ReferenceRecord]:
    # Surveys reporting the species absent, well outside its synthetic niche (tropics).
    return [
        ReferenceRecord(
            TAXON_KEY,
            GBIF_ABSENCES_DATASET_ID,
            70_000 + i,
            100.0 + i * 0.5,
            5.0,
            occurrence_status="absent",
            event_date=date(2020, 1, 1),
        )
        for i in range(n)
    ]


@pytest.fixture
def trained(settings):
    gbif = FakeGbif(initial=initial_records() + bad_records(), absences=absences())
    p = make_pipeline(settings, gbif)
    p.run(TAXON_KEY)
    confirm_native(p)
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.TRAINED
    return p, gbif


def test_quality_tests_exclude_bad_records_and_are_logged(trained):
    p, _ = trained
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    dq = mv.metrics["reproducibility"]["data_quality"]
    by_id = {t["id"]: t for t in dq["tests"]}
    assert by_id["gbif_headquarters"]["n_flagged"] == 4
    assert by_id["equal_coordinates"]["n_flagged"] == 3
    assert by_id["gbif_geospatial_issue"]["n_flagged"] == 2
    assert dq["n_excluded"] >= 9 and dq["version"] == "quality_v1"
    # no gazetteer installed in tests: reference tests are reported as skipped
    assert by_id["capital"]["status"] == "skipped"
    # the snapshot lists every assessed record; excluded ones never reach training
    flags = pd.read_parquet(p.artifacts.local_path(mv.artifacts["quality_flags"]))
    assert dq["snapshot"]["sha256"] and len(flags) == dq["n_records"]
    excluded_ids = set(flags.loc[flags["quality_excluded"], "record_id"].astype(int))
    assert {90_000, 91_000, 92_000} <= excluded_ids
    training = pd.read_parquet(p.artifacts.local_path(mv.artifacts["training_data"]))
    assert not set(training["record_id"].dropna().astype(int)) & excluded_ids
    assert mv.metrics["reproducibility"]["random_seeds"]["quality_outlier_fit"] == 42


def test_quality_exclusions_are_configurable(settings):
    lenient = settings.model_copy(update={"quality_exclude": []})
    p = make_pipeline(lenient, FakeGbif(initial=initial_records() + bad_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    dq = p.repo.get_model_version(TAXON_KEY, 1).metrics["reproducibility"]["data_quality"]
    assert dq["n_excluded"] == 0 and dq["config"]["exclude"] == []


def test_absences_become_an_evaluation_dataset_not_occurrences(trained):
    p, gbif = trained
    ds = p.repo.get_reference_dataset(TAXON_KEY, GBIF_ABSENCES_DATASET_ID)
    assert ds is not None and ds.kind == "gbif_absences"
    assert (ds.n_absent, ds.n_present) == (25, 0) and ds.last_fetch_ts is not None
    assert p.repo.occurrence_counts(TAXON_KEY)["total"] == len(initial_records()) + 9
    ev = p.repo.get_model_version(TAXON_KEY, 1).artifacts["independent_evaluation"]
    m = ev[GBIF_ABSENCES_DATASET_ID]["metrics"]
    assert m["n_absence"] == 25 and m["specificity"] == 1.0  # tropics: far outside the niche
    # incremental runs ask only for absences since the dataset's own watermark
    p.run(TAXON_KEY)
    assert gbif.absence_calls[0] is None and gbif.absence_calls[-1] == ds.last_fetch_ts


def test_versions_without_an_extrapolation_raster_are_still_scored(trained):
    p, _ = trained
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    legacy = {k: v for k, v in mv.artifacts.items() if k != "extrapolation"}
    p.repo.update_model_artifacts(TAXON_KEY, 1, legacy)
    assert p.evaluate_references(TAXON_KEY)["model_versions"] == [1]
    m = p.repo.get_model_version(TAXON_KEY, 1).artifacts["independent_evaluation"]
    assert m[GBIF_ABSENCES_DATASET_ID]["metrics"]["n_absence"] == 25


def test_absence_fetch_failure_never_blocks_the_pipeline(settings):
    gbif = FakeGbif(initial=initial_records(), absence_error=RuntimeError("GBIF down"))
    p = make_pipeline(settings, gbif)
    assert p.run(TAXON_KEY).decision == Decision.INGESTED_AWAITING_REVIEW
    assert p.repo.get_reference_dataset(TAXON_KEY, GBIF_ABSENCES_DATASET_ID) is None


def test_effort_layer_and_unsampled_shares(trained):
    p, gbif = trained
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    assert p.artifacts.local_path(mv.artifacts["effort"]).exists()
    eff = mv.metrics["sampling_effort"]
    assert eff["status"] == "available" and eff["target_group"]["name"] == "Lamiales"
    # the fake has effort only south of 50°N: part of the land is unsampled
    assert 0 < eff["land_unsampled_fraction"] < 1
    assert (-180.0, -90.0, 180.0, 90.0) in [bbox for _, bbox in gbif.effort_calls]


def test_effort_layer_failure_is_recorded_not_fatal(settings):
    class FlakyGbif(FakeGbif):
        def fetch_effort_density(self, taxon_key, bbox, max_pixel_deg):
            if bbox == (-180.0, -90.0, 180.0, 90.0):
                raise TimeoutError("map API stalled")
            return super().fetch_effort_density(taxon_key, bbox, max_pixel_deg)

    p = make_pipeline(settings, FlakyGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    assert p.run(TAXON_KEY).decision == Decision.TRAINED
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    assert "effort" not in mv.artifacts
    assert mv.metrics["sampling_effort"]["status"] == "unavailable"
    assert "map API stalled" in mv.metrics["sampling_effort"]["reason"]


def test_record_count_weighting_is_used_and_logged(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.training_config = replace(p.training_config, target_group_weighting="record_count")
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    bg = mv.metrics["reproducibility"]["background"]
    assert bg["method"] == "target_group_order_record_count"
    assert bg["target_group"]["weighting"] == "record_count"
    assert bg["target_group"]["n_background_cells"] >= 500
    assert mv.metrics["reproducibility"]["training_config"]["target_group_weighting"] == (
        "record_count"
    )


def upload(p, csv: str, name: str = "Agency survey", role: str = "evaluation") -> ReferenceDataset:
    sp = p.repo.get_species(TAXON_KEY)
    parsed = parse_upload(csv.encode(), "survey.csv", TAXON_KEY, "upload-test", [sp.canonical_name])
    ds = p.repo.save_reference_dataset(
        ReferenceDataset(
            TAXON_KEY,
            "upload-test",
            "upload",
            name,
            role=role,
            parse_report=parsed.report,
            n_present=parsed.n_present,
            n_absent=parsed.n_absent,
        )
    )
    p.repo.append_reference_records(parsed.records)
    return ds


def test_uploads_are_checked_for_independence_and_every_version_is_scored(trained):
    p, gbif = trained
    occ = p.repo.load_occurrences(TAXON_KEY)
    dup = occ[occ["source"] == "GBIF"].head(5)
    rows = ["longitude,latitude,date,species"]
    rows += [
        f"{r.longitude},{r.latitude},{r.event_date},Testus invasivus" for r in dup.itertuples()
    ]
    rows += [f"{20 + i * 0.3},{45 + (i % 5)},2021-05-01,Testus invasivus" for i in range(30)]
    rows += ["30.5,40.5,2021-05-01,Other plant"]
    upload(p, "\n".join(rows))

    # second model version (manual retrain), then evaluate all versions on the upload
    p.run(TAXON_KEY, force_retrain=True)
    result = p.evaluate_references(TAXON_KEY)
    assert sorted(result["model_versions"]) == [1, 2]

    ds = p.repo.get_reference_dataset(TAXON_KEY, "upload-test")
    assert ds.overlap["n_overlap"] == 5 and ds.overlap["n_checked"] == 35
    assert ds.parse_report["skipped"] == {"other species": 1}
    for version in (1, 2):
        entry = p.repo.get_model_version(TAXON_KEY, version).artifacts["independent_evaluation"]
        up = entry["upload-test"]
        assert up["n_excluded_overlap"] == 5 and up["metrics"]["n_presence"] == 30
        assert up["metrics_all_records"]["n_presence"] == 35
        assert up["metrics"]["sensitivity"] is not None
        assert GBIF_ABSENCES_DATASET_ID in entry


def test_issues_are_filled_once_for_records_ingested_without_them():
    from src.domain import SpeciesRecord

    repo = InMemoryRepository()
    repo.create_species(SpeciesRecord(TAXON_KEY, "Testus invasivus"))
    rec = box_records(1, (0, 1), (40, 41), start_id=1)[0]
    repo.append_occurrences([rec])
    df = repo._occ[TAXON_KEY]
    df.at[df.index[0], "issues"] = None  # simulate a row stored before issues were kept
    assert repo.append_occurrences([replace(rec, issues=("TAXON_MATCH_FUZZY",))]) == 0
    assert list(repo.load_occurrences(TAXON_KEY)["issues"][0]) == ["TAXON_MATCH_FUZZY"]
    # once known, issues are never rewritten (append-only)
    repo.append_occurrences([replace(rec, issues=("ZERO_COORDINATE",))])
    assert list(repo.load_occurrences(TAXON_KEY)["issues"][0]) == ["TAXON_MATCH_FUZZY"]


def test_training_uploads_join_training_and_only_their_absences_evaluate(trained):
    p, _ = trained
    occ = p.repo.load_occurrences(TAXON_KEY)
    dup = occ[occ["source"] == "GBIF"].head(5)
    rows = ["longitude,latitude,date,presence"]
    rows += [f"{r.longitude},{r.latitude},{r.event_date},1" for r in dup.itertuples()]
    # presences inside the native box, absences far outside the niche (tropics)
    rows += [f"{1 + i * 1.1},{41 + (i % 13)},2021-05-01,1" for i in range(30)]
    rows += [f"{100 + i * 0.5},5.0,2021-05-01,0" for i in range(10)]
    upload(p, "\n".join(rows), name="Agency plots", role="training")

    out = p.run(TAXON_KEY, force_retrain=True)
    mv = p.repo.get_model_version(TAXON_KEY, out.model_version)
    repro = mv.metrics["reproducibility"]
    (used,) = repro["training_uploads"]
    assert used["dataset_id"] == "upload-test"
    assert (used["n_presences"], used["n_overlap_excluded"], used["n_added"]) == (35, 5, 30)
    assert used["n_absences_not_used"] == 10 and used["file_sha256"] is None

    source = "upload:upload-test"
    manifest = pd.read_parquet(p.artifacts.local_path(mv.artifacts["occurrence_manifest"]))
    assert (manifest["source"] == source).sum() == 30
    training = pd.read_parquet(p.artifacts.local_path(mv.artifacts["training_data"]))
    assert (training["source"] == source).any()  # some survive cell de-duplication/thinning

    ev = mv.artifacts["independent_evaluation"]["upload-test"]
    assert ev["role"] == "training" and ev["records_scored"] == "absences"
    assert ev["metrics"]["n_presence"] == 0 and ev["metrics"]["n_absence"] == 10
    assert ev["metrics_all_records"] is None


def test_details_are_filled_once_for_records_ingested_without_them():
    from src.domain import SpeciesRecord

    repo = InMemoryRepository()
    repo.create_species(SpeciesRecord(TAXON_KEY, "Testus invasivus"))
    rec = box_records(1, (0, 1), (40, 41), start_id=1)[0]
    repo.append_occurrences([rec])  # stored before details were kept (None)
    assert repo.load_occurrences(TAXON_KEY)["details"][0] is None
    repo.append_occurrences([replace(rec, details={"locality": "Botanic garden"})])
    assert repo.load_occurrences(TAXON_KEY)["details"][0] == {"locality": "Botanic garden"}
    repo.append_occurrences([replace(rec, details={"locality": "elsewhere"})])
    assert repo.load_occurrences(TAXON_KEY)["details"][0] == {"locality": "Botanic garden"}
