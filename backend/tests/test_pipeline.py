"""End-to-end tests of the incremental fetch / conditional retrain contract (CLAUDE.md).

Runs the real pipeline (elapid training, projection, MESS, severity) against the synthetic
bioclim stack with offline GBIF/iNaturalist fakes and the in-memory repository.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.domain import SpeciesStatus
from src.modeling.severity_index import SeverityConfig
from src.orchestration.pipeline import Decision, SpeciesPipeline
from src.orchestration.registry_ops import recompute_severity, update_native_range
from src.persistence.artifact_store import ArtifactStore
from src.persistence.model_registry import NullModelRegistry
from src.persistence.repository import InMemoryRepository
from tests.conftest import (
    FAST_TRAINING,
    TAXON_KEY,
    FakeGbif,
    FakeInat,
    add_synthetic_scenario,
    box_records,
    build_stack,
    native_box_geojson,
)

NATIVE_BOX = ((0.0, 40.0), (35.0, 55.0))
INVADED_BOX = ((-100.0, -80.0), (35.0, 50.0))


def initial_records(with_introduced: bool = True):
    recs = box_records(300, *NATIVE_BOX, start_id=1, seed=1)
    if with_introduced:
        recs += box_records(
            60, *INVADED_BOX, start_id=10_000, label="introduced", seed=2, year=2016
        )
    return recs


def make_pipeline(settings, gbif, inat=None, repo=None) -> SpeciesPipeline:
    return SpeciesPipeline(
        repo=repo or InMemoryRepository(),
        gbif=gbif,
        inat=inat,
        artifacts=ArtifactStore(settings.artifact_root, "/data/artifacts"),
        registry=NullModelRegistry(),
        settings=settings,
        bulletin=None,
        training_config=FAST_TRAINING,
    )


def confirm_native(pipeline: SpeciesPipeline) -> None:
    update_native_range(pipeline.repo, TAXON_KEY, native_box_geojson(), confirm=True)


def test_full_lifecycle_follows_retrain_contract(settings):
    gbif = FakeGbif(initial=initial_records())
    p = make_pipeline(settings, gbif, FakeInat())

    # 1. New species → full ingest, draft native range, NO automatic training.
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.INGESTED_AWAITING_REVIEW
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.status == SpeciesStatus.AWAITING_NATIVE_RANGE_REVIEW
    assert sp.n_occurrences_total == 360 and sp.model_version == 0
    assert sp.common_name == "Test weed" and sp.inat_taxon_id == 777
    nr = p.repo.get_native_range(TAXON_KEY)
    assert nr.status == "draft" and nr.source.startswith("heuristic:")

    # 2. Re-running without review must still not train.
    assert p.run(TAXON_KEY).decision == Decision.AWAITING_REVIEW
    assert p.repo.get_species(TAXON_KEY).model_version == 0

    # 3. User confirms polygon → training → Model B (introduced records exist).
    confirm_native(p)
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.TRAINED and out.model_version == 1
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.status == SpeciesStatus.UP_TO_DATE and sp.model_version == 1
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    assert mv.model_type == "B" and mv.metrics["confidence_label"] == "standard"
    for key in (
        "suitability",
        "mess",
        "zones",
        "extrapolation",
        "exdet",
        "mop",
        "shape",
        "aoa",
        "consensus",
        "model",
        "training_data",
        "occurrence_manifest",
    ):
        assert p.artifacts.local_path(mv.artifacts[key]).exists(), key
    ext = mv.metrics["projection"]["extrapolation"]
    assert [d["id"] for d in ext["diagnostics"]] == ["mess", "exdet", "mop", "shape", "aoa"]
    for d in ext["diagnostics"]:
        assert 0 <= d["land_fraction"] <= 1 and 0 <= d["candidate_flagged_fraction"] <= 1
        assert d["threshold"] is not None and d["flag_rule"] and d["reference"]
    mess_diag = ext["diagnostics"][0]
    # MESS verdict of the advanced summary agrees with the legacy MESS area
    assert mess_diag["land_area_km2"] == pytest.approx(
        mv.metrics["projection"]["extrapolation_area_km2"], rel=1e-6, abs=0.2
    )
    assert sum(ext["consensus_land_area_km2"]) == pytest.approx(
        mv.metrics["projection"]["land_area_km2"], rel=1e-6
    )
    assert ext["n_cv_folds"] >= 2 and set(ext["aoa_weights"]) == set(mv.metrics["predictors"])

    m = mv.metrics
    for k in ("auc_mean", "cbi_mean", "tss_mean"):
        assert m[k] is not None, k
    assert m["auc_mean"] > 0.6
    assert "spatial block CV" in m["cv_method"]
    assert "0.7" in m["transferability_caveat"]
    repro = m["reproducibility"]
    assert repro["bioclim_version_used"] == settings.bioclim_version
    assert {"elapid", "pygbif", "pyinaturalist", "rasterio"} <= set(repro["package_versions"])
    assert {"background_sampling", "cv_fold_assignment"} <= set(repro["random_seeds"])
    assert "GBIF.org" in repro["gbif_citation"] and "None" not in repro["gbif_citation"]
    assert sp.model_metrics["reproducibility"] == repro  # registry mirrors current version
    assert mv.severity["confidence"]["level"] in {"low", "moderate", "high"}
    assert 0 <= mv.severity["score"] <= 1
    assert len(p.registry.logged) == 1

    # Features filled once in the store (append-only occurrence rows otherwise untouched).
    occ = p.repo.load_occurrences(TAXON_KEY)
    assert occ["bio1"].notna().sum() > 300

    # 4. No new records → serve cached results.
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.CACHED
    assert p.repo.get_species(TAXON_KEY).model_version == 1

    # 5. Small delta (≤5%) inside the suitability envelope → minor update pending.
    gbif.deltas.append(box_records(5, (18.0, 22.0), (43.0, 47.0), start_id=50_000, seed=5))
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.MINOR_UPDATE and out.n_new_records == 5
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.status == SpeciesStatus.MINOR_UPDATE_PENDING and sp.n_pending_records == 5
    assert sp.model_version == 1

    # 6. Re-delivered (already stored) records are not "new".
    gbif.deltas.append(box_records(5, (18.0, 22.0), (43.0, 47.0), start_id=50_000, seed=5))
    assert p.run(TAXON_KEY).decision == Decision.CACHED

    # 7. Large delta (>5%) → retrain + regenerate.
    gbif.deltas.append(box_records(40, *NATIVE_BOX, start_id=60_000, seed=6))
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.RETRAINED and out.model_version == 2
    assert "5%" in p.repo.get_model_version(TAXON_KEY, 2).trigger
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.n_pending_records == 0 and sp.status == SpeciesStatus.UP_TO_DATE

    # 8. Tiny delta outside the suitability envelope → retrain despite < 5%.
    gbif.deltas.append(box_records(2, (-62.0, -58.0), (-12.0, -8.0), start_id=70_000, seed=7))
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.RETRAINED and out.model_version == 3
    assert "envelope" in p.repo.get_model_version(TAXON_KEY, 3).trigger

    # 9. Editing the confirmed native range flags a retrain honoured on the next run.
    edited = {
        "type": "Polygon",
        "coordinates": [[[-5, 30], [50, 30], [50, 62], [-5, 62], [-5, 30]]],
    }
    update_native_range(p.repo, TAXON_KEY, edited, confirm=True)
    assert p.repo.get_species(TAXON_KEY).status == SpeciesStatus.RETRAIN_PENDING
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.TRAINED and out.model_version == 4
    lineage = p.repo.list_model_versions(TAXON_KEY)
    assert [v.model_version for v in lineage] == [4, 3, 2, 1]


def test_model_a_when_no_introduced_records(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records(with_introduced=False)))
    p.run(TAXON_KEY)
    confirm_native(p)
    out = p.run(TAXON_KEY)
    mv = p.repo.get_model_version(TAXON_KEY, out.model_version)
    assert mv.model_type == "A"
    assert mv.metrics["confidence_label"] == "lower"
    assert mv.severity["confidence"]["level"] == "low"


def test_threshold_is_configurable_but_documented(settings):
    """A stricter delta threshold turns the 'minor update' into a retrain."""
    gbif = FakeGbif(initial=initial_records())
    p = make_pipeline(replace_settings(settings, retrain_delta_fraction=0.01), gbif)
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    gbif.deltas.append(box_records(5, (18.0, 22.0), (43.0, 47.0), start_id=50_000, seed=5))
    assert p.run(TAXON_KEY).decision == Decision.RETRAINED


def replace_settings(settings, **kw):
    return settings.model_copy(update=kw)


def test_inat_records_already_in_gbif_are_deduplicated(settings):
    gbif_recs = initial_records()
    gbif_recs[0] = replace(gbif_recs[0], inat_crossref_id=1001)
    inat_obs = box_records(3, *NATIVE_BOX, start_id=1000, source="iNaturalist", seed=9)
    p = make_pipeline(settings, FakeGbif(initial=gbif_recs), FakeInat(observations=inat_obs))
    p.run(TAXON_KEY)
    occ = p.repo.load_occurrences(TAXON_KEY)
    inat_ids = set(occ.loc[occ["source"] == "iNaturalist", "record_id"].astype(int))
    assert inat_ids == {1000, 1002}  # 1001 already arrived via GBIF's iNat dataset


class FailingInat(FakeInat):
    def get_observations(self, *args, **kwargs):
        raise RuntimeError("429 Client Error: normal_throttling")


def test_failed_inat_fetch_keeps_its_watermark(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()), FailingInat())
    p.run(TAXON_KEY)
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.last_gbif_fetch_ts is not None
    assert sp.last_inat_fetch_ts is None  # nothing fetched → nothing marked as fetched

    # iNaturalist recovers: the next check fetches its full history, then advances the mark.
    inat_obs = box_records(3, *NATIVE_BOX, start_id=5000, source="iNaturalist", seed=9)
    p.inat = FakeInat(observations=inat_obs)
    delta = p.fetch_incremental(sp)
    assert {r.record_id for r in delta if r.source == "iNaturalist"} == {5000, 5001, 5002}
    p.update_watermark(TAXON_KEY, p._last_fetch_started)
    assert p.repo.get_species(TAXON_KEY).last_inat_fetch_ts is not None


def test_concurrent_runs_of_one_species_are_serialized(settings):
    """Two jobs for the same species (e.g. confirm-and-train while a check is in flight) must
    not both train version N — the second waits and then finds nothing left to do."""
    import threading

    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    waits: list[str] = []
    p.progress = lambda stage, frac, msg: waits.append(msg) if "Waiting" in msg else None

    outcomes, errors = [], []

    def go():
        try:
            outcomes.append(p.run(TAXON_KEY))
        except Exception as exc:  # pragma: no cover - the failure mode under test
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)

    assert errors == []
    assert sorted(o.decision for o in outcomes) == sorted([Decision.TRAINED, Decision.CACHED])
    assert [m.model_version for m in p.repo.list_model_versions(TAXON_KEY)] == [1]
    assert waits  # the second run reported that it was queued behind the first


def test_each_version_stores_a_hashed_training_snapshot(settings):
    import pandas as pd

    from src.persistence.training_snapshot import verify

    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    p.run(TAXON_KEY, force_retrain=True)  # same data, same seeds
    v1, v2 = p.repo.list_model_versions(TAXON_KEY)

    for mv in (v1, v2):
        repro = mv.metrics["reproducibility"]
        td, man = repro["training_data"], repro["occurrence_manifest"]
        assert mv.artifacts["training_data"] == td["path"]
        td_path = p.artifacts.local_path(td["path"])
        assert verify(td_path, td["sha256"]) and verify(
            p.artifacts.local_path(man["path"]), man["sha256"]
        )
        df = pd.read_parquet(td_path)
        assert (df["role"] == "presence").sum() == mv.metrics["n_presence"]
        assert (df["role"] == "background").sum() == mv.metrics["n_background"]
        assert df.loc[df["role"] == "presence", "record_id"].notna().all()
        assert set(mv.metrics["predictors"]) <= set(df.columns)
        manifest = pd.read_parquet(p.artifacts.local_path(man["path"]))
        assert set(manifest["range_label_effective"]) <= {"native", "introduced", "unknown"}
        assert man["n_rows"] == sum(mv.metrics["n_records_by_effective_label"].values())

    # Versions live in their own directories and are bit-for-bit reproducible.
    assert v1.metrics["reproducibility"]["training_data"]["path"] != (
        v2.metrics["reproducibility"]["training_data"]["path"]
    )
    assert (
        v1.metrics["reproducibility"]["training_data"]["sha256"]
        == v2.metrics["reproducibility"]["training_data"]["sha256"]
    )


def test_append_only_occurrences(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    assert p.repo.append_occurrences(initial_records()) == 0


def test_curated_confirmed_polygon_trains_on_first_run(tmp_path, settings):
    data_root = tmp_path / "data"
    build_stack(data_root / "bioclim")
    (data_root / "native_range_polygons").mkdir()
    (data_root / "native_range_polygons" / f"{TAXON_KEY}.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": native_box_geojson(),
                        "properties": {"status": "confirmed", "source": "POWO 2025"},
                    }
                ],
            }
        )
    )
    p = make_pipeline(
        replace_settings(settings, data_root=data_root), FakeGbif(initial=initial_records())
    )
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.TRAINED
    assert p.repo.get_native_range(TAXON_KEY).source == "curated:POWO 2025"


def test_severity_reweighting_without_retraining(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    before = p.repo.get_model_version(TAXON_KEY, 1).severity
    cfg = SeverityConfig(
        weights={
            "suitability": 0,
            "climate_analogy": 0,
            "spread_rate": 0,
            "ecological_impact_prior": 1,
        },
        impact_prior=0.9,
        impact_prior_source="GISD",
    )
    after = recompute_severity(p.repo, TAXON_KEY, cfg)
    assert after["score"] == pytest.approx(0.9)
    assert after["components"] == {**before["components"], "ecological_impact_prior": 0.9}
    assert p.repo.get_species(TAXON_KEY).model_version == 1
    assert p.repo.get_species(TAXON_KEY).severity_config["impact_prior_source"] == "GISD"


def test_failed_first_ingest_is_recorded_and_recoverable(settings):
    gbif = FakeGbif(initial=initial_records())
    p = make_pipeline(settings, gbif)
    calls = {"n": 0}
    real = gbif.search_occurrences

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("GBIF unreachable")
        return real(*a, **kw)

    gbif.search_occurrences = flaky
    with pytest.raises(ConnectionError):
        p.run(TAXON_KEY)
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.status == SpeciesStatus.FAILED and "GBIF unreachable" in sp.last_error
    # Next run takes the incremental path with since=None, i.e. a full re-fetch.
    out = p.run(TAXON_KEY)
    assert out.decision == Decision.AWAITING_REVIEW
    assert p.repo.get_species(TAXON_KEY).n_occurrences_total == 360


def test_scenarios_projected_from_stored_model_with_own_mess(tmp_path, settings):
    """What-if scenarios: no retraining, idempotent, each paired with its own MESS mask."""
    data_root = tmp_path / "data"
    build_stack(data_root / "bioclim")
    (data_root / "native_range_polygons").mkdir()
    p = make_pipeline(
        replace_settings(settings, data_root=data_root), FakeGbif(initial=initial_records())
    )
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    assert p.project_scenarios(TAXON_KEY)["projected"] == []  # none registered yet

    add_synthetic_scenario(data_root / "bioclim", "warm_2050", warming=6.0)
    out = p.project_scenarios(TAXON_KEY)
    assert out["projected"] == ["warm_2050"] and out["model_version"] == 1
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    sc = mv.artifacts["scenarios"]["warm_2050"]
    for key in ("suitability", "extrapolation", "consensus"):
        assert p.artifacts.local_path(sc[key]).exists()
    # Warming pushes BIO1 outside the training range somewhere → more extrapolation.
    assert sc["extrapolated_land_fraction"] > 0
    assert sc["diagnostics_land_fraction"]["mess"] == sc["extrapolated_land_fraction"]
    current = {
        d["id"]: d["land_fraction"]
        for d in mv.metrics["projection"]["extrapolation"]["diagnostics"]
    }
    assert sc["diagnostics_land_fraction"]["exdet"] >= current["exdet"]
    assert p.repo.get_species(TAXON_KEY).model_version == 1  # no retraining
    assert p.project_scenarios(TAXON_KEY)["projected"] == []  # idempotent


def test_full_resync_backfills_records_missed_by_a_capped_ingest(settings):
    """A capped/interrupted ingest leaves gaps that incremental pulls never revisit."""
    all_recs = initial_records()
    gbif = FakeGbif(initial=all_recs[:200])  # first ingest only saw part of the data
    p = make_pipeline(settings, gbif)
    p.run(TAXON_KEY)
    assert p.repo.get_species(TAXON_KEY).n_occurrences_total == 200
    gbif.initial = all_recs  # a full (since=None) query now returns everything
    p.run(TAXON_KEY)  # incremental pull: the gap is never revisited
    assert p.repo.get_species(TAXON_KEY).n_occurrences_total == 200
    p.run(TAXON_KEY, full_resync=True)
    assert p.repo.get_species(TAXON_KEY).n_occurrences_total == 360
    assert gbif.calls[-1] is None  # full re-sync queried without a lastInterpreted filter


def test_failed_retrain_stays_owed_after_the_watermark_advanced(settings):
    """A delta-triggered retrain that crashes must be retried: its records are stored and the
    watermark moved on, so the next run sees no delta."""
    gbif = FakeGbif(initial=initial_records())
    p = make_pipeline(settings, gbif)
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)

    real_fit = p.fit_model

    def crash(*a, **kw):
        raise MemoryError("worker ran out of memory")

    p.fit_model = crash
    gbif.deltas.append(box_records(40, *NATIVE_BOX, start_id=60_000, seed=6))  # > 5 %
    with pytest.raises(MemoryError):
        p.run(TAXON_KEY)
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.status == SpeciesStatus.FAILED and sp.retrain_needed and sp.model_version == 1

    p.fit_model = real_fit
    out = p.run(TAXON_KEY)  # no new records, but the retrain is still owed
    assert out.decision == Decision.TRAINED and out.model_version == 2
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.status == SpeciesStatus.UP_TO_DATE and not sp.retrain_needed


class DownloadingGbif(FakeGbif):
    """GBIF with credentials: first ingest and full re-syncs use the DOI-backed download."""

    def __init__(self, initial):
        super().__init__(initial=initial, download_enabled=True)
        self.downloads = 0

    def download_occurrences(self, taxon_key, workdir):
        from src.connectors.gbif_client import DownloadResult

        self.downloads += 1
        n = self.downloads
        return DownloadResult(f"000{n}-dl", f"10.15468/dl.test{n}", list(self.initial))


def test_full_resync_with_credentials_uses_a_doi_backed_download(settings):
    all_recs = initial_records()
    gbif = DownloadingGbif(initial=all_recs[:200])
    p = make_pipeline(settings, gbif)
    p.run(TAXON_KEY)
    sp = p.repo.get_species(TAXON_KEY)
    assert sp.last_gbif_download_doi == "10.15468/dl.test1" and sp.n_occurrences_total == 200

    gbif.initial = all_recs
    p.run(TAXON_KEY, full_resync=True)
    sp = p.repo.get_species(TAXON_KEY)
    assert gbif.downloads == 2 and gbif.calls == []  # never fell back to the capped search
    assert sp.n_occurrences_total == 360
    assert sp.last_gbif_download_key == "0002-dl"
    assert sp.last_gbif_download_doi == "10.15468/dl.test2"

    p.run(TAXON_KEY)  # routine checks stay on the incremental search API
    assert gbif.downloads == 2 and len(gbif.calls) == 1


def test_target_group_background_mirrors_effort_and_is_logged(settings):
    """Background comes from cells with target-group (order) records only — the fake effort
    has none north of 50°N — and the effort query is recorded with the model version."""
    import pandas as pd

    gbif = FakeGbif(initial=initial_records())
    p = make_pipeline(settings, gbif)
    p.run(TAXON_KEY)
    confirm_native(p)
    out = p.run(TAXON_KEY)
    mv = p.repo.get_model_version(TAXON_KEY, out.model_version)

    bg_info = mv.metrics["reproducibility"]["background"]
    assert bg_info["method"] == "target_group_order" == mv.metrics["background_method"]
    assert "fallback_reason" not in bg_info
    tg = bg_info["target_group"]
    assert (tg["rank"], tg["taxon_key"], tg["name"]) == ("order", 408, "Lamiales")
    assert tg["query"] == "fake://density" and tg["n_records"] > 0
    assert gbif.effort_calls and gbif.effort_calls[0][0] == 408

    td = pd.read_parquet(
        p.artifacts.local_path(mv.metrics["reproducibility"]["training_data"]["path"])
    )
    bg = td[td["role"] == "background"]
    assert len(bg) >= FAST_TRAINING.min_target_group_cells
    assert (bg["latitude"] < 50).all()


def test_target_group_falls_back_to_buffer_when_effort_is_unavailable(settings):
    gbif = FakeGbif(initial=initial_records(), effort_error=ConnectionError("GBIF maps down"))
    p = make_pipeline(settings, gbif)
    p.run(TAXON_KEY)
    confirm_native(p)
    mv = p.repo.get_model_version(TAXON_KEY, p.run(TAXON_KEY).model_version)
    bg_info = mv.metrics["reproducibility"]["background"]
    assert bg_info["requested_method"] == "target_group"
    assert bg_info["method"] == "buffer_500km"
    assert "GBIF maps down" in bg_info["fallback_reason"]


def test_sparse_target_group_falls_back_to_buffer(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.training_config = replace(FAST_TRAINING, min_target_group_cells=10**6)
    p.run(TAXON_KEY)
    confirm_native(p)
    mv = p.repo.get_model_version(TAXON_KEY, p.run(TAXON_KEY).model_version)
    bg_info = mv.metrics["reproducibility"]["background"]
    assert bg_info["method"].startswith("buffer_")
    assert "cells with order Lamiales records" in bg_info["fallback_reason"]
    assert bg_info["target_group"]["taxon_key"] == 408  # what was tried is still logged


@pytest.fixture(scope="module")
def extended_data_root(tmp_path_factory):
    """Static layer with a CHELSA cross-check stack and a finer ('30s') companion stack."""
    from src.features.bioclim_store import add_hires, build_chelsa
    from tests.conftest import BIOCLIM_VERSION, write_synthetic_chelsa, write_synthetic_stack_tif

    root = tmp_path_factory.mktemp("extended")
    build_stack(root / "bioclim")
    (root / "native_range_polygons").mkdir()
    build_chelsa(
        "chelsa_t", BIOCLIM_VERSION, root / "bioclim", src_dir=write_synthetic_chelsa(root / "c")
    )
    add_hires(
        BIOCLIM_VERSION, "30s", root / "bioclim", src_tif=write_synthetic_stack_tif(root / "f.tif")
    )
    return root


def trained_pipeline(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    return p


def test_climate_crosscheck_refits_the_exact_training_set_on_chelsa(settings, extended_data_root):
    s = replace_settings(
        settings, data_root=extended_data_root, crosscheck_bioclim_version="chelsa_t"
    )
    p = trained_pipeline(s)
    out = p.crosscheck_climate(TAXON_KEY)
    assert out["alt_bioclim_version"] == "chelsa_t" and out["model_version"] == 1

    mv = p.repo.get_model_version(TAXON_KEY, 1)
    report = mv.artifacts["crosschecks"]["chelsa_t"]
    assert report["primary_bioclim_version"] == "bioclim_test"
    assert report["n_presence"] == mv.metrics["n_presence"]
    assert report["n_background"] + report["n_dropped"] >= mv.metrics["n_background"] - 1
    assert report["feature_classes"] == mv.metrics["feature_classes"]
    by_pred = {a["predictor"]: a for a in report["predictors"]}
    assert set(by_pred) == set(mv.metrics["predictors"])
    # bio1/bio4/bio12 are the same synthetic climate in both sources (units harmonized).
    assert all(by_pred[p]["pearson_r"] > 0.95 and not by_pred[p]["units_suspect"] for p in by_pred)
    assert report["suitability_rank_correlation"] > 0.8
    assert report["verdict"] in {"consistent", "moderate"}
    assert p.artifacts.local_path(report["path"]).exists()
    # The model version itself is untouched (immutable metrics).
    assert "crosschecks" not in mv.metrics


def test_crosscheck_requires_a_configured_stack(settings):
    p = trained_pipeline(settings)
    with pytest.raises(ValueError, match="SDM_CROSSCHECK_BIOCLIM_VERSION"):
        p.crosscheck_climate(TAXON_KEY)


def test_hires_projection_covers_only_the_region_with_its_own_mess(settings, extended_data_root):
    import rasterio

    p = trained_pipeline(replace_settings(settings, data_root=extended_data_root))
    out = p.project_hires(TAXON_KEY, (5.0, 38.0, 25.0, 52.0))
    assert out["resolution"] == "30s" and out["bbox"] == [5.0, 38.0, 25.0, 52.0]
    entry = p.repo.get_model_version(TAXON_KEY, 1).artifacts["hires"]
    with rasterio.open(p.artifacts.local_path(entry["suitability"])) as src:
        assert (src.width, src.height) == (40, 28) and src.res == (0.5, 0.5)
        assert src.bounds.left == 5.0 and src.bounds.top == 52.0
    for key in ("extrapolation", "consensus"):
        with rasterio.open(p.artifacts.local_path(entry[key])) as src:
            assert (src.width, src.height) == (40, 28)
    assert 0 <= entry["extrapolated_land_fraction"] <= 1
    assert set(entry["diagnostics_land_fraction"]) == {"mess", "exdet", "mop", "shape", "aoa"}

    p.settings = replace_settings(p.settings, hires_max_cells=100)
    with pytest.raises(ValueError, match="choose a smaller area"):
        p.project_hires(TAXON_KEY, (5.0, 38.0, 25.0, 52.0))


def test_hires_requires_a_registered_stack(settings):
    p = trained_pipeline(settings)
    with pytest.raises(ValueError, match="No high-resolution stack"):
        p.project_hires(TAXON_KEY, (5.0, 38.0, 25.0, 52.0))
