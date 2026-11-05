"""Pipeline integration of priority 2/3: thresholds, response curves, limiting factor,
ensembles + pseudo-absences, integrated model, change maps, survey priority, inferred
absences, regional models, record annotations, early-detection alerts and the (proposed,
off by default) effort-aware retrain trigger."""

from __future__ import annotations

from dataclasses import replace

import elapid
import numpy as np
import pandas as pd
import pytest
import rasterio

from src.domain import AlertSubscription, RecordAnnotation, SpeciesStatus
from src.modeling.algorithms import WeightedEnsemble
from src.orchestration.pipeline import Decision
from tests.conftest import (
    TAXON_KEY,
    FakeGbif,
    add_synthetic_scenario,
    box_records,
    build_stack,
)
from tests.test_pipeline import confirm_native, initial_records, make_pipeline
from tests.test_pipeline_quality import upload

ENSEMBLE = dict(
    algorithm="ensemble",
    ensemble_members=("maxent", "glm", "rf"),
    n_pseudo_absences=800,
    pa_repeats=2,
    pa_min_distance_km=100.0,
)


def trained(settings, gbif=None, **cfg):
    p = make_pipeline(settings, gbif or FakeGbif(initial=initial_records()))
    p.training_config = replace(p.training_config, **cfg)
    p.run(TAXON_KEY)
    confirm_native(p)
    assert p.run(TAXON_KEY).decision == Decision.TRAINED
    return p, p.repo.get_model_version(TAXON_KEY, 1)


@pytest.fixture(scope="module")
def default_version(tmp_path_factory, data_root):
    from src.config import Settings
    from tests.conftest import BIOCLIM_VERSION

    s = Settings(
        _env_file=None,
        crosscheck_bioclim_version=None,
        database_url="sqlite://",
        data_root=data_root,
        artifact_root=tmp_path_factory.mktemp("p23") / "artifacts",
        bioclim_version=BIOCLIM_VERSION,
        mlflow_tracking_uri=None,
    )
    return trained(s)


def test_every_threshold_rule_response_curves_and_new_layers(default_version):
    p, mv = default_version
    m = mv.metrics
    assert set(m["thresholds"]) >= {"p10", "max_tss", "equal_sens_spec", "min_training_presence"}
    assert m["threshold"] == m["thresholds"]["p10"]["value"]
    assert m["threshold_rule"] == "10th percentile training presence"
    assert m["algorithm"] == "maxent"
    curves = m["response_curves"]["curves"]
    assert [c["predictor"] for c in curves] == m["predictors"]
    # limiting factor raster + summary (Elith et al. 2010)
    with rasterio.open(p.artifacts.local_path(mv.artifacts["limiting"])) as src:
        codes = set(np.unique(src.read(1))) - {255}
    assert codes <= {0, 1, 2, 254} and codes
    lim = m["projection"]["limiting_factor"]
    assert {e["variable"] for e in lim["land"]} <= {*m["predictors"], "none"}
    # survey priority from the effort layer, ranked cells inside the suitable area
    prio = m["survey_priority"]
    assert prio["status"] == "available" and prio["uncertainty_source"] == "consensus"
    assert prio["top_cells"][0]["rank"] == 1 and "priority" in mv.artifacts
    # inferred absences: fake effort (3 records per pixel) never reaches 50 → insufficient
    assert m["inferred_absences"]["status"] == "insufficient"
    repro = m["reproducibility"]
    assert repro["algorithm"] == {"requested": "maxent", "used": "maxent"}
    assert repro["annotations"]["n_records_annotated"] == 0
    assert repro["study_region"] is None and "change_previous" not in m


def test_configured_threshold_rule_binarises_the_zones(settings):
    _, mv = trained(settings, threshold_rule="max_tss")
    m = mv.metrics
    assert m["threshold"] == m["thresholds"]["max_tss"]["value"]
    assert m["threshold_rule"].startswith("maximum TSS")


def test_ensemble_with_disk_pseudo_absences(settings):
    p, mv = trained(settings, **ENSEMBLE)
    m = mv.metrics
    assert m["algorithm"] == "ensemble"
    members = {e["algorithm"]: e for e in m["ensemble"]["members"]}
    assert set(members) == {"maxent", "glm", "rf"}
    assert sum(e["weight"] for e in members.values()) == pytest.approx(1.0, abs=1e-3)
    assert "pseudo-absences" in m["cv_method"]
    pa = m["reproducibility"]["pseudo_absences"]
    assert pa["strategy"] == "disk" and pa["n"] == 800 and pa["min_distance_km"] == 100
    assert m["reproducibility"]["random_seeds"]["pseudo_absence_sampling"] == 44
    training = pd.read_parquet(p.artifacts.local_path(mv.artifacts["training_data"]))
    assert (training["role"] == "pseudo_absence").sum() == 800
    saved = elapid.load_object(str(p.artifacts.local_path(mv.artifacts["model"])))
    assert isinstance(saved["model"], WeightedEnsemble)
    assert "agreement" in mv.artifacts
    agree = m["projection"]["ensemble_agreement"]
    assert 0.5 <= agree["land_mean"] <= 1.0
    # the survey-priority layer prefers ensemble disagreement as its uncertainty
    assert m["survey_priority"]["uncertainty_source"] == "agreement"
    from src.bulletin.generator import BulletinGenerator

    html = BulletinGenerator(p.repo, p.artifacts, p.settings).render_html(TAXON_KEY, 1)
    assert "CV-weighted ensemble" in html and "800 pseudo-absences" in html
    with pytest.raises(ValueError, match="MaxEnt versions only"):
        p.settings = p.settings.model_copy(update={"crosscheck_bioclim_version": "other"})
        p.crosscheck_climate(TAXON_KEY)


@pytest.mark.parametrize("strategy", ["environmental_profile", "completeness"])
def test_other_pseudo_absence_strategies(settings, strategy):
    _, mv = trained(settings, **{**ENSEMBLE, "pseudo_absence_strategy": strategy})
    pa = mv.metrics["reproducibility"]["pseudo_absences"]
    assert pa["strategy"] == strategy and pa["n"] > 0
    if strategy == "completeness":  # fake effort is far below 50 records → disk fallback
        assert "well-surveyed" in pa["fallback_reason"] and pa["method"].startswith("disk")


def survey_csv() -> str:
    rows = ["longitude,latitude,date,presence"]
    rows += [f"{2 + i * 1.1},{41 + (i % 13)},2021-05-01,1" for i in range(30)]
    rows += [f"{100 + i * 0.5},5.0,2021-05-01,0" for i in range(15)]
    rows += [f"{-60 + i},{-20 - (i % 5)},2021-05-01,0" for i in range(15)]
    return "\n".join(rows)


def test_integrated_model_falls_back_without_survey_data(settings):
    _, mv = trained(settings, algorithm="integrated")
    algo = mv.metrics["reproducibility"]["algorithm"]
    assert algo["used"] == "maxent" and "absences" in algo["fallback_reason"]


def test_integrated_model_uses_survey_absences_and_estimates_bias(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.training_config = replace(p.training_config, algorithm="integrated")
    p.run(TAXON_KEY)
    confirm_native(p)
    upload(p, survey_csv(), name="Plots", role="training")
    assert p.run(TAXON_KEY).decision == Decision.TRAINED
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    m = mv.metrics
    assert m["algorithm"] == "integrated"
    integ = m["integrated"]
    assert integ["n_pa_absent"] == 30 and integ["n_pa_present"] == 30
    assert "bias_target_group_effort" in integ["bias_coefficients"]
    (up,) = m["reproducibility"]["training_uploads"]
    assert up["used_as"] == "survey" and up["n_added"] == 0 and up["n_survey_absences"] == 30
    bias = m["reproducibility"]["bias_covariates"]
    assert bias["condition"] == "estimated" and bias["conditioning_value"] is None
    training = pd.read_parquet(p.artifacts.local_path(mv.artifacts["training_data"]))
    assert (training["role"] == "survey_absence").sum() == 30
    assert m["reproducibility"]["background"]["bias_covariate_override"] == "integrated"


def test_change_since_previous_version_and_scenario_change(tmp_path, settings):
    root = tmp_path / "data"
    build_stack(root / "bioclim")
    (root / "native_range_polygons").mkdir()
    add_synthetic_scenario(root / "bioclim", "ssp585_2050", warming=4.0)
    s = settings.model_copy(update={"data_root": root})
    p, mv1 = trained(s)
    sc = mv1.artifacts["scenarios"]["ssp585_2050"]
    assert sc["change"] and sc["change_summary"]["gain_km2"] >= 0
    out = p.run(TAXON_KEY, force_retrain=True)
    mv2 = p.repo.get_model_version(TAXON_KEY, out.model_version)
    ch = mv2.metrics["change_previous"]
    assert ch["status"] == "available" and (ch["from_version"], ch["to_version"]) == (1, 2)
    total = ch["gain_km2"] + ch["loss_km2"] + ch["stable_suitable_km2"]
    assert total > 0 and "change_previous" in mv2.artifacts


def test_regional_study_region_restricts_training_and_projection(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    p.repo.update_species(TAXON_KEY, study_region={"bbox": [-20, 25, 50, 65], "resolution": None})
    assert p.run(TAXON_KEY).decision == Decision.TRAINED
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    region = mv.metrics["reproducibility"]["study_region"]
    assert region["bbox"] == [-20, 25, 50, 65] and region["n_records_outside"] == 60
    assert mv.model_type == "A"  # the invaded-range records lie outside the region
    with rasterio.open(p.artifacts.local_path(mv.artifacts["suitability"])) as src:
        b = src.bounds
    assert (b.left, b.bottom, b.right, b.top) == (-20, 25, 50, 65)
    training = pd.read_parquet(p.artifacts.local_path(mv.artifacts["training_data"]))
    assert training["longitude"].between(-20, 50).all()
    assert training["latitude"].between(25, 65).all()


def test_annotations_change_the_training_set_and_are_snapshotted(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    occ = p.repo.load_occurrences(TAXON_KEY)
    gbif_ids = occ.loc[occ["source"] == "GBIF", "record_id"].astype(int).tolist()
    for rid in gbif_ids[:5]:
        p.repo.add_annotation(RecordAnnotation(TAXON_KEY, "GBIF", rid, "cultivated", "true"))
    p.repo.add_annotation(RecordAnnotation(TAXON_KEY, "GBIF", gbif_ids[10], "exclude", "true"))
    # introduced-range records relabelled native by an expert
    intro = occ.loc[occ["range_label"] == "introduced", "record_id"].astype(int).tolist()[:3]
    for rid in intro:
        p.repo.add_annotation(RecordAnnotation(TAXON_KEY, "GBIF", rid, "range_label", "native"))
    p.run(TAXON_KEY)
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    info = mv.metrics["reproducibility"]["annotations"]
    assert info["n_records_annotated"] == 9 and info["n_excluded_cultivated"] == 5
    assert info["n_excluded_suspicious"] == 1 and info["n_label_overrides"] == 3
    assert info["snapshot"]["n_rows"] == 9
    manifest = pd.read_parquet(p.artifacts.local_path(mv.artifacts["occurrence_manifest"]))
    assert not manifest["record_id"].isin(gbif_ids[:5]).any()
    eff = manifest.set_index("record_id").loc[intro, "range_label_effective"]
    assert (eff == "native").all()


def far_records(start_id: int, n: int = 6):
    # Southern "Africa"-like part of the synthetic Eurasia block: hot, far from everything
    return box_records(n, (20.0, 30.0), (-30.0, -25.0), start_id, seed=9, year=2025)


def test_early_detection_alerts_with_effort_classes(settings, monkeypatch):
    p, _ = trained(settings)
    p.settings = p.settings.model_copy(update={"effort_sampled_min_records": 2})
    p.gbif.deltas.append(far_records(50_000))
    p.repo.save_subscription(AlertSubscription("webhook", "https://hooks.example.org/x"))
    sent = []
    monkeypatch.setattr(
        "src.orchestration.pipeline.deliver",
        lambda alert, subs, name, s: sent.append(alert.kind) or [{"status": "sent"}],
    )
    out = p.run(TAXON_KEY)
    early = out.details["early_detection"]
    assert early["n_new_records_checked"] == 6
    assert early["outside_known_range"] == 6
    alerts = {a.kind: a for a in p.repo.list_alerts(TAXON_KEY)}
    far = alerts["outside_known_range"]
    assert far.n_records == 6 and far.records[0]["distance_km"] > 100
    # fake effort exists south of 50°N → previously sampled → "expansion"
    assert far.effort_classes["expansion"] == 6 and "expansion" in far.summary
    assert "outside_known_range" in sent and far.deliveries == [{"status": "sent"}]
    # the same records never alert twice
    p.gbif.deltas.append(far_records(50_000))
    p.run(TAXON_KEY, full_resync=True)
    assert len(p.repo.list_alerts(TAXON_KEY)) == len(alerts)


@pytest.mark.parametrize(("mode", "retrains"), [("envelope", True), ("effort_aware", False)])
def test_effort_aware_trigger_is_opt_in(settings, mode, retrains):
    # Hot southern records: unsuitable, and this target group was never recorded south of the
    # equator, so B10 classes them as discoveries. Only the proposed mode ignores them.
    class NorthernEffort(FakeGbif):
        def fetch_effort_density(self, taxon_key, bbox, max_pixel_deg):
            eff = super().fetch_effort_density(taxon_key, bbox, max_pixel_deg)
            keep = [i for i, la in enumerate(eff.lat) if la > 0]
            return replace(
                eff,
                lon=[eff.lon[i] for i in keep],
                lat=[eff.lat[i] for i in keep],
                count=[eff.count[i] for i in keep],
            )

    p, _ = trained(settings, NorthernEffort(initial=initial_records()))
    p.gbif.deltas.append(far_records(70_000, n=3))
    p.settings = p.settings.model_copy(
        update={"retrain_trigger_mode": mode, "effort_sampled_min_records": 2}
    )
    out = p.run(TAXON_KEY)
    assert (out.decision == Decision.RETRAINED) is retrains
    if not retrains:
        assert out.decision == Decision.MINOR_UPDATE
        assert p.repo.get_species(TAXON_KEY).status == SpeciesStatus.MINOR_UPDATE_PENDING
    classes = out.details["early_detection"]["outside_envelope_effort_classes"]
    assert classes["discovery"] == 3 and classes["expansion"] == 0
