"""Remaining improvements: multi-GCM ensembles (U7), model cards (U13), thinning (B3 / B4),
target-group refinement (B1), point process with effort offset (B8), shared bias (B9) and
static bias layers (travel time, …)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import rasterio
from fastapi.testclient import TestClient

from src.api import deps
from src.api.main import create_app
from src.bulletin.model_card import model_card, model_card_markdown
from src.config import Settings
from src.features.background_sampler import distance_thin
from src.features.bioclim_store import add_bias_layer
from src.features.raster_sampler import STACK_FILENAME
from src.features.thinning import (
    autocorrelation_range_km,
    environmental_thin,
    max_retention_thin,
)
from src.modeling.point_process import (
    PpmData,
    SpeciesBiasData,
    fit_ppm,
    fit_shared_bias,
    offset_test,
)
from src.modeling.projection import gcm_ensemble, scenario_groups
from src.orchestration.bias_studies import compare_target_groups, estimate_shared_bias
from src.orchestration.pipeline import Decision
from tests.conftest import (
    BIOCLIM_VERSION,
    TAXON_KEY,
    FakeGbif,
    FakeInat,
    add_synthetic_scenario,
    build_stack,
)
from tests.test_pipeline import confirm_native, initial_records, make_pipeline

GCMS = ("GCMA_ssp245_2041-2060", "GCMB_ssp245_2041-2060")


# ------------------------------------------------------------------ unit: thinning
def test_max_retention_thinning_keeps_at_least_one_greedy_pass():
    rng = np.random.default_rng(3)
    lon, lat = rng.uniform(0, 5, 400), rng.uniform(40, 45, 400)
    single, info1 = max_retention_thin(lon, lat, 30, seed=7, reps=1)
    assert np.array_equal(single, distance_thin(lon, lat, 30, seed=7))  # default unchanged
    assert info1 == {"reps": 1, "selected_seed": 7}
    best, info = max_retention_thin(lon, lat, 30, seed=7, reps=10)
    assert len(best) == info["n_kept_max"] >= len(single) and info["reps"] == 10
    assert "spThin" in info["reference"]


def test_environmental_thinning_keeps_one_record_per_bin():
    rng = np.random.default_rng(4)
    # heavily sampled climate: most records in one corner of predictor space
    x = pd.DataFrame(
        {"a": np.r_[rng.normal(0, 0.05, 500), rng.uniform(-3, 3, 50)], "b": rng.normal(size=550)}
    )
    x["c"] = x["a"] * 0.5 + rng.normal(0, 0.1, 550)
    keep, info = environmental_thin(x, n_bins=10, seed=1, n_axes=2)
    assert len(keep) == info["n_occupied_cells"] < len(x)
    assert info["n_axes"] == 2 and len(info["variance_explained"]) == 2
    assert np.all(np.diff(keep) > 0)


def test_autocorrelation_range_grows_with_spatial_smoothness():
    rng = np.random.default_rng(5)
    lon, lat = rng.uniform(0, 20, 800), rng.uniform(30, 50, 800)
    smooth = pd.DataFrame({"t": np.sin(lon / 6.0) + np.cos(lat / 6.0)})
    rough = pd.DataFrame({"t": rng.normal(size=800)})
    r_smooth = autocorrelation_range_km(lon, lat, smooth, seed=1)
    r_rough = autocorrelation_range_km(lon, lat, rough, seed=1)
    assert r_smooth["range_km"] > r_rough["range_km"] > 0
    assert "blockCV" in r_smooth["reference"]


# ------------------------------------------------------------------ unit: point process
def _synthetic_po(seed: int, beta: float, gamma: float, n: int = 600):
    rng = np.random.default_rng(seed)
    quad = pd.DataFrame({"t": rng.normal(size=4000), "bias_e": rng.uniform(0, 4, 4000)})
    lam = np.exp(beta * quad["t"] - 0.5 * quad["t"] ** 2 + gamma * quad["bias_e"])
    idx = rng.choice(len(quad), n, p=(lam / lam.sum()).to_numpy())
    return quad.iloc[idx].reset_index(drop=True), quad


def test_point_process_offset_is_consistent_with_effort_proportional_records():
    po, quad = _synthetic_po(0, beta=1.0, gamma=1.0)
    data = PpmData(["t"], po, quad, fixed={"bias_e": 1.0})
    fit = fit_ppm(data, l2=0.01)
    assert fit["report"]["converged"]
    assert fit["report"]["coefficients"]["t"] > 0 > fit["report"]["coefficients"]["t²"]
    pred = fit["model"].predict(quad)
    assert pred.min() >= 0 and pred.max() <= 1
    assert pred[quad["t"].abs() < 0.5].mean() > pred[quad["t"].abs() > 2].mean()
    test = offset_test(data, "bias_e", l2=0.01)
    assert abs(test["free_coefficient"] - 1.0) < 0.3 and test["p_value"] > 0.01
    # records that do not scale with effort reject the offset
    po0, quad0 = _synthetic_po(1, beta=1.0, gamma=0.0)
    rejected = offset_test(PpmData(["t"], po0, quad0, fixed={"bias_e": 1.0}), "bias_e", 0.01)
    assert rejected["p_value"] < 0.01 and "reject" in rejected["interpretation"]


def test_shared_bias_recovers_the_common_effort_coefficient():
    species = []
    for i, beta in enumerate((1.0, -0.5, 0.3)):
        po, quad = _synthetic_po(10 + i, beta=beta, gamma=0.8, n=400)
        species.append(SpeciesBiasData(i, ["t"], po, quad))
    fit = fit_shared_bias(species, ["bias_e"], l2=0.01)
    assert fit["converged"] and abs(fit["coefficients"]["bias_e"] - 0.8) < 0.2
    assert [s["taxon_key"] for s in fit["species"]] == [0, 1, 2]
    with pytest.raises(ValueError):
        fit_shared_bias(species[:1], ["bias_e"])


# ------------------------------------------------------------------ unit: GCM ensembles
def test_scenario_groups_need_two_gcms_per_ssp_and_period():
    names = [*GCMS, "GCMA_ssp585_2041-2060", "ensemble_ssp245_2041-2060", "custom"]
    assert scenario_groups(names) == {"ssp245_2041-2060": list(GCMS)}


def _raster(path, arr, dtype, nodata):
    profile = dict(
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=rasterio.transform.from_origin(0, 10, 1, 1),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def test_gcm_ensemble_mean_agreement_and_any_gcm_extrapolation(tmp_path):
    a = np.full((10, 10), 0.2)
    a[:, :5] = 0.8
    b = np.full((10, 10), 0.2)
    b[:, :3] = 0.9
    b[0, 0] = -9999.0  # nodata in one member → nodata in the ensemble
    ea, eb = np.zeros((10, 10)), np.zeros((10, 10))
    ea[9, :] = 1
    members = {
        "G1_ssp1_p": (
            _raster(tmp_path / "s1.tif", a, "float32", -9999.0),
            _raster(tmp_path / "e1.tif", ea, "uint8", 255),
        ),
        "G2_ssp1_p": (
            _raster(tmp_path / "s2.tif", b, "float32", -9999.0),
            _raster(tmp_path / "e2.tif", eb, "uint8", 255),
        ),
    }
    out = gcm_ensemble(members, 0.5, tmp_path, "ssp1_p")
    with rasterio.open(out.suitability) as s, rasterio.open(out.agreement) as g:
        mean, votes = s.read(1), g.read(1)
    with rasterio.open(out.extrapolation) as e:
        ext = e.read(1)
    assert mean[0, 0] == -9999.0 and votes[0, 0] == 255
    assert mean[1, 1] == pytest.approx(0.85) and votes[1, 1] == 2
    assert votes[1, 4] == 1 and votes[1, 8] == 0
    assert set(np.unique(ext[9, 1:])) == {1} and ext[5, 5] == 0
    sm = out.summary
    assert sm["n_gcms"] == 2 and sm["gcms"] == ["G1_ssp1_p", "G2_ssp1_p"]
    assert sm["suitable_any_gcm_land_fraction"] > sm["suitable_all_gcms_land_fraction"] > 0
    assert sm["gcm_disagreement_land_fraction"] > 0
    assert sm["extrapolated_any_gcm_land_fraction"] > sm["extrapolated_all_gcms_land_fraction"]


# ------------------------------------------------------------------ pipeline
@pytest.fixture(scope="module")
def fresh_root(tmp_path_factory):
    """Own static layer: two GCMs of one SSP × period and a travel-time bias raster."""
    root = tmp_path_factory.mktemp("p4data")
    bioclim = root / "bioclim"
    vdir = build_stack(bioclim)
    (root / "native_range_polygons").mkdir()
    for name, warming in zip(GCMS, (2.0, 4.0), strict=True):
        add_synthetic_scenario(bioclim, name, warming)
    with rasterio.open(vdir / STACK_FILENAME) as src:
        profile = {**src.profile, "count": 1, "dtype": "float32", "nodata": -1.0}
        tr = src.transform
        lon = tr.c + (np.arange(src.width) + 0.5) * tr.a
        lat = tr.f + (np.arange(src.height) + 0.5) * tr.e
    # travel time grows with distance from (10° E, 45° N), in hours
    lon2, lat2 = np.meshgrid(lon, lat)
    hours = (np.hypot(lon2 - 10, lat2 - 45) * 5).astype("float32")
    src_tif = root / "travel_time_src.tif"
    with rasterio.open(src_tif, "w", **profile) as dst:
        dst.write(hours, 1)
    entry = add_bias_layer(BIOCLIM_VERSION, "travel_time", src_tif, bioclim)
    assert entry["direction"] == "decreasing" and entry["transform"] == "log1p"
    return root


@pytest.fixture(scope="module")
def study(tmp_path_factory, fresh_root):
    s = Settings(
        _env_file=None,
        crosscheck_bioclim_version=None,
        database_url="sqlite://",
        data_root=fresh_root,
        artifact_root=tmp_path_factory.mktemp("p4") / "artifacts",
        bioclim_version=BIOCLIM_VERSION,
        mlflow_tracking_uri=None,
        shared_bias_min_species=3,
    )
    gbif = FakeGbif(initial=initial_records())
    p = make_pipeline(s, gbif, FakeInat())
    p.training_config = replace(p.training_config, bias_static_layers=("travel_time",))
    p.run(TAXON_KEY)
    confirm_native(p)
    assert p.run(TAXON_KEY).decision == Decision.TRAINED
    v1 = p.repo.get_model_version(TAXON_KEY, 1)

    comparison = compare_target_groups(p, TAXON_KEY, ranks=("order", "phylum"))
    # two more registered species of the same order (copies of the first)
    sp = p.repo.get_species(TAXON_KEY)
    for k in (TAXON_KEY + 1, TAXON_KEY + 2):
        p.repo.create_species(replace(sp, taxon_key=k, scientific_name=f"Copy {k}"))
        p.repo.add_model_version(replace(v1, taxon_key=k))
        p.repo.update_species(k, model_version=1)
    shared = estimate_shared_bias(p, TAXON_KEY)

    gbif.effort_filters.clear()
    p.training_config = replace(
        p.training_config,
        algorithm="ppm",
        ppm_bias="shared",
        thin_method="autocorrelation",
        thin_reps=3,
        target_group_match_source=True,
        target_group_match_years=True,
    )
    assert p.run(TAXON_KEY, force_retrain=True).decision == Decision.TRAINED
    v2 = p.repo.get_model_version(TAXON_KEY, 2)
    return {"p": p, "gbif": gbif, "v1": v1, "v2": v2, "comparison": comparison, "shared": shared}


def test_gcm_ensemble_is_projected_and_served_with_its_own_extrapolation(study):
    p, v1 = study["p"], study["v1"]
    entry = v1.artifacts["scenarios"]["ensemble_ssp245_2041-2060"]
    ens = entry["gcm_ensemble"]
    assert ens["gcms"] == list(GCMS) and ens["n_gcms"] == 2
    assert entry["extrapolation"] and entry["change"] and ens["agreement"]
    with rasterio.open(p.artifacts.local_path(ens["agreement"])) as src:
        assert set(np.unique(src.read(1))) <= {0, 1, 2, 255}
    # ensemble mean = mean of the members
    sa, sb, se = (
        p.artifacts.local_path(v1.artifacts["scenarios"][n]["suitability"])
        for n in (*GCMS, "ensemble_ssp245_2041-2060")
    )
    with rasterio.open(sa) as a, rasterio.open(sb) as b, rasterio.open(se) as e:
        av, bv, ev = a.read(1), b.read(1), e.read(1)
    ok = (av != -9999) & (bv != -9999)
    assert np.allclose(ev[ok], (av[ok] + bv[ok]) / 2, atol=1e-5)
    # API: the ensemble is a scenario with an agreement layer and its own mask
    app = create_app()
    app.dependency_overrides[deps.repository_dep] = lambda: p.repo
    app.dependency_overrides[deps.settings_dep] = lambda: p.settings
    app.dependency_overrides[deps.artifacts_dep] = lambda: p.artifacts
    layers = TestClient(app).get(f"/species/{TAXON_KEY}/layers").json()
    sc = {s["name"]: s for s in layers["scenarios"]}["ensemble_ssp245_2041-2060"]
    assert sc["label"].startswith("GCM ensemble ssp245 2041-2060 (2 GCMs)")
    assert sc["extrapolation"]["id"] == "extrapolation:ensemble_ssp245_2041-2060"
    agree = sc["gcm_ensemble"]["agreement"]
    assert agree["id"] == "gcm-agreement:ensemble_ssp245_2041-2060"
    assert [e["label"] for e in agree["legend"]][-1] == "2 of 2 GCMs (all)"
    # re-projecting rebuilds (never duplicates) the ensemble
    out = p.project_scenarios(TAXON_KEY, only_missing=False)
    assert out["gcm_ensembles"] == ["ensemble_ssp245_2041-2060"]


def test_static_bias_layer_is_a_conditioned_covariate(study):
    repro = study["v1"].metrics["reproducibility"]
    layer = repro["bias_covariates"]["static_layers"][0]
    assert layer["name"] == "travel_time" and layer["status"] == "used"
    assert layer["direction"] == "decreasing" and "Weiss" in layer["citation"]
    # held at the most-accessible end (minimum travel time) for prediction
    assert layer["conditioning_value"] <= layer["background_mean"]
    assert repro["training_config"]["bias_static_layers"] == ["travel_time"]
    assert repro["thinning"]["method"] == "distance" and repro["thinning"]["distance_km"] == 50


def test_model_card_puts_confidence_and_caveats_first(study):
    p, v1 = study["p"], study["v1"]
    card = model_card(p.repo.get_species(TAXON_KEY), v1)
    assert card["caveats"][0] == v1.metrics["transferability_caveat"]
    assert card["confidence_label"] == v1.metrics["confidence_label"]
    assert [s["id"] for s in card["sections"]][:2] == ["what", "quality"]
    md = model_card_markdown(card)
    assert md.startswith("# Model card") and "**Confidence:" in md
    assert md.index("Confidence") < md.index("How good is the model?")


def test_target_group_comparison_reports_every_option(study):
    report = study["comparison"]
    options = {o["option"]: o for o in report["options"]}
    assert set(options) == {"order", "phylum", "order (matched source + years)"}
    assert options["phylum"]["target_group_taxon_key"] == 7_707_728
    matched = options["order (matched source + years)"]["matched"]
    assert matched["basis_of_record"] == ["HUMAN_OBSERVATION"] and len(matched["years"]) == 2
    assert all(o.get("cv_cbi_mean") is not None for o in report["options"])
    assert report["best_by_cv_cbi"] in options
    stored = study["p"].artifacts.local_path(f"target_group_comparisons/{TAXON_KEY}.json")
    assert json.loads(stored.read_text())["comparison_id"] == report["comparison_id"]


def test_shared_bias_estimate_feeds_the_point_process(study):
    shared, v2 = study["shared"], study["v2"]
    assert len(shared["species"]) == 3 and shared["target_group"]["taxon_key"] == 408
    assert set(shared["columns"]) == {"bias_target_group_effort", "bias_travel_time"}
    m = v2.metrics
    assert m["algorithm"] == "ppm"
    pp = m["point_process"]
    assert pp["bias_mode"] == "shared" and pp["converged"]
    assert pp["fixed_bias_coefficients"] == pytest.approx(shared["coefficients"], abs=1e-5)
    assert pp["offset_test"]["column"] in shared["columns"]
    algo = m["reproducibility"]["algorithm"]
    assert algo["used"] == "ppm" and algo["point_process"]["shared_bias"]["n_species"] == 3
    assert m["reproducibility"]["background"]["bias_covariate_override"] == "ppm"
    assert m["reproducibility"]["bias_covariates"]["condition"] == "offset"


def test_autocorrelation_thinning_and_matched_effort_are_logged(study):
    repro = study["v2"].metrics["reproducibility"]
    thin = repro["thinning"]
    assert thin["method"] == "autocorrelation" and thin["reps"] == 3
    ac = thin["autocorrelation"]
    assert ac["clamped_to_km"] == [50, 50.0] and thin["distance_km"] == 50
    assert set(ac["range_km_by_predictor"]) == set(study["v2"].metrics["predictors"])
    assert repro["random_seeds"]["thinning_variogram"] == 46
    filters = study["gbif"].effort_filters
    assert filters and all(f["basis_of_record"] == ["HUMAN_OBSERVATION"] for f in filters[:1])
    assert filters[0]["years"][0] >= 2015


def test_bulletin_methods_describe_thinning_static_layers_and_point_process(study):
    from src.bulletin.generator import BulletinGenerator

    p = study["p"]
    gen = BulletinGenerator(p.repo, p.artifacts, p.settings)
    v1 = " ".join(gen.render_html(TAXON_KEY, 1).split())
    assert "distance-thinned (50" in v1 and "travel time (decreasing effort" in v1
    v2 = " ".join(gen.render_html(TAXON_KEY, 2).split())
    assert "spatial autocorrelation range" in v2 and "best of 3 passes (spThin)" in v2
    assert "Poisson point process (experimental)" in v2 and "bias mode shared" in v2
    assert "Likelihood-ratio test of the offset coefficient" in v2
    assert "HUMAN_OBSERVATION" in v2
