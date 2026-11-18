"""Choosing the climate data (WorldClim / CHELSA) and the predictors per species.

* `build-chelsa` with extra CHELSA-BIOCLIM+ variables (climate layers) and a 30″-style
  high-resolution grid; CHELSA CMIP6 scenarios on CHELSA versions only;
* present-only predictors (relative humidity has no CHELSA future data) block projection onto
  futures, with the reason recorded, instead of silently reusing present values;
* per-species climate version + custom predictors, VIF-screened on the species' background;
* predictor-set comparison on a version's exact training locations.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.features.bioclim_store import add_scenario, add_soil, build_chelsa
from src.features.raster_sampler import (
    METADATA_FILENAME,
    BioclimMetadata,
    BioclimStack,
    PresentOnlyPredictorError,
    choose_crosscheck_version,
)
from tests.conftest import (
    BIOCLIM_VERSION,
    TAXON_KEY,
    FakeGbif,
    build_stack,
    write_synthetic_chelsa,
    write_synthetic_chelsa_extras,
    write_synthetic_soilgrids,
    write_synthetic_stack_tif,
)
from tests.test_pipeline import confirm_native, initial_records, make_pipeline

CHELSA = "chelsa_test"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every file comes from the synthetic src_dir: a download attempt is a test bug."""

    def refuse(url, dest):
        raise AssertionError(f"test tried to download {url}")

    monkeypatch.setattr("src.features.bioclim_store._download", refuse)


FUTURE = ("MPI-ESM1-2-HR", "ssp585", "2041-2070")
FUTURE_NAME = "MPI-ESM1-2-HR_ssp585_2041-2070"


@pytest.fixture(scope="module")
def climate_root(tmp_path_factory):
    from src.features.bioclim_store import add_hires

    root = tmp_path_factory.mktemp("climate")
    bioclim = root / "bioclim"
    build_stack(bioclim)
    (root / "native_range_polygons").mkdir()
    add_hires(BIOCLIM_VERSION, "30s", bioclim, src_tif=write_synthetic_stack_tif(root / "f.tif"))
    src = write_synthetic_chelsa(root / "chelsa")
    write_synthetic_chelsa_extras(src, ("hurs_mean", "gdd5"), future=FUTURE)
    build_chelsa(
        CHELSA,
        BIOCLIM_VERSION,
        bioclim,
        src_dir=src,
        variables=("hurs_mean", "gdd5"),
        with_hires=True,
    )
    add_scenario(CHELSA, *FUTURE, bioclim, src_dir=src)
    return root


def chelsa_meta(root) -> BioclimMetadata:
    return BioclimMetadata.load(root / "bioclim" / CHELSA / METADATA_FILENAME)


# --------------------------------------------------------------------------- building
def test_chelsa_version_has_family_baseline_extras_and_hires(climate_root):
    meta = chelsa_meta(climate_root)
    assert meta.family == "chelsa" and meta.baseline == "1981-2010"
    assert meta.predictor_bands[-2:] == ("hurs_mean", "gdd5")
    assert set(meta.hires) == {"30s"}
    hurs = meta.climate_layers["hurs_mean"]
    assert hurs["group"] == "humidity" and hurs["kind"] == "climate"
    assert "30s" in hurs["hires"] and len(hurs["sha256"]) == 64
    stack = BioclimStack.open_version(climate_root / "bioclim", CHELSA)
    lon, lat = np.array([10.25, -80.25]), np.array([45.25, 40.25])
    df = stack.sample(lon, lat, ["bio1", "hurs_mean", "gdd5"])
    # scale/offset applied: hurs = 40 + 0.4·|lat| + 0.05·lon (%), averaged over the 1° cell
    assert df["hurs_mean"].to_numpy() == pytest.approx(40 + 0.4 * np.abs(lat) + 0.05 * lon, abs=0.3)
    assert df["bio1"].to_numpy() == pytest.approx(28 - 0.45 * np.abs(lat), abs=0.3)
    # The 30″-style grid exists for CHELSA too (finest resolution for regional models).
    hires = BioclimStack.open_version(climate_root / "bioclim", CHELSA, hires="30s")
    assert hires.res == (0.5, 0.5)
    hv = hires.sample(lon, lat, ["bio1", "hurs_mean"])
    assert hv["hurs_mean"].to_numpy() == pytest.approx(40 + 0.4 * np.abs(lat) + 0.05 * lon, abs=0.1)


def test_chelsa_futures_and_present_only_layers(climate_root):
    meta = chelsa_meta(climate_root)
    assert FUTURE_NAME in meta.scenarios
    # CHELSA publishes gdd5 for futures but not relative humidity.
    assert FUTURE_NAME in meta.climate_layers["gdd5"]["scenarios"]
    assert meta.climate_layers["hurs_mean"]["scenarios"] == {}
    assert meta.unavailable_in_scenario(FUTURE_NAME, ["bio1", "gdd5", "hurs_mean"]) == ["hurs_mean"]
    assert meta.present_only_layers() == ["hurs_mean"]
    bioclim = climate_root / "bioclim"
    fut = BioclimStack.open_version(bioclim, CHELSA, scenario=FUTURE_NAME)
    now = BioclimStack.open_version(bioclim, CHELSA)
    pts = ([10.25], [45.25])
    assert fut.sample(*pts, ["bio1"]).iloc[0, 0] == pytest.approx(
        now.sample(*pts, ["bio1"]).iloc[0, 0] + 3.0, abs=0.05
    )
    assert fut.sample(*pts, ["gdd5"]).iloc[0, 0] > now.sample(*pts, ["gdd5"]).iloc[0, 0]
    with pytest.raises(PresentOnlyPredictorError, match="present-climate only"):
        fut.sample(*pts, ["hurs_mean"])


def test_scenarios_never_mix_climate_sources(climate_root, tmp_path):
    bioclim = climate_root / "bioclim"
    with pytest.raises(ValueError, match="must come from CHELSA too"):
        add_scenario(CHELSA, *FUTURE, bioclim, source="worldclim")
    with pytest.raises(ValueError, match="CHELSA futures need a CHELSA version"):
        add_scenario(BIOCLIM_VERSION, *FUTURE, bioclim, source="chelsa")
    with pytest.raises(ValueError, match="CHELSA CMIP6 GCMs"):
        add_scenario(CHELSA, "ACCESS-CM2", "ssp585", "2041-2070", bioclim, src_dir=tmp_path)


def test_crosscheck_partner_comes_from_the_other_family(climate_root):
    bioclim = climate_root / "bioclim"
    assert choose_crosscheck_version(bioclim, BIOCLIM_VERSION, [CHELSA, BIOCLIM_VERSION]) == CHELSA
    # A CHELSA-trained model is checked against WorldClim, not against CHELSA itself.
    assert choose_crosscheck_version(bioclim, CHELSA, [CHELSA, BIOCLIM_VERSION]) == BIOCLIM_VERSION
    assert choose_crosscheck_version(bioclim, CHELSA, [CHELSA, "missing"]) is None


def test_soil_can_be_added_to_a_chelsa_version(climate_root):
    bioclim = climate_root / "bioclim"
    add_soil(
        CHELSA,
        "chelsa_test_soil",
        bioclim,
        properties=("sand",),
        src_dir=write_synthetic_soilgrids(climate_root / "sg", props=("sand",)),
    )
    meta = BioclimMetadata.load(bioclim / "chelsa_test_soil" / METADATA_FILENAME)
    assert meta.family == "chelsa"
    assert meta.predictor_bands[-3:] == ("hurs_mean", "gdd5", "soil_sand")
    # Copied climate layers (and their scenario files) still resolve.
    fut = BioclimStack.open_version(bioclim, "chelsa_test_soil", scenario=FUTURE_NAME)
    assert np.isfinite(fut.sample([10.25], [45.25], ["gdd5", "soil_sand"]).to_numpy()).all()


# --------------------------------------------------------------------------- pipeline
def chelsa_settings(settings, climate_root):
    return settings.model_copy(
        update={"data_root": climate_root, "crosscheck_bioclim_version": CHELSA}
    )


def trained_on_chelsa(settings, climate_root, predictor_config):
    p = make_pipeline(chelsa_settings(settings, climate_root), FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    p.repo.update_species(TAXON_KEY, climate_version=CHELSA, predictor_config=predictor_config)
    confirm_native(p)
    p.run(TAXON_KEY)
    return p


def test_species_trains_on_chosen_climate_and_predictors(settings, climate_root):
    cfg = {
        "mode": "custom",
        "predictors": ["bio1", "bio4", "bio12", "hurs_mean", "bio7", "not_here"],
        "vif_screen": True,
        "protected": ["hurs_mean"],
    }
    p = trained_on_chelsa(settings, climate_root, cfg)
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    repro = mv.metrics["reproducibility"]
    assert repro["bioclim_version_used"] == CHELSA
    assert repro["climate_source"] == {
        "family": "chelsa",
        "baseline_period": "1981-2010",
        "resolution": "1.0deg",
        "chosen_for_species": True,
    }
    sel = repro["predictor_selection"]
    assert sel["mode"] == "custom" and sel["unavailable"] == ["not_here"]
    assert "hurs_mean" in sel["used"] and sel["used"] == mv.metrics["predictors"]
    assert set(sel["dropped"]) | set(sel["used"]) == {"bio1", "bio4", "bio12", "hurs_mean", "bio7"}
    assert sel["present_only"] == ["hurs_mean"]
    # CHELSA's 1981 baseline drives the data-quality date test.
    assert repro["data_quality"]["config"]["baseline_start_year"] == 1981
    # Relative humidity has no future data: the future is reported unavailable, not faked.
    assert FUTURE_NAME not in mv.artifacts["scenarios"]
    assert "hurs_mean" in mv.artifacts["scenarios_unavailable"][FUTURE_NAME]
    # Cross-checked against WorldClim (the other family), humidity kept at training values.
    p.crosscheck_climate(TAXON_KEY)
    report = p.repo.get_model_version(TAXON_KEY, 1).artifacts["crosschecks"]
    assert list(report) == [BIOCLIM_VERSION]
    assert "hurs_mean" in report[BIOCLIM_VERSION]["unchanged_predictors"]


def test_projectable_predictors_reach_the_chelsa_future(settings, climate_root):
    p = trained_on_chelsa(
        settings,
        climate_root,
        {"mode": "custom", "predictors": ["bio1", "bio12", "gdd5"], "vif_screen": False},
    )
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    assert mv.metrics["predictors"] == ["bio1", "bio12", "gdd5"]
    assert FUTURE_NAME in mv.artifacts["scenarios"]
    assert not mv.artifacts.get("scenarios_unavailable")


def test_predictor_set_comparison_ranks_sets_on_the_training_data(settings, climate_root):
    p = trained_on_chelsa(settings, climate_root, None)
    out = p.compare_predictor_sets(TAXON_KEY)
    report = p.repo.get_model_version(TAXON_KEY, 1).artifacts["predictor_comparison"]
    assert out["best"] == report["best"] and report["bioclim_version"] == CHELSA
    names = [s["name"] for s in report["sets"]]
    assert {"current", "temperature", "precipitation", "climate_humidity_wind"} <= set(names)
    fitted = [s for s in report["sets"] if s["status"] == "fitted"]
    assert [s["rank"] for s in fitted] == list(range(1, len(fitted) + 1))
    cbis = [s["cbi_mean"] for s in fitted]
    assert cbis == sorted(cbis, reverse=True)
    humid = next(s for s in fitted if s["name"] == "climate_humidity_wind")
    assert humid["present_only"] == ["hurs_mean"]
    assert p.artifacts.local_path(report["path"]).exists()
    # Custom sets are accepted too.
    p.compare_predictor_sets(
        TAXON_KEY, [{"name": "mine", "label": "Mine", "predictors": ["bio1", "gdd5"]}]
    )
    report = p.repo.get_model_version(TAXON_KEY, 1).artifacts["predictor_comparison"]
    assert [s["name"] for s in report["sets"]] == ["mine"]


# --------------------------------------------------------------------------- API
@pytest.fixture
def api(settings, climate_root):
    from src.api import deps
    from src.api.main import create_app
    from src.persistence.repository import InMemoryRepository
    from tests.test_api import RecordingQueue

    s = chelsa_settings(settings, climate_root)
    repo, queue = InMemoryRepository(), RecordingQueue()
    app = create_app()
    app.dependency_overrides[deps.settings_dep] = lambda: s
    app.dependency_overrides[deps.repository_dep] = lambda: repo
    app.dependency_overrides[deps.queue_dep] = lambda: queue
    p = make_pipeline(s, FakeGbif(initial=initial_records()), repo=repo)
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    with TestClient(app) as client:
        yield client, repo, queue


def test_api_lists_climate_versions_with_grouped_predictors(api):
    client, _, _ = api
    versions = {v["version"]: v for v in client.get("/bioclim/versions").json()}
    assert versions[BIOCLIM_VERSION]["is_default"]
    chelsa = versions[CHELSA]
    assert chelsa["family"] == "chelsa" and chelsa["baseline_period"] == "1981-2010"
    assert chelsa["label"].startswith("CHELSA V2.1 (1981-2010")
    assert chelsa["hires_resolutions"] == ["30s"] and chelsa["scenarios"] == [FUTURE_NAME]
    by_name = {p["name"]: p for p in chelsa["predictors"]}
    assert by_name["hurs_mean"]["group"] == "humidity"
    assert by_name["hurs_mean"]["future_available"] is False
    assert by_name["gdd5"]["future_available"] is True
    assert (
        by_name["bio1"]["group"] == "temperature" and by_name["bio12"]["group"] == "precipitation"
    )


def test_api_environment_update_validates_and_flags_retrain(api):
    client, repo, _ = api
    url = f"/species/{TAXON_KEY}/environment"
    r = client.put(url, json={"climate_version": "nope"})
    assert r.status_code == 422 and "not installed" in r.json()["detail"]
    bad = {"mode": "custom", "predictors": ["bio1", "hurs_mean"]}
    r = client.put(url, json={"predictor_config": bad})  # WorldClim has no humidity
    assert r.status_code == 422 and "hurs_mean" in r.json()["detail"]
    r = client.put(
        url,
        json={
            "climate_version": CHELSA,
            "predictor_config": {"mode": "custom", "predictors": ["bio1"]},
        },
    )
    assert r.status_code == 422
    r = client.put(url, json={"climate_version": CHELSA, "predictor_config": bad})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["climate_version"] == CHELSA and body["effective_climate_version"] == CHELSA
    assert body["predictor_config"]["predictors"] == ["bio1", "hurs_mean"]
    assert body["status"] == "retrain_pending"
    assert repo.get_species(TAXON_KEY).predictor_config["mode"] == "custom"
    # Back to the defaults.
    body = client.put(url, json={}).json()
    assert body["climate_version"] is None and body["predictor_config"]["mode"] == "auto"
    assert body["effective_climate_version"] == BIOCLIM_VERSION


def test_api_enqueues_predictor_comparison(api):
    client, _, queue = api
    r = client.post(f"/species/{TAXON_KEY}/predictor-comparison", json={})
    assert r.status_code == 202 and r.json()["kind"] == "predictor_sets"
    r = client.post(
        f"/species/{TAXON_KEY}/predictor-comparison",
        json={"sets": [{"name": "t", "predictors": ["bio1", "bio4"]}]},
    )
    assert r.status_code == 202
    assert queue.calls[-1][3] == [{"name": "t", "label": None, "predictors": ["bio1", "bio4"]}]
