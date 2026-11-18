"""SoilGrids soil predictors as static layers of a bioclim version (`bioclim_store add-soil`).

Synthetic SoilGrids files in Interrupted Goode Homolosine (tests/conftest.py) are reprojected
onto the synthetic 1° climate grid; the resulting version is then used end to end.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features.bioclim_store import add_soil, build_chelsa, derive
from src.features.raster_sampler import METADATA_FILENAME, BioclimMetadata, BioclimStack
from tests.conftest import (
    BIOCLIM_VERSION,
    SOIL_GAP,
    TAXON_KEY,
    FakeGbif,
    add_synthetic_scenario,
    build_stack,
    write_synthetic_chelsa,
    write_synthetic_soilgrids,
    write_synthetic_stack_tif,
)
from tests.test_pipeline import confirm_native, initial_records, make_pipeline

SOIL_VERSION = "bioclim_test_soil"
PH_TOPSOIL = (5 * 6.0 + 10 * 7.0 + 15 * 8.0) / 30


@pytest.fixture(scope="module")
def soil_root(tmp_path_factory):
    """Static layer with a soil version derived from the synthetic climate stack, a CHELSA
    cross-check stack and a finer ('30s') companion stack."""
    from src.features.bioclim_store import add_hires

    root = tmp_path_factory.mktemp("soil")
    bioclim = root / "bioclim"
    build_stack(bioclim)
    (root / "native_range_polygons").mkdir()
    add_synthetic_scenario(bioclim, "warm_ssp585_2071-2100")
    add_hires(BIOCLIM_VERSION, "30s", bioclim, src_tif=write_synthetic_stack_tif(root / "f.tif"))
    build_chelsa("chelsa_t", BIOCLIM_VERSION, bioclim, src_dir=write_synthetic_chelsa(root / "c"))
    add_soil(
        BIOCLIM_VERSION,
        SOIL_VERSION,
        bioclim,
        properties=("phh2o", "sand", "clay", "soc", "bdod", "awc"),
        src_dir=write_synthetic_soilgrids(root / "soilgrids"),
    )
    return root


def soil_meta(root) -> BioclimMetadata:
    return BioclimMetadata.load(root / "bioclim" / SOIL_VERSION / METADATA_FILENAME)


# --------------------------------------------------------------------------- building
def test_add_soil_registers_static_layers_with_provenance(soil_root):
    meta = soil_meta(soil_root)
    names = ["soil_phh2o", "soil_sand", "soil_clay", "soil_soc", "soil_bdod", "soil_awc"]
    assert list(meta.static_layers) == names
    assert meta.predictor_bands == (*meta.bands, *names)
    assert meta.derived_from == BIOCLIM_VERSION
    # The climate stack, its scenario and high-resolution stack come along unchanged.
    assert "warm_ssp585_2071-2100" in meta.scenarios and "30s" in meta.hires
    for name in names:
        entry = meta.static_layers[name]
        assert entry["kind"] == "soil" and entry["static_under_scenarios"] is True
        assert len(entry["sha256"]) == 64 and "Poggio" in entry["citation"]
        assert entry["source_resolution_m"] == 5000  # 1° grid ≥ 5′ → 5 km product
        # The gap and the projection's interruptions leave some land without soil.
        assert 0.5 < entry["land_coverage"] < 1.0
    assert meta.static_layers["soil_awc"]["derived_from"] == ["wv0033", "wv1500"]
    # The VIF screen ran over climate + soil.
    assert set(meta.vif) == set(meta.selected_predictors)
    assert set(meta.selected_predictors) <= set(meta.predictor_bands)
    assert {"bio1", "bio12"} <= set(meta.selected_predictors)  # protected core


def test_source_version_is_untouched(soil_root):
    meta = BioclimMetadata.load(soil_root / "bioclim" / BIOCLIM_VERSION / METADATA_FILENAME)
    assert meta.static_layers == {}
    assert not (soil_root / "bioclim" / BIOCLIM_VERSION / "static_layers").exists()


def test_soil_values_are_depth_weighted_and_in_physical_units(soil_root):
    stack = BioclimStack.open_version(soil_root / "bioclim", SOIL_VERSION)
    lon = np.array([10.5, 20.5, -80.5, 110.5, -150.5])
    lat = np.array([45.5, 50.5, 40.5, 30.5, 0.5])  # last two: soil gap, ocean
    df = stack.sample(lon, lat, ["soil_phh2o", "bio1", "soil_sand", "soil_awc", "soil_bdod"])
    assert list(df.columns) == ["soil_phh2o", "bio1", "soil_sand", "soil_awc", "soil_bdod"]
    assert df["soil_phh2o"].iloc[:3].to_numpy() == pytest.approx(PH_TOPSOIL, abs=0.02)
    # sand = (300 + 2·lon) g/kg → %, averaged over the cell
    assert df["soil_sand"].iloc[:3].to_numpy() == pytest.approx((300 + 2 * lon[:3]) / 10, abs=1)
    # awc = (wv0033 − wv1500) / 10 = (300 + lat − 100) / 10 vol %
    assert df["soil_awc"].iloc[:3].to_numpy() == pytest.approx((200 + lat[:3]) / 10, abs=0.5)
    assert df["soil_bdod"].iloc[:3].to_numpy() == pytest.approx(
        (130 + 0.2 * lon[:3]) / 100, abs=0.02
    )
    (gx0, gx1), (gy0, gy1) = SOIL_GAP
    assert gx0 < lon[3] < gx1 and gy0 < lat[3] < gy1
    assert np.isfinite(df["bio1"].iloc[3]) and np.isnan(df["soil_sand"].iloc[3])
    assert df.iloc[4].isna().all()  # ocean: no climate, no soil


def test_blocks_and_overview_merge_climate_and_soil_in_request_order(soil_root):
    stack = BioclimStack.open_version(soil_root / "bioclim", SOIL_VERSION)
    bands = ["soil_clay", "bio12", "soil_phh2o"]
    blocks = list(stack.iter_blocks(bands, block_rows=50))
    data = np.concatenate([b.data for b in blocks], axis=1)
    assert data.shape == (3, stack.height, stack.width)
    # Same values as point sampling at cell centres.
    r, c = 40, 190  # 49.5°N, 10.5°E
    pts = stack.sample([10.5], [49.5], bands).iloc[0].to_numpy()
    assert data[:, r, c] == pytest.approx(pts, rel=1e-5)
    # Soil only on land (masked to the climate land cells).
    assert not np.any(np.isfinite(data[0]) & np.isnan(data[1]))
    arr, _ = stack.read_overview(bands, max_width=180)
    assert arr.shape == (3, 90, 180)
    sample = stack.random_valid_samples(500, bands, seed=1)
    assert list(sample.columns) == bands and not sample.isna().any().any()


def test_scenarios_and_regional_views_reuse_the_static_soil(soil_root):
    bioclim = soil_root / "bioclim"
    current = BioclimStack.open_version(bioclim, SOIL_VERSION)
    future = BioclimStack.open_version(bioclim, SOIL_VERSION, scenario="warm_ssp585_2071-2100")
    pts = ([10.5, 20.5], [45.5, 50.5])
    now, fut = current.sample(*pts, ["bio1", "soil_sand"]), future.sample(
        *pts, ["bio1", "soil_sand"]
    )
    assert (fut["bio1"] > now["bio1"]).all()  # climate changes …
    assert fut["soil_sand"].to_numpy() == pytest.approx(now["soil_sand"].to_numpy())  # … soil not
    # A regional subset reads the same soil cells as the full grid.
    sub = current.subset((5.0, 40.0, 25.0, 55.0))
    block = next(sub.iter_blocks(["soil_sand"]))
    assert block.data.shape == (1, sub.height, sub.width)
    full = next(current.iter_blocks(["soil_sand"], block_rows=current.height)).data[0]
    assert block.data[0] == pytest.approx(full[35:50, 185:205], nan_ok=True)
    # A finer view gets the version-grid soil, nearest-resampled onto its grid.
    hires = BioclimStack.open_version(bioclim, SOIL_VERSION, hires="30s")
    hb = next(hires.subset((10.0, 45.0, 11.0, 46.0)).iter_blocks(["bio1", "soil_sand"]))
    assert hb.data.shape == (2, 2, 2)
    assert np.allclose(hb.data[1], now["soil_sand"].iloc[0], atol=1e-4)


def test_unknown_predictor_is_a_clear_error(soil_root):
    stack = BioclimStack.open_version(soil_root / "bioclim", SOIL_VERSION)
    with pytest.raises(KeyError, match="neither a band nor a layer"):
        stack.sample([0.0], [0.0], ["soil_nope"])


def test_add_soil_validates_and_never_overwrites(soil_root, tmp_path):
    bioclim = soil_root / "bioclim"
    with pytest.raises(ValueError, match="Unknown soil properties"):
        add_soil(BIOCLIM_VERSION, "x_soil", bioclim, properties=("magic",))
    with pytest.raises(FileExistsError):
        add_soil(BIOCLIM_VERSION, SOIL_VERSION, bioclim, src_dir=tmp_path)
    with pytest.raises(ValueError, match="already has predictors"):
        add_soil(SOIL_VERSION, "x2_soil", bioclim, properties=("sand",), src_dir=tmp_path)
    # A failed build (missing files) leaves no half-built version behind.
    with pytest.raises(FileNotFoundError):
        add_soil(BIOCLIM_VERSION, "x3_soil", bioclim, properties=("sand",), src_dir=tmp_path)
    assert not (bioclim / "x3_soil").exists()


def test_derive_keeps_static_layers_resolvable(soil_root):
    bioclim = soil_root / "bioclim"
    derive(SOIL_VERSION, "bioclim_test_soil_d", bioclim, core=("bio1", "bio12", "soil_sand"))
    meta = BioclimMetadata.load(bioclim / "bioclim_test_soil_d" / METADATA_FILENAME)
    assert "soil_sand" in meta.selected_predictors
    stack = BioclimStack.open_version(bioclim, "bioclim_test_soil_d")
    assert np.isfinite(stack.sample([10.5], [45.5], ["soil_sand"]).iloc[0, 0])
    hires = BioclimStack.open_version(bioclim, "bioclim_test_soil_d", hires="30s")
    assert hires.path.exists()


# --------------------------------------------------------------------------- pipeline
def soil_settings(settings, soil_root, predictors):
    """Settings on the soil version with a fixed predictor set (the synthetic soil is too
    regular for the VIF screen to be a meaningful test of which layers it keeps)."""
    path = soil_root / "bioclim" / SOIL_VERSION / METADATA_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["selected_predictors"] = predictors
    path.write_text(json.dumps(raw), encoding="utf-8")
    return settings.model_copy(
        update={
            "data_root": soil_root,
            "bioclim_version": SOIL_VERSION,
            "crosscheck_bioclim_version": "chelsa_t",
        }
    )


def test_pipeline_trains_and_projects_with_soil_predictors(settings, soil_root):
    s = soil_settings(settings, soil_root, ["bio1", "bio4", "bio12", "soil_sand", "soil_clay"])
    p = make_pipeline(s, FakeGbif(initial=initial_records()))
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    mv = p.repo.get_model_version(TAXON_KEY, 1)
    assert mv is not None
    assert mv.metrics["predictors"] == ["bio1", "bio4", "bio12", "soil_sand", "soil_clay"]
    repro = mv.metrics["reproducibility"]
    assert repro["bioclim_version_used"] == SOIL_VERSION
    layers = {layer["name"]: layer for layer in repro["static_predictor_layers"]}
    assert layers["soil_sand"]["used"] and layers["soil_clay"]["used"]
    assert not layers["soil_phh2o"]["used"]
    assert (
        layers["soil_sand"]["sha256"] == soil_meta(soil_root).static_layers["soil_sand"]["sha256"]
    )
    # The hashed training snapshot holds the soil values the model was fitted on.
    training = pd.read_parquet(p.artifacts.local_path(repro["training_data"]["path"]))
    assert {"soil_sand", "soil_clay"} <= set(training.columns)
    assert training.loc[training["role"] == "presence", "soil_sand"].notna().all()
    # The occurrence store keeps caching climate only (no soil columns appear there).
    assert "soil_sand" not in p.repo.load_occurrences(TAXON_KEY).columns
    # Projection, extrapolation diagnostics and the future scenario include soil.
    assert mv.artifacts["suitability"]
    assert "warm_ssp585_2071-2100" in mv.artifacts["scenarios"]

    # Climate cross-check swaps only the climate bands; soil keeps its training values.
    p.crosscheck_climate(TAXON_KEY)
    report = p.repo.get_model_version(TAXON_KEY, 1).artifacts["crosschecks"]["chelsa_t"]
    assert {a["predictor"] for a in report["predictors"]} == {"bio1", "bio4", "bio12"}
    assert report["unchanged_predictors"] == ["soil_sand", "soil_clay"]
    assert report["suitability_rank_correlation"] > 0.8


def test_api_reports_soil_layers_and_lineage(settings, soil_root):
    from fastapi.testclient import TestClient

    from src.api import deps
    from src.api.main import create_app

    s = soil_settings(settings, soil_root, ["bio1", "bio4", "bio12", "soil_sand"])
    app = create_app()
    app.dependency_overrides[deps.settings_dep] = lambda: s
    with TestClient(app) as client:
        info = client.get("/bioclim").json()
    assert info["static_layers"][:2] == ["soil_phh2o", "soil_sand"]
    assert any("SoilGrids" in c for c in info["static_layer_citations"])
    assert info["descriptions"]["soil_sand"].startswith("Sand content")
