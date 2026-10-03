"""Feature-extraction tests: raster sampling, thinning, background sampling, VIF."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.features.background_sampler import (
    EARTH_RADIUS_KM,
    buffer_region,
    cell_dedupe,
    distance_thin,
    sample_buffered_background,
    sample_target_group_background,
)
from src.features.collinearity import select_low_collinearity, variance_inflation_factors
from src.features.raster_sampler import BioclimStack, cell_area_km2
from tests.conftest import BIOCLIM_VERSION, is_land, synthetic_bands


@pytest.fixture
def stack(data_root) -> BioclimStack:
    return BioclimStack.open_version(data_root / "bioclim", BIOCLIM_VERSION)


def test_sample_matches_cell_values(stack):
    lon = np.array([10.2, -80.7, 100.0])
    lat = np.array([45.3, 40.1, 20.9])
    df = stack.sample(lon, lat, ["bio1", "bio12"])
    # Expected values come from the cell centre the points fall in.
    cx, cy = np.floor(lon) + 0.5, np.floor(lat) + 0.5
    expected = synthetic_bands(cx, cy)
    np.testing.assert_allclose(df["bio1"], expected[0], rtol=1e-5)
    np.testing.assert_allclose(df["bio12"], expected[11], rtol=1e-5)


def test_sample_ocean_and_outside_grid_are_nan(stack):
    df = stack.sample([-160.0, 500.0], [0.0, 0.0], ["bio1"])
    assert df["bio1"].isna().all()


def test_sample_empty(stack):
    assert stack.sample([], [], ["bio1"]).empty


def test_iter_blocks_covers_grid(stack):
    rows = 0
    for block in stack.iter_blocks(["bio1"], block_rows=50):
        assert block.data.shape == (1, block.height, stack.width)
        rows += block.height
    assert rows == stack.height


def test_cell_area_decreases_with_latitude(stack):
    areas = cell_area_km2(stack.transform, np.array([90, 0]))  # row 90 ≈ equator, row 0 ≈ pole
    assert areas[0] > areas[1]
    assert 12_000 < areas[0] < 12_500  # 1°×1° at the equator ≈ 12,390 km²


def test_cell_dedupe_one_per_cell(stack):
    idx = cell_dedupe([10.1, 10.9, 11.5], [45.1, 45.8, 45.2], stack.transform)
    assert list(idx) == [0, 2]


def test_distance_thin_enforces_min_distance_and_is_seeded():
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(0, 5, 400), rng.uniform(40, 45, 400)
    keep = distance_thin(lon, lat, 50, seed=3)
    assert keep.size < 400
    assert np.array_equal(keep, distance_thin(lon, lat, 50, seed=3))
    coords = np.deg2rad(np.c_[lat[keep], lon[keep]])
    from sklearn.metrics.pairwise import haversine_distances

    d = haversine_distances(coords) * EARTH_RADIUS_KM
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 50


def test_buffered_background_is_on_land_inside_buffer_and_reproducible(stack):
    plon, plat = np.array([10.0, 20.0]), np.array([45.0, 50.0])
    a = sample_buffered_background(plon, plat, stack, 300, 500, seed=9, bands=["bio1", "bio12"])
    b = sample_buffered_background(plon, plat, stack, 300, 500, seed=9, bands=["bio1", "bio12"])
    assert len(a.lon) == 300
    np.testing.assert_array_equal(a.lon, b.lon)
    assert is_land(a.lon, a.lat).all()
    assert not a.features.isna().any().any()
    region = buffer_region(plon, plat, 500)
    import shapely

    assert shapely.contains_xy(region, a.lon, a.lat).all()


def test_target_group_background(stack):
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(0, 30, 500), rng.uniform(35, 60, 500)
    bg = sample_target_group_background(lon, lat, stack, 100, seed=1, bands=["bio1"])
    assert 0 < len(bg.lon) <= 100 and bg.method == "target_group"


def test_vif_drops_collinear_predictor():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=2000), rng.normal(size=2000)
    df = pd.DataFrame({"a": a, "b": b, "c": a * 2 + b + rng.normal(0, 0.01, 2000)})
    assert variance_inflation_factors(df).max() > 100
    kept, vif = select_low_collinearity(df)
    assert len(kept) == 2 and max(vif.values()) < 5


def test_vif_handles_exact_linear_dependency():
    """BIO7 = BIO5 - BIO6 exactly in WorldClim: VIF must be inf, never < 1."""
    rng = np.random.default_rng(1)
    b5, b6, b1 = rng.normal(30, 5, 3000), rng.normal(-5, 8, 3000), rng.normal(size=3000)
    df = pd.DataFrame({"bio5": b5, "bio6": b6, "bio7": b5 - b6, "bio1": b1})
    vif = variance_inflation_factors(df)
    assert np.isinf(vif[["bio5", "bio6", "bio7"]]).all()
    assert vif["bio1"] == pytest.approx(1.0, abs=0.01)
    kept, final = select_low_collinearity(df)
    assert len(kept) == 3 and all(1.0 <= v < 5 for v in final.values())


def test_derive_new_bioclim_version_keeps_core_and_raster(tmp_path):
    from src.features.bioclim_store import derive
    from src.features.raster_sampler import BioclimMetadata
    from tests.conftest import build_stack

    root = tmp_path / "bioclim"
    build_stack(root)
    dst = derive(BIOCLIM_VERSION, "bioclim_test_v2", root, core=("bio1", "bio12"))
    meta = BioclimMetadata.load(dst / "metadata.json")
    assert meta.derived_from == BIOCLIM_VERSION
    assert {"bio1", "bio12"} <= set(meta.selected_predictors)
    assert meta.core_predictors == ("bio1", "bio12")
    assert (dst / "bioclim.tif").read_bytes() == (
        root / BIOCLIM_VERSION / "bioclim.tif"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        derive(BIOCLIM_VERSION, "bioclim_test_v2", root)


def test_dvc_pointer_md5_reads_the_version_directory_hash(tmp_path):
    from src.features.bioclim_store import dvc_pointer_md5

    (tmp_path / "bioclim_v2.dvc").write_text(
        "outs:\n- md5: 324497f047daf285857f2a1dbf02b8ca.dir\n  size: 122591564\n"
        "  nfiles: 4\n  hash: md5\n  path: bioclim_v2\n",
        encoding="utf-8",
    )
    assert dvc_pointer_md5(tmp_path, "bioclim_v2") == "324497f047daf285857f2a1dbf02b8ca.dir"
    assert dvc_pointer_md5(tmp_path, "bioclim_v9") is None  # not tracked by DVC


def test_subset_iterates_only_the_region_with_its_own_transform(stack):
    sub = stack.subset((10.0, 40.0, 20.0, 45.0))
    assert (sub.width, sub.height) == (10, 5)
    assert (sub.transform.c, sub.transform.f) == (10.0, 45.0)
    blocks = list(sub.iter_blocks(["bio1"], block_rows=2))
    assert [b.row_off for b in blocks] == [0, 2, 4] and blocks[0].width == 10
    data = np.concatenate([b.data[0] for b in blocks])
    lon, lat = np.meshgrid(np.arange(10.5, 20), np.arange(44.5, 40, -1))
    np.testing.assert_allclose(data, synthetic_bands(lon, lat)[0], rtol=1e-5)


def test_default_blocks_scale_with_grid_width(stack):
    from src.features.raster_sampler import BLOCK_CELLS

    blocks = list(stack.iter_blocks(["bio1"]))
    assert blocks[0].height == min(stack.height, BLOCK_CELLS // stack.width)
    assert sum(b.height for b in blocks) == stack.height


def test_build_chelsa_harmonizes_units_onto_the_target_grid_and_land_mask(tmp_path):
    from src.features.bioclim_store import build_chelsa
    from tests.conftest import build_stack, write_synthetic_chelsa

    root = tmp_path / "bioclim"
    build_stack(root)
    src = write_synthetic_chelsa(tmp_path / "chelsa_src")
    build_chelsa("chelsa_t", BIOCLIM_VERSION, root, src_dir=src)

    target = BioclimStack.open_version(root, BIOCLIM_VERSION)
    chelsa = BioclimStack.open_version(root, "chelsa_t")
    assert chelsa.metadata.aligned_to == BIOCLIM_VERSION
    assert (chelsa.width, chelsa.height, chelsa.transform) == (
        target.width,
        target.height,
        target.transform,
    )
    assert "Karger" in (chelsa.metadata.citation or "")
    rng = np.random.default_rng(3)
    lon, lat = rng.uniform(-180, 180, 4000), rng.uniform(-55, 83, 4000)
    a, b = target.sample(lon, lat), chelsa.sample(lon, lat)
    # Same land mask: CHELSA's ocean values are dropped.
    np.testing.assert_array_equal(a["bio1"].isna(), b["bio1"].isna())
    land = a["bio1"].notna()
    for band in ("bio1", "bio12"):  # Kelvin offset; plain scale
        r = np.corrcoef(a.loc[land, band], b.loc[land, band])[0, 1]
        ratio = np.median(np.abs(b.loc[land, band])) / np.median(np.abs(a.loc[land, band]))
        assert r > 0.98 and 0.9 < ratio < 1.1, band
    assert np.abs(a.loc[land, "bio1"] - b.loc[land, "bio1"]).max() < 0.6

    # BIO3 ratio→percent: each 1° cell is the average of its four 0.5° source cells (the
    # synthetic BIO3 has random coefficients per grid shape, so compare to the source grid).
    src_lon, src_lat = np.meshgrid(
        -180 + (np.arange(720) + 0.5) * 0.5, 84 - (np.arange(280) + 0.5) * 0.5
    )
    bio3 = synthetic_bands(src_lon, src_lat)[2].reshape(140, 2, 360, 2).mean(axis=(1, 3))
    rows, cols = (84 - lat[land]).astype(int), (lon[land] + 180).astype(int)
    np.testing.assert_allclose(b.loc[land, "bio3"], bio3[rows, cols], rtol=1e-3, atol=1e-2)


def test_add_hires_registers_a_finer_stack_and_rejects_coarser(tmp_path):
    from src.features.bioclim_store import add_hires
    from tests.conftest import build_stack, write_synthetic_stack_tif

    root = tmp_path / "bioclim"
    build_stack(root)
    src = write_synthetic_stack_tif(tmp_path / "fine.tif", res=0.5)
    rel = add_hires(BIOCLIM_VERSION, "30s", root, src_tif=src)
    meta = BioclimStack.open_version(root, BIOCLIM_VERSION).metadata
    assert meta.hires == {"30s": rel} and meta.finest_hires() == "30s"
    fine = BioclimStack.open_version(root, BIOCLIM_VERSION, hires="30s")
    assert fine.res == (0.5, 0.5)
    with pytest.raises(FileExistsError):
        add_hires(BIOCLIM_VERSION, "30s", root, src_tif=src)

    from src.features.bioclim_store import write_metadata

    write_metadata(root / BIOCLIM_VERSION, replace(meta, resolution="30s", hires={}))
    with pytest.raises(ValueError, match="not finer"):
        add_hires(BIOCLIM_VERSION, "2.5m", root, src_tif=src)


def test_hires_stacks_are_only_downloaded_for_worldclim(tmp_path):
    from src.features.bioclim_store import add_hires
    from tests.conftest import build_stack

    root = tmp_path / "bioclim"
    build_stack(root)  # source: "synthetic test stack"
    with pytest.raises(ValueError, match="not a WorldClim stack"):
        add_hires(BIOCLIM_VERSION, "30s", root)
