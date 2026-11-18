"""One-time construction of the versioned global bioclim layer (Phase 1).

Downloads WorldClim v2.1 BIO1–19 once, stacks them into a single 19-band Cloud-Optimized
GeoTIFF, records checksum + provenance in metadata.json, and precomputes the global VIF
predictor shortlist. Species runs only ever *read* this layer (via raster_sampler).

    python -m src.features.bioclim_store build --version bioclim_v1 --resolution 2.5m
    python -m src.features.bioclim_store add-scenario --version bioclim_v1 \
        --gcm ACCESS-CM2 --ssp ssp245 --period 2041-2060
    python -m src.features.bioclim_store add-hires --version bioclim_v1 --resolution 30s
    python -m src.features.bioclim_store build-chelsa --version chelsa_v1 --match bioclim_v1
    python -m src.features.bioclim_store add-soil --from bioclim_v2 --version bioclim_v3_soil

`add-hires` attaches a finer WorldClim stack to an existing version (regional high-resolution
projection of that version's models). `build-chelsa` builds an independent CHELSA v2.1 stack
in WorldClim units on the grid and land mask of `--match`, used by the climate-data
cross-check (SDM_CROSSCHECK_BIOCLIM_VERSION). `add-soil` derives a new version whose
predictors are the climate bands plus SoilGrids 2.0 topsoil properties (static layers, see
raster_sampler), with the VIF screen re-run over both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import tempfile
import zipfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.warp import reproject
from rasterio.windows import Window

from src.config import get_settings
from src.domain import (
    BIOCLIM_BANDS,
    CHELSA_EXTRA_DESCRIPTIONS,
    SOIL_DESCRIPTIONS,
    predictor_group,
)
from src.features.collinearity import select_low_collinearity
from src.features.raster_sampler import (
    BIAS_LAYER_TRANSFORMS,
    METADATA_FILENAME,
    STACK_FILENAME,
    BioclimMetadata,
    BioclimStack,
)

log = logging.getLogger(__name__)

WORLDCLIM_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"

_DVC_MD5 = re.compile(r"^\s*-?\s*md5:\s*([0-9a-f]{32}\.dir)\s*$", re.MULTILINE)


def dvc_pointer_md5(bioclim_root: Path, version: str) -> str | None:
    """DVC content hash of a version directory (`<root>/<version>.dvc`, written by
    `dvc add data/bioclim/<version>`), or None if the version is not tracked by DVC.
    `dvc pull` with this pointer restores the exact stack a model was trained on."""
    pointer = bioclim_root / f"{version}.dvc"
    if not pointer.is_file():
        return None
    m = _DVC_MD5.search(pointer.read_text(encoding="utf-8"))
    return m.group(1) if m else None


WORLDCLIM_CMIP6 = "https://geodata.ucdavis.edu/cmip6"
RESOLUTIONS = ("10m", "5m", "2.5m", "30s")
WORLDCLIM_CITATION = (
    "Fick, S.E. and R.J. Hijmans (2017). WorldClim 2: new 1km spatial resolution climate "
    "surfaces for global land areas. International Journal of Climatology 37(12): 4302-4315."
)
# Always-retained predictors in the VIF screen. Plain VIF elimination is blind to ecology and,
# on WorldClim, drops every cold-limit variable (BIO1/5/6/10/11) because they are mutually
# correlated. Protecting annual mean temperature + annual precipitation (the canonical core)
# still satisfies VIF < 5 and keeps temperature level + seasonality (with BIO4) in the model.
DEFAULT_CORE_PREDICTORS: tuple[str, ...] = ("bio1", "bio12")
COG_OPTIONS = {"compress": "deflate", "predictor": "2", "blocksize": "512", "bigtiff": "IF_SAFER"}

CHELSA_ROOT = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies"
CHELSA_BASE = f"{CHELSA_ROOT}/1981-2010/bio"
# CHELSA V2.1 CMIP6 futures (ISIMIP3b GCMs), as listed in the CHELSA bucket.
CHELSA_GCMS = ("GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL")
CHELSA_SSPS = ("ssp126", "ssp370", "ssp585")
CHELSA_FUTURE_PERIODS = ("2011-2040", "2041-2070", "2071-2100")
# A moderate default set of extra variables (one per process); see CHELSA_EXTRA_DESCRIPTIONS.
DEFAULT_CHELSA_VARIABLES = (
    "hurs_mean",
    "vpd_max",
    "sfcWind_mean",
    "cmi_mean",
    "gdd5",
    "gsl",
    "fcf",
)
CHELSA_FILE = "CHELSA_bio{i}_1981-2010_V.2.1.tif"
CHELSA_CITATION = (
    "Karger, D.N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza, R.W., "
    "Zimmermann, N.E., Linder, H.P. & Kessler, M. (2017). Climatologies at high resolution for "
    "the earth's land surface areas. Scientific Data 4, 170122. CHELSA V2.1, "
    "https://doi.org/10.16904/envidat.228"
)
CHELSA_BIOCLIM_PLUS_CITATION = (
    "Brun, P., Zimmermann, N. E., Hari, C., Pellissier, L. & Karger, D. N. (2022). Global "
    "climate-related predictors at kilometer resolution for the past and future. Earth Syst. "
    "Sci. Data 14, 5573–5603. https://doi.org/10.5194/essd-14-5573-2022"
)
# CHELSA V2.1 files declare their scale/offset (applied on read, giving °C and mm), which
# brings every band into WorldClim units except BIO3: isothermality is a ratio in CHELSA
# and a percentage in WorldClim.
CHELSA_TO_WORLDCLIM_FACTOR: dict[str, float] = {"bio3": 100.0}
_NODATA = -9999.0


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> Path:
    log.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
    return dest


def stack_to_cog(band_files: list[Path], out_path: Path, band_names: tuple[str, ...]) -> None:
    """Stack single-band GeoTIFFs (same grid) into one multiband COG."""
    with rasterio.open(band_files[0]) as first:
        profile = first.profile.copy()
    profile.update(
        count=len(band_files),
        dtype="float32",
        driver="GTiff",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_tif = Path(tmp) / "stack.tif"
        with rasterio.open(tmp_tif, "w", **profile) as dst:
            for i, f in enumerate(band_files, start=1):
                with rasterio.open(f) as src:
                    if (src.width, src.height, src.transform) != (
                        profile["width"],
                        profile["height"],
                        profile["transform"],
                    ):
                        raise ValueError(f"{f} is not aligned with {band_files[0]}")
                    for _, win in src.block_windows(1):
                        dst.write(src.read(1, window=win).astype("float32"), i, window=win)
                dst.set_band_description(i, band_names[i - 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rio_copy(tmp_tif, out_path, driver="COG", **COG_OPTIONS)


def write_metadata(root: Path, meta: BioclimMetadata) -> None:
    (root / METADATA_FILENAME).write_text(json.dumps(meta.to_json(), indent=2), encoding="utf-8")


def compute_vif_shortlist(
    root: Path,
    meta: BioclimMetadata,
    seed: int = 42,
    core: tuple[str, ...] = DEFAULT_CORE_PREDICTORS,
) -> BioclimMetadata:
    stack = BioclimStack(root / STACK_FILENAME, meta)
    sample = stack.random_valid_samples(20_000, seed=seed)
    selected, vif = select_low_collinearity(sample, keep_first=core)
    return BioclimMetadata(
        **{
            **meta.__dict__,
            "selected_predictors": tuple(selected),
            "vif": vif,
            "core_predictors": tuple(core),
        }
    )


# Sub-directories a version may carry next to its stack; all referenced from metadata.json.
VERSION_SUBDIRS = ("scenarios", "hires", "bias_layers", "static_layers", "climate_layers")


def _copy_version_files(src: Path, dst: Path) -> None:
    """The stack and every attached sub-directory of `src`, so the copied metadata entries
    (scenarios, high-resolution stacks, bias and static layers) resolve in `dst`. Real
    copies, never hard links: a later in-place write to one version must not change another."""
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / STACK_FILENAME, dst / STACK_FILENAME)
    for sub in VERSION_SUBDIRS:
        if (src / sub).exists():
            shutil.copytree(src / sub, dst / sub)


def derive(
    src_version: str,
    new_version: str,
    bioclim_root: Path,
    core: tuple[str, ...] = DEFAULT_CORE_PREDICTORS,
    seed: int = 42,
) -> Path:
    """New layer version from an existing raster with a different predictor screening.

    Versions are immutable once a model has used them, so changing the shortlist means a new
    tag; the raster is copied unchanged (identical checksum) and `derived_from` is recorded.
    """
    src = bioclim_root / src_version
    dst = bioclim_root / new_version
    if (dst / STACK_FILENAME).exists():
        raise FileExistsError(f"{dst} already exists — pick a new version tag")
    src_meta = BioclimMetadata.load(src / METADATA_FILENAME)
    _copy_version_files(src, dst)
    meta = BioclimMetadata(
        **{
            **src_meta.__dict__,
            "version": new_version,
            "created": datetime.now(UTC).isoformat(),
            "derived_from": src_version,
        }
    )
    meta = compute_vif_shortlist(dst, meta, seed=seed, core=core)
    write_metadata(dst, meta)
    log.info(
        "Derived %s from %s: predictors %s", new_version, src_version, meta.selected_predictors
    )
    return dst


def build(version: str, resolution: str, bioclim_root: Path, seed: int = 42) -> Path:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    root = bioclim_root / version
    if (root / STACK_FILENAME).exists():
        raise FileExistsError(
            f"{root} already exists — bioclim layers are immutable per version; "
            "create a new version tag instead of re-downloading."
        )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        zip_path = _download(f"{WORLDCLIM_BASE}/wc2.1_{resolution}_bio.zip", tmp_p / "bio.zip")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_p / "bio")
        files = [
            next((tmp_p / "bio").rglob(f"wc2.1_{resolution}_bio_{i}.tif")) for i in range(1, 20)
        ]
        stack_to_cog(files, root / STACK_FILENAME, BIOCLIM_BANDS)

    meta = BioclimMetadata(
        version=version,
        source=f"WorldClim v2.1 ({WORLDCLIM_BASE}/wc2.1_{resolution}_bio.zip)",
        resolution=resolution,
        crs="EPSG:4326",
        bands=BIOCLIM_BANDS,
        created=datetime.now(UTC).isoformat(),
        sha256=sha256_file(root / STACK_FILENAME),
        citation=WORLDCLIM_CITATION,
        source_family="worldclim",
        baseline_period="1970-2000",
    )
    meta = compute_vif_shortlist(root, meta, seed=seed)
    write_metadata(root, meta)
    log.info("Built %s: predictors kept after VIF<5: %s", version, meta.selected_predictors)
    return root


def add_scenario(
    version: str,
    gcm: str,
    ssp: str,
    period: str,
    bioclim_root: Path,
    source: str | None = None,
    src_dir: Path | None = None,
) -> str:
    """Register a CMIP6 future stack (same grid) for the 'what-if' scenario toggle, from the
    version's own climate source: WorldClim CMIP6 for WorldClim versions, CHELSA CMIP6 for
    CHELSA versions. Mixing sources is refused: a CHELSA future on a WorldClim baseline (or the
    reverse) would turn differences between the datasets into apparent climate change."""
    root = bioclim_root / version
    meta = BioclimMetadata.load(root / METADATA_FILENAME)
    family = meta.family
    source = source or ("chelsa" if family == "chelsa" else "worldclim")
    if source == "chelsa" and family != "chelsa":
        raise ValueError(
            f"CHELSA futures need a CHELSA version; {version} is a {family} version "
            "(its scenarios must come from its own climate source)"
        )
    if source == "worldclim" and family == "chelsa":
        raise ValueError(
            f"{version} is a CHELSA version; its scenarios must come from CHELSA too "
            "(got worldclim)"
        )
    if source == "chelsa":
        return add_chelsa_scenario(version, gcm, ssp, period, bioclim_root, src_dir)
    name = f"{gcm}_{ssp}_{period}"
    fname = f"wc2.1_{meta.resolution}_bioc_{gcm}_{ssp}_{period}.tif"
    rel = f"scenarios/{name}.tif"
    out = root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = _download(
            f"{WORLDCLIM_CMIP6}/{meta.resolution}/{gcm}/{ssp}/{fname}", Path(tmp) / fname
        )
        rio_copy(raw, out, driver="COG", **COG_OPTIONS)
    scenarios = {**meta.scenarios, name: rel}
    write_metadata(root, BioclimMetadata(**{**meta.__dict__, "scenarios": scenarios}))
    return name


def install_local_stack(
    src_tif: Path, version: str, bioclim_root: Path, source: str, resolution: str, seed: int = 42
) -> Path:
    """Register an existing 19-band GeoTIFF (e.g. CHELSA or an offline mirror) as a version."""
    root = bioclim_root / version
    if (root / STACK_FILENAME).exists():
        raise FileExistsError(
            f"{root} already exists — bioclim layers are immutable per version; "
            "install under a new version tag."
        )
    root.mkdir(parents=True, exist_ok=True)
    rio_copy(src_tif, root / STACK_FILENAME, driver="COG", **COG_OPTIONS)
    meta = BioclimMetadata(
        version=version,
        source=source,
        resolution=resolution,
        crs="EPSG:4326",
        bands=BIOCLIM_BANDS,
        created=datetime.now(UTC).isoformat(),
        sha256=sha256_file(root / STACK_FILENAME),
    )
    write_metadata(root, compute_vif_shortlist(root, meta, seed=seed))
    return root


def add_hires(
    version: str, resolution: str, bioclim_root: Path, src_tif: Path | None = None
) -> str:
    """Attach a finer stack of the *same* source (WorldClim) to an existing version, so its
    models can be projected at high resolution over a region without retraining."""
    root = bioclim_root / version
    meta = BioclimMetadata.load(root / METADATA_FILENAME)
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    if meta.resolution in RESOLUTIONS and RESOLUTIONS.index(resolution) <= RESOLUTIONS.index(
        meta.resolution
    ):
        raise ValueError(f"{resolution} is not finer than {version}'s {meta.resolution}")
    if src_tif is None and not meta.source.startswith("WorldClim"):
        raise ValueError(
            f"{version} is not a WorldClim stack: pass --src with a {resolution} stack of the "
            "same source, or predictors would change meaning at high resolution"
        )
    rel = f"hires/bioclim_{resolution}.tif"
    out = root / rel
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    if src_tif is not None:
        with rasterio.open(src_tif) as src:
            if src.count != len(BIOCLIM_BANDS):
                raise ValueError(f"{src_tif} has {src.count} bands, expected 19 (BIO1–19)")
        rio_copy(src_tif, out, driver="COG", **COG_OPTIONS)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            zip_path = _download(f"{WORLDCLIM_BASE}/wc2.1_{resolution}_bio.zip", tmp_p / "bio.zip")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_p / "bio")
            files = [
                next((tmp_p / "bio").rglob(f"wc2.1_{resolution}_bio_{i}.tif")) for i in range(1, 20)
            ]
            stack_to_cog(files, out, BIOCLIM_BANDS)
    write_metadata(
        root, BioclimMetadata(**{**meta.__dict__, "hires": {**meta.hires, resolution: rel}})
    )
    log.info("Registered %s high-resolution stack for %s", resolution, version)
    return rel


def chelsa_present_file(var: str) -> str:
    """CHELSA V2.1 1981–2010 file name of a BIO band (`bio3`) or extra variable."""
    if var in BIOCLIM_BANDS:
        return CHELSA_FILE.format(i=int(var[3:]))
    if var.startswith("rsds_"):  # the one family with the statistic after the period
        return f"CHELSA_rsds_1981-2010_{var.split('_', 1)[1]}_V.2.1.tif"
    return f"CHELSA_{var}_1981-2010_V.2.1.tif"


def chelsa_future_file(var: str, gcm: str, ssp: str, period: str) -> str:
    """CHELSA V2.1 CMIP6 file name: GCM upper-case in the folder, lower-case in the name."""
    return f"CHELSA_{var}_{period}_{gcm.lower()}_{ssp}_V.2.1.tif"


def chelsa_future_url(var: str, gcm: str, ssp: str, period: str) -> str:
    return f"{CHELSA_ROOT}/{period}/{gcm}/{ssp}/bio/{chelsa_future_file(var, gcm, ssp, period)}"


def _fetch_optional(url: str, fname: str, tmp: Path, src_dir: Path | None) -> Path | None:
    """A file from `src_dir`, or downloaded; None when it does not exist (HTTP 404)."""
    if src_dir is not None:
        path = src_dir / fname
        return path if path.exists() else None
    try:
        return _download(url, tmp / fname)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise


def _fetch_required(url: str, fname: str, tmp: Path, src_dir: Path | None) -> Path:
    path = _fetch_optional(url, fname, tmp, src_dir)
    if path is None:
        raise FileNotFoundError(f"{fname} not found ({src_dir or url})")
    return path


def _grid_profile(target: BioclimStack, count: int) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "width": target.width,
        "height": target.height,
        "count": count,
        "dtype": "float32",
        "crs": target.crs,
        "transform": target.transform,
        "nodata": _NODATA,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }


def _write_chelsa_stack(
    band_files: list[tuple[str, Path]],
    targets: list[tuple[BioclimStack, Path]],
    tmp: Path,
    cleanup: bool,
) -> None:
    """BIO1–19 CHELSA files → one 19-band COG per target grid (each source file read once)."""
    with ExitStack() as es:
        writers = []
        for k, (target, _) in enumerate(targets):
            path = tmp / f"stack_{k}.tif"
            writers.append(
                (path, es.enter_context(rasterio.open(path, "w", **_grid_profile(target, 19))))
            )
        for i, (band, src_path) in enumerate(band_files, start=1):
            for (target, _), (_, dst) in zip(targets, writers, strict=True):
                _resample_chelsa_band(src_path, band, target, dst, i)
                dst.set_band_description(i, band)
            if cleanup:
                src_path.unlink(missing_ok=True)
    for (_, out), (path, _) in zip(targets, writers, strict=True):
        out.parent.mkdir(parents=True, exist_ok=True)
        rio_copy(path, out, driver="COG", **COG_OPTIONS)


def _write_chelsa_layer(
    var: str, src_path: Path, targets: list[tuple[BioclimStack, Path]], tmp: Path
) -> None:
    """One CHELSA variable → a single-band COG per target grid, masked to its land cells."""
    for k, (target, out) in enumerate(targets):
        path = tmp / f"{var}_{k}.tif"
        with rasterio.open(path, "w", **_grid_profile(target, 1)) as dst:
            _resample_chelsa_band(src_path, var, target, dst, 1)
            dst.set_band_description(1, var)
        out.parent.mkdir(parents=True, exist_ok=True)
        rio_copy(path, out, driver="COG", **COG_OPTIONS)


def build_chelsa(
    version: str,
    match_version: str,
    bioclim_root: Path,
    seed: int = 42,
    src_dir: Path | None = None,
    variables: tuple[str, ...] = (),
    with_hires: bool = False,
) -> Path:
    """CHELSA V2.1 (1981–2010) version on the grid and land mask of `match_version`.

    * BIO1–19 in WorldClim units (each file's declared scale/offset applied, BIO3 as a
      percentage), average-resampled, as the stack. Usable for training (select it per species)
      and as the climate-data cross-check stack.
    * `variables`: extra CHELSA-BIOCLIM+ variables (e.g. `hurs_mean`, `vpd_max`,
      `sfcWind_mean`, `cmi_mean`, `gdd5`) as climate layers. Future data only exists for
      some of them (growing season, frost, snow, NPP); the others stay present-only.
    * `with_hires`: also build every high-resolution grid of `match_version` (e.g. 30″, the
      finest CHELSA offers) from the same 30″ source files, for regional models.

    Files are downloaded one at a time (≈5 GB for BIO1–19, 0.2–1 GB per extra variable) and
    deleted after resampling, unless `src_dir` holds them already.
    """
    unknown = [v for v in variables if v not in CHELSA_EXTRA_DESCRIPTIONS]
    if unknown:
        raise ValueError(
            f"Unknown CHELSA variables {unknown}; choose from "
            f"{', '.join(CHELSA_EXTRA_DESCRIPTIONS)}"
        )
    root = bioclim_root / version
    if (root / STACK_FILENAME).exists():
        raise FileExistsError(f"{root} already exists — pick a new version tag")
    match_meta = BioclimMetadata.load(bioclim_root / match_version / METADATA_FILENAME)
    if with_hires and not match_meta.hires:
        raise ValueError(f"{match_version} has no high-resolution grid to match (add-hires)")
    target = BioclimStack.open_version(bioclim_root, match_version)
    hires = list(match_meta.hires) if with_hires else []
    grids: list[tuple[str | None, BioclimStack]] = [
        (None, target),
        *((r, BioclimStack.open_version(bioclim_root, match_version, hires=r)) for r in hires),
    ]
    root.mkdir(parents=True, exist_ok=True)
    cleanup = src_dir is None
    climate_layers: dict[str, dict[str, Any]] = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            bio_files = [
                (
                    band,
                    _fetch_required(
                        f"{CHELSA_BASE}/{chelsa_present_file(band)}",
                        chelsa_present_file(band),
                        tmp_p,
                        src_dir,
                    ),
                )
                for band in BIOCLIM_BANDS
            ]
            _write_chelsa_stack(
                bio_files,
                [
                    (g, root / (STACK_FILENAME if r is None else f"hires/bioclim_{r}.tif"))
                    for r, g in grids
                ],
                tmp_p,
                cleanup,
            )
            for var in variables:
                fname = chelsa_present_file(var)
                src_path = _fetch_required(f"{CHELSA_BASE}/{fname}", fname, tmp_p, src_dir)
                rels = {
                    r: (
                        f"climate_layers/{var}.tif"
                        if r is None
                        else f"climate_layers/hires_{r}/{var}.tif"
                    )
                    for r, _ in grids
                }
                _write_chelsa_layer(var, src_path, [(g, root / rels[r]) for r, g in grids], tmp_p)
                if cleanup:
                    src_path.unlink(missing_ok=True)
                climate_layers[var] = {
                    "path": rels[None],
                    "hires": {r: rel for r, rel in rels.items() if r is not None},
                    "scenarios": {},
                    "kind": "climate",
                    "group": predictor_group(var),
                    "description": CHELSA_EXTRA_DESCRIPTIONS[var],
                    "source": f"CHELSA V2.1 1981–2010 {fname}, average-resampled to the "
                    f"{match_version} grid",
                    "citation": CHELSA_BIOCLIM_PLUS_CITATION,
                    "sha256": sha256_file(root / rels[None]),
                }
        meta = BioclimMetadata(
            version=version,
            source=(
                f"CHELSA V2.1 1981–2010 ({CHELSA_BASE}), average-resampled to the "
                f"{match_version} grid and land mask, converted to WorldClim units"
            ),
            resolution=target.metadata.resolution,
            crs="EPSG:4326",
            bands=BIOCLIM_BANDS,
            created=datetime.now(UTC).isoformat(),
            sha256=sha256_file(root / STACK_FILENAME),
            citation=CHELSA_CITATION,
            aligned_to=match_version,
            hires={r: f"hires/bioclim_{r}.tif" for r in hires},
            climate_layers=climate_layers,
            source_family="chelsa",
            baseline_period="1981-2010",
        )
        meta = compute_vif_shortlist(root, meta, seed=seed)
        write_metadata(root, meta)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    log.info(
        "Built CHELSA stack %s on the %s grid (extra variables: %s; high-resolution: %s)",
        version,
        match_version,
        list(variables) or "none",
        hires or "none",
    )
    return root


def add_chelsa_scenario(
    version: str,
    gcm: str,
    ssp: str,
    period: str,
    bioclim_root: Path,
    src_dir: Path | None = None,
) -> str:
    """Register a CHELSA V2.1 CMIP6 future for a CHELSA version: BIO1–19 (same units and
    resampling as the baseline) plus every climate layer CHELSA publishes for that future.
    Layers without a future file (humidity, VPD, wind, PET, moisture index, radiation) stay
    present-only for this scenario and are listed in the log and the metadata."""
    if gcm not in CHELSA_GCMS:
        raise ValueError(f"CHELSA CMIP6 GCMs: {', '.join(CHELSA_GCMS)}")
    if ssp not in CHELSA_SSPS:
        raise ValueError(f"CHELSA CMIP6 SSPs: {', '.join(CHELSA_SSPS)}")
    if period not in CHELSA_FUTURE_PERIODS:
        raise ValueError(f"CHELSA CMIP6 periods: {', '.join(CHELSA_FUTURE_PERIODS)}")
    root = bioclim_root / version
    meta = BioclimMetadata.load(root / METADATA_FILENAME)
    target = BioclimStack(root / STACK_FILENAME, meta, root=root)
    name = f"{gcm}_{ssp}_{period}"
    rel = f"scenarios/{name}.tif"
    cleanup = src_dir is None
    layers = {k: dict(v) for k, v in meta.climate_layers.items()}
    present_only: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        bio_files = [
            (
                band,
                _fetch_required(
                    chelsa_future_url(band, gcm, ssp, period),
                    chelsa_future_file(band, gcm, ssp, period),
                    tmp_p,
                    src_dir,
                ),
            )
            for band in BIOCLIM_BANDS
        ]
        _write_chelsa_stack(bio_files, [(target, root / rel)], tmp_p, cleanup)
        for var, entry in layers.items():
            fname = chelsa_future_file(var, gcm, ssp, period)
            src_path = _fetch_optional(
                chelsa_future_url(var, gcm, ssp, period), fname, tmp_p, src_dir
            )
            if src_path is None:
                present_only.append(var)
                continue
            layer_rel = f"scenarios/{name}/{var}.tif"
            _write_chelsa_layer(var, src_path, [(target, root / layer_rel)], tmp_p)
            if cleanup:
                src_path.unlink(missing_ok=True)
            entry["scenarios"] = {**(entry.get("scenarios") or {}), name: layer_rel}
    write_metadata(
        root,
        BioclimMetadata(
            **{
                **meta.__dict__,
                "scenarios": {**meta.scenarios, name: rel},
                "climate_layers": layers,
            }
        ),
    )
    if present_only:
        log.warning(
            "%s: no CHELSA future data for %s — present-only for this scenario", name, present_only
        )
    return name


# Defaults for common observer-bias rasters (the file itself must be supplied: the providers
# have no stable direct-download URL). Values are resampled (average) onto the version grid.
BIAS_LAYER_PRESETS: dict[str, dict[str, str]] = {
    "travel_time": {
        "direction": "decreasing",
        "transform": "log1p",
        "citation": "Weiss, D. J. et al. (2018). A global map of travel time to cities to "
        "assess inequalities in accessibility in 2015. Nature 553, 333–336.",
    },
    "road_distance": {
        "direction": "decreasing",
        "transform": "log1p",
        "citation": "Meijer, J. R. et al. (2018). Global patterns of current and future road "
        "infrastructure. Environ. Res. Lett. 13, 064006 (GRIP4).",
    },
    "population_density": {
        "direction": "increasing",
        "transform": "log1p",
        "citation": "CIESIN (2018). Gridded Population of the World, Version 4.11.",
    },
}


def add_bias_layer(
    version: str,
    name: str,
    src: Path,
    bioclim_root: Path,
    direction: str | None = None,
    transform: str | None = None,
    citation: str | None = None,
) -> dict[str, Any]:
    """Register a static observer-bias raster (B6) on a version's grid and land mask."""
    preset = BIAS_LAYER_PRESETS.get(name, {})
    direction = direction or preset.get("direction")
    transform = transform or preset.get("transform", "none")
    if direction not in ("increasing", "decreasing"):
        raise ValueError("direction must be 'increasing' or 'decreasing' (more / less effort)")
    if transform not in BIAS_LAYER_TRANSFORMS:
        raise ValueError(f"transform must be one of {BIAS_LAYER_TRANSFORMS}")
    root = bioclim_root / version
    meta = BioclimMetadata.load(root / METADATA_FILENAME)
    target = BioclimStack.open_version(bioclim_root, version)
    rel = f"bias_layers/{name}.tif"
    out = root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": target.width,
        "height": target.height,
        "count": 1,
        "dtype": "float32",
        "crs": target.crs,
        "transform": target.transform,
        "nodata": _NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }
    with tempfile.TemporaryDirectory() as tmp, rasterio.open(src) as source:
        tmp_path = Path(tmp) / f"{name}.tif"
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for block in target.iter_blocks(["bio1"]):
                raw = np.full((block.height, block.width), np.nan, dtype="float32")
                reproject(
                    source=rasterio.band(source, 1),
                    destination=raw,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=block.transform,
                    dst_crs=target.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                )
                raw[np.isnan(block.data[0])] = np.nan
                dst.write(
                    np.where(np.isnan(raw), _NODATA, raw).astype("float32"),
                    1,
                    window=Window(0, block.row_off, block.width, block.height),
                )
        rio_copy(tmp_path, out, driver="COG", **COG_OPTIONS)
    entry = {
        "path": rel,
        "direction": direction,
        "transform": transform,
        "citation": citation or preset.get("citation"),
        "source": src.name,
        "sha256": sha256_file(out),
        "added": datetime.now(UTC).isoformat(),
    }
    layers = {**meta.bias_layers, name: entry}
    write_metadata(root, BioclimMetadata(**{**meta.__dict__, "bias_layers": layers}))
    log.info("Registered bias layer %s on %s", name, version)
    return entry


# ---------------------------------------------------------------------------
# SoilGrids 2.0 soil predictors (static layers)
# ---------------------------------------------------------------------------
SOILGRIDS_BASE = "https://files.isric.org/soilgrids/latest/data_aggregated"
SOILGRIDS_FILE = "{prop}_{depth}_mean_{res}.tif"
SOILGRIDS_RESOLUTIONS = ("1000", "5000")
SOILGRIDS_CITATION = (
    "Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E. "
    "& Rossiter, D. (2021). SoilGrids 2.0: producing soil information for the globe with "
    "quantified spatial uncertainty. SOIL 7, 217–240. https://doi.org/10.5194/soil-7-217-2021 "
    "(CC BY 4.0)"
)
SOILGRIDS_NODATA = -32768.0
# Mapped integer units → conventional units (ISRIC SoilGrids conversion factors).
SOILGRIDS_PROPERTIES: dict[str, dict[str, Any]] = {
    "phh2o": {"factor": 0.1, "units": "pH"},
    "sand": {"factor": 0.1, "units": "%"},
    "clay": {"factor": 0.1, "units": "%"},
    "silt": {"factor": 0.1, "units": "%"},
    "soc": {"factor": 0.1, "units": "g/kg"},
    "bdod": {"factor": 0.01, "units": "kg/dm³"},
    "cfvo": {"factor": 0.1, "units": "vol %"},
    "cec": {"factor": 0.1, "units": "cmol(c)/kg"},
    "nitrogen": {"factor": 0.01, "units": "g/kg"},
    "wv0033": {"factor": 0.1, "units": "vol %"},
    "wv1500": {"factor": 0.1, "units": "vol %"},
}
# Derived properties: available water capacity = field capacity (33 kPa) − wilting point
# (1500 kPa) water content.
SOIL_DERIVED: dict[str, dict[str, Any]] = {
    "awc": {"from": ("wv0033", "wv1500"), "units": "vol %"},
}
# Default set (reply.md §3.1): texture (sand, clay; silt is their complement and left out to
# avoid compositional collinearity), pH, organic carbon, bulk density, water-holding capacity.
DEFAULT_SOIL_PROPERTIES: tuple[str, ...] = ("phh2o", "sand", "clay", "soc", "bdod", "awc")
# Topsoil 0–30 cm as the thickness-weighted mean of the three upper standard depths.
TOPSOIL_DEPTHS: tuple[tuple[str, float], ...] = (
    ("0-5cm", 5.0),
    ("5-15cm", 10.0),
    ("15-30cm", 15.0),
)
SOIL_PREFIX = "soil_"


def default_soilgrids_resolution(cell_deg: float) -> str:
    """5 km source data for grids of 5′ and coarser, 1 km otherwise (finer than the cell, so
    average resampling aggregates it)."""
    return "5000" if cell_deg >= (1 / 12) - 1e-9 else "1000"


def _soilgrids_file(prop: str, depth: str, res: str, tmp: Path, src_dir: Path | None) -> Path:
    fname = SOILGRIDS_FILE.format(prop=prop, depth=depth, res=res)
    if src_dir is not None:
        path = src_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (expected SoilGrids file {fname})")
        return path
    return _download(f"{SOILGRIDS_BASE}/{res}m/{prop}/{fname}", tmp / fname)


def _reproject_block(
    src: rasterio.io.DatasetReader, block: Any, crs: Any, factor: float
) -> np.ndarray:
    """One SoilGrids file (any CRS, e.g. Homolosine) averaged onto a target block, in
    conventional units; NaN where SoilGrids has no prediction."""
    raw = np.full((block.height, block.width), np.nan, dtype="float32")
    reproject(
        source=rasterio.band(src, 1),
        destination=raw,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata if src.nodata is not None else SOILGRIDS_NODATA,
        dst_transform=block.transform,
        dst_crs=crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return raw * factor


def _topsoil_property(
    prop: str, block: Any, crs: Any, files: dict[tuple[str, str], Path]
) -> np.ndarray:
    """Thickness-weighted 0–30 cm mean of one SoilGrids property on a block (depths with no
    value are left out of the weighting; NaN when no depth has a value)."""
    factor = SOILGRIDS_PROPERTIES[prop]["factor"]
    num = np.zeros((block.height, block.width), dtype="float64")
    den = np.zeros_like(num)
    for depth, weight in TOPSOIL_DEPTHS:
        with rasterio.open(files[(prop, depth)]) as src:
            v = _reproject_block(src, block, crs, factor)
        ok = np.isfinite(v)
        num[ok] += weight * v[ok]
        den[ok] += weight
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan).astype("float32")


def _soil_layer_values(
    prop: str, block: Any, crs: Any, files: dict[tuple[str, str], Path]
) -> np.ndarray:
    """Topsoil values of a SoilGrids property or a derived one (e.g. awc) on a block."""
    if prop in SOIL_DERIVED:
        a, b = SOIL_DERIVED[prop]["from"]
        return _topsoil_property(a, block, crs, files) - _topsoil_property(b, block, crs, files)
    return _topsoil_property(prop, block, crs, files)


def _write_soil_layer(
    prop: str, target: BioclimStack, files: dict[tuple[str, str], Path], out: Path
) -> dict[str, Any]:
    """Build one soil layer on `target`'s grid, masked to its land cells (BIO1)."""
    profile = {
        "driver": "GTiff",
        "width": target.width,
        "height": target.height,
        "count": 1,
        "dtype": "float32",
        "crs": target.crs,
        "transform": target.transform,
        "nodata": _NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }
    n_land = n_covered = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / out.name
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for block in target.iter_blocks(["bio1"]):
                vals = _soil_layer_values(prop, block, target.crs, files)
                land = ~np.isnan(block.data[0])
                vals[~land] = np.nan
                n_land += int(land.sum())
                n_covered += int((land & np.isfinite(vals)).sum())
                dst.write(
                    np.where(np.isnan(vals), _NODATA, vals).astype("float32"),
                    1,
                    window=Window(0, block.row_off, block.width, block.height),
                )
        out.parent.mkdir(parents=True, exist_ok=True)
        rio_copy(tmp_path, out, driver="COG", **COG_OPTIONS)
    return {
        "n_land_cells": n_land,
        # Land cells without a SoilGrids prediction (water bodies, glaciers, some urban and
        # bare-rock areas): occurrences there drop out of training, and projection is
        # nodata there.
        "land_coverage": round(n_covered / n_land, 4) if n_land else None,
    }


def add_soil(
    src_version: str,
    new_version: str,
    bioclim_root: Path,
    properties: tuple[str, ...] = DEFAULT_SOIL_PROPERTIES,
    source_resolution: str | None = None,
    src_dir: Path | None = None,
    core: tuple[str, ...] = DEFAULT_CORE_PREDICTORS,
    seed: int = 42,
    with_hires: bool = False,
) -> Path:
    """New version = `src_version` (stack, scenarios, high-resolution stacks, bias layers,
    copied unchanged) + SoilGrids 2.0 topsoil (0–30 cm) properties as static predictor
    layers `soil_<property>`, with the VIF shortlist recomputed over climate + soil.

    Soil is static: future scenarios of the version reuse the same layers. SoilGrids is a
    model product with its own uncertainty, and both points are recorded in the metadata.
    `src_dir` holds pre-downloaded files named as on files.isric.org
    (`<prop>_<depth>_mean_<res>.tif`); otherwise each file is downloaded once and deleted.
    With `with_hires`, every registered high-resolution stack also gets its own 1 km soil
    layers (otherwise high-resolution views use the version-grid soil, nearest-resampled).
    """
    unknown = [p for p in properties if p not in SOILGRIDS_PROPERTIES and p not in SOIL_DERIVED]
    if unknown:
        raise ValueError(
            f"Unknown soil properties {unknown}; choose from "
            f"{', '.join([*SOILGRIDS_PROPERTIES, *SOIL_DERIVED])}"
        )
    if source_resolution is not None and source_resolution not in SOILGRIDS_RESOLUTIONS:
        raise ValueError(f"source resolution must be one of {SOILGRIDS_RESOLUTIONS} (metres)")
    src = bioclim_root / src_version
    dst = bioclim_root / new_version
    if (dst / STACK_FILENAME).exists():
        raise FileExistsError(f"{dst} already exists — pick a new version tag")
    src_meta = BioclimMetadata.load(src / METADATA_FILENAME)
    raw_props = sorted(
        {q for p in properties for q in (SOIL_DERIVED[p]["from"] if p in SOIL_DERIVED else (p,))}
    )
    names = [SOIL_PREFIX + p for p in properties]
    clash = [n for n in names if n in src_meta.bands or n in src_meta.static_layers]
    if clash:
        raise ValueError(f"{src_version} already has predictors {clash}")

    _copy_version_files(src, dst)
    try:
        # Views of the copied files (metadata.json is written once the layers exist).
        target = BioclimStack(dst / STACK_FILENAME, src_meta, root=dst)
        res = source_resolution or default_soilgrids_resolution(abs(target.transform.a))
        grids: list[tuple[str | None, BioclimStack, str]] = [(None, target, res)]
        if with_hires:
            grids += [
                (r, BioclimStack(dst / rel, src_meta, root=dst, hires=r), "1000")
                for r, rel in src_meta.hires.items()
            ]
        static: dict[str, dict[str, Any]] = {}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            for hires_res, grid, sres in grids:
                files = {
                    (p, d): _soilgrids_file(p, d, sres, tmp_p, src_dir)
                    for p in raw_props
                    for d, _ in TOPSOIL_DEPTHS
                }
                for prop, name in zip(properties, names, strict=True):
                    rel = (
                        f"static_layers/{name}.tif"
                        if hires_res is None
                        else f"static_layers/hires_{hires_res}/{name}.tif"
                    )
                    log.info("Building %s (%s m SoilGrids) for %s", rel, sres, new_version)
                    stats = _write_soil_layer(prop, grid, files, dst / rel)
                    if hires_res is None:
                        units = (SOIL_DERIVED.get(prop) or SOILGRIDS_PROPERTIES[prop])["units"]
                        static[name] = {
                            "path": rel,
                            "kind": "soil",
                            "description": SOIL_DESCRIPTIONS.get(name, name),
                            "units": units,
                            "property": prop,
                            "derived_from": (
                                list(SOIL_DERIVED[prop]["from"]) if prop in SOIL_DERIVED else None
                            ),
                            "depth": "0-30cm (thickness-weighted 0-5, 5-15, 15-30 cm)",
                            "source": f"SoilGrids 2.0 mean, {sres} m aggregated "
                            f"({SOILGRIDS_BASE}), average-resampled to the version grid",
                            "source_resolution_m": int(sres),
                            "citation": SOILGRIDS_CITATION,
                            "static_under_scenarios": True,
                            "sha256": sha256_file(dst / rel),
                            "hires": {},
                            **stats,
                        }
                    else:
                        static[name]["hires"][hires_res] = rel
                        static[name][f"sha256_hires_{hires_res}"] = sha256_file(dst / rel)
                if src_dir is None:
                    for f in files.values():
                        f.unlink(missing_ok=True)
        meta = BioclimMetadata(
            **{
                **src_meta.__dict__,
                "version": new_version,
                "created": datetime.now(UTC).isoformat(),
                "derived_from": src_version,
                "static_layers": {**src_meta.static_layers, **static},
            }
        )
        meta = compute_vif_shortlist(dst, meta, seed=seed, core=core)
        write_metadata(dst, meta)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    kept = [p for p in meta.selected_predictors if p in static]
    log.info(
        "Built %s from %s + soil %s; predictors after VIF<5: %s (soil kept: %s)",
        new_version,
        src_version,
        names,
        meta.selected_predictors,
        kept or "none",
    )
    return dst


def _resample_chelsa_band(
    src_path: Path, band: str, target: BioclimStack, dst: rasterio.io.DatasetWriter, index: int
) -> None:
    factor = CHELSA_TO_WORLDCLIM_FACTOR.get(band, 1.0)
    with rasterio.open(src_path) as src:
        scale, offset = src.scales[0], src.offsets[0]
        # The target's BIO1 defines the land mask (CHELSA also covers the oceans).
        for block in target.iter_blocks(["bio1"]):
            raw = np.full((block.height, block.width), np.nan, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=raw,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=block.transform,
                dst_crs=target.crs,
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            vals = (raw * scale + offset) * factor
            vals[np.isnan(block.data[0])] = np.nan
            dst.write(
                np.where(np.isnan(vals), _NODATA, vals).astype("float32"),
                index,
                window=Window(0, block.row_off, block.width, block.height),
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--version", default=settings.bioclim_version)
    b.add_argument("--resolution", default="2.5m", choices=RESOLUTIONS)
    s = sub.add_parser("add-scenario")
    s.add_argument("--version", default=settings.bioclim_version)
    s.add_argument("--gcm", required=True)
    s.add_argument("--ssp", required=True)
    s.add_argument("--period", required=True)
    s.add_argument(
        "--source",
        choices=("worldclim", "chelsa"),
        help="default: the version's own source (mixing sources is refused)",
    )
    s.add_argument("--src-dir", type=Path, help="pre-downloaded CHELSA future files")
    d = sub.add_parser("derive", help="new version with a different predictor screening")
    d.add_argument("--from", dest="src_version", required=True)
    d.add_argument("--version", required=True)
    d.add_argument(
        "--core",
        default=",".join(DEFAULT_CORE_PREDICTORS),
        help="comma-separated predictors always retained by the VIF screen",
    )
    i = sub.add_parser("install-local")
    i.add_argument("src", type=Path)
    i.add_argument("--version", required=True)
    i.add_argument("--source", required=True)
    i.add_argument("--resolution", required=True)
    h = sub.add_parser("add-hires", help="attach a finer WorldClim stack to a version")
    h.add_argument("--version", default=settings.bioclim_version)
    h.add_argument("--resolution", default="30s", choices=RESOLUTIONS)
    h.add_argument("--src", type=Path, help="local 19-band stack instead of downloading")
    bl = sub.add_parser("add-bias-layer", help="static observer-bias raster (travel time, ...)")
    bl.add_argument("--version", default=settings.bioclim_version)
    bl.add_argument("--name", required=True, help=f"presets: {', '.join(BIAS_LAYER_PRESETS)}")
    bl.add_argument("--src", type=Path, required=True, help="single-band raster (any CRS)")
    bl.add_argument("--direction", choices=("increasing", "decreasing"))
    bl.add_argument("--transform", choices=BIAS_LAYER_TRANSFORMS)
    bl.add_argument("--citation")
    c = sub.add_parser("build-chelsa", help="CHELSA V2.1 cross-check stack on a version's grid")
    c.add_argument("--version", required=True)
    c.add_argument("--match", default=settings.bioclim_version)
    c.add_argument("--src-dir", type=Path, help="directory with the CHELSA_bio*_V.2.1.tif files")
    c.add_argument(
        "--variables",
        default="",
        help="extra CHELSA-BIOCLIM+ climate variables, comma-separated, or 'default' for "
        f"{','.join(DEFAULT_CHELSA_VARIABLES)}",
    )
    c.add_argument(
        "--with-hires",
        action="store_true",
        help="also build the match version's high-resolution grids (30 arc-seconds) from CHELSA",
    )
    so = sub.add_parser(
        "add-soil", help="new version = an existing one + SoilGrids 2.0 topsoil predictors"
    )
    so.add_argument("--from", dest="src_version", default=settings.bioclim_version)
    so.add_argument("--version", required=True, help="new version tag, e.g. bioclim_v3_soil")
    so.add_argument(
        "--properties",
        default=",".join(DEFAULT_SOIL_PROPERTIES),
        help=f"comma-separated, from: {', '.join([*SOILGRIDS_PROPERTIES, *SOIL_DERIVED])}",
    )
    so.add_argument(
        "--source-res",
        choices=SOILGRIDS_RESOLUTIONS,
        help="SoilGrids aggregated product in metres (default: 5000 for grids >= 5', else 1000)",
    )
    so.add_argument("--src-dir", type=Path, help="directory with pre-downloaded SoilGrids files")
    so.add_argument(
        "--core",
        default=",".join(DEFAULT_CORE_PREDICTORS),
        help="predictors always kept by the VIF screen (e.g. bio1,bio12,soil_sand)",
    )
    so.add_argument(
        "--with-hires",
        action="store_true",
        help="also build 1 km soil layers for each registered high-resolution stack",
    )
    args = p.parse_args()
    if args.cmd == "build":
        build(args.version, args.resolution, settings.bioclim_root, settings.random_seed)
    elif args.cmd == "add-bias-layer":
        add_bias_layer(
            args.version,
            args.name,
            args.src,
            settings.bioclim_root,
            args.direction,
            args.transform,
            args.citation,
        )
    elif args.cmd == "add-hires":
        add_hires(args.version, args.resolution, settings.bioclim_root, args.src)
    elif args.cmd == "build-chelsa":
        variables = (
            DEFAULT_CHELSA_VARIABLES
            if args.variables.strip() == "default"
            else tuple(v.strip() for v in args.variables.split(",") if v.strip())
        )
        build_chelsa(
            args.version,
            args.match,
            settings.bioclim_root,
            settings.random_seed,
            args.src_dir,
            variables,
            args.with_hires,
        )
    elif args.cmd == "add-soil":
        add_soil(
            args.src_version,
            args.version,
            settings.bioclim_root,
            tuple(x.strip() for x in args.properties.split(",") if x.strip()),
            args.source_res,
            args.src_dir,
            tuple(x.strip() for x in args.core.split(",") if x.strip()),
            settings.random_seed,
            args.with_hires,
        )
    elif args.cmd == "derive":
        core = tuple(c.strip() for c in args.core.split(",") if c.strip())
        derive(args.src_version, args.version, settings.bioclim_root, core, settings.random_seed)
    elif args.cmd == "add-scenario":
        add_scenario(
            args.version,
            args.gcm,
            args.ssp,
            args.period,
            settings.bioclim_root,
            args.source,
            args.src_dir,
        )
    else:
        install_local_stack(
            args.src,
            args.version,
            settings.bioclim_root,
            args.source,
            args.resolution,
            settings.random_seed,
        )


if __name__ == "__main__":
    main()
