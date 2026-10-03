"""One-time construction of the versioned global bioclim layer (Phase 1).

Downloads WorldClim v2.1 BIO1–19 once, stacks them into a single 19-band Cloud-Optimized
GeoTIFF, records checksum + provenance in metadata.json, and precomputes the global VIF
predictor shortlist. Species runs only ever *read* this layer (via raster_sampler).

    python -m src.features.bioclim_store build --version bioclim_v1 --resolution 2.5m
    python -m src.features.bioclim_store add-scenario --version bioclim_v1 \
        --gcm ACCESS-CM2 --ssp ssp245 --period 2041-2060
    python -m src.features.bioclim_store add-hires --version bioclim_v1 --resolution 30s
    python -m src.features.bioclim_store build-chelsa --version chelsa_v1 --match bioclim_v1

`add-hires` attaches a finer WorldClim stack to an existing version (regional high-resolution
projection of that version's models). `build-chelsa` builds an independent CHELSA v2.1 stack
in WorldClim units on the grid and land mask of `--match`, used by the climate-data
cross-check (SDM_CROSSCHECK_BIOCLIM_VERSION).
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
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.warp import reproject
from rasterio.windows import Window

from src.config import get_settings
from src.domain import BIOCLIM_BANDS
from src.features.collinearity import select_low_collinearity
from src.features.raster_sampler import (
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

CHELSA_BASE = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio"
CHELSA_FILE = "CHELSA_bio{i}_1981-2010_V.2.1.tif"
CHELSA_CITATION = (
    "Karger, D.N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza, R.W., "
    "Zimmermann, N.E., Linder, H.P. & Kessler, M. (2017). Climatologies at high resolution for "
    "the earth's land surface areas. Scientific Data 4, 170122. CHELSA V2.1, "
    "https://doi.org/10.16904/envidat.228"
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
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / STACK_FILENAME, dst / STACK_FILENAME)
    if (src / "scenarios").exists():
        shutil.copytree(src / "scenarios", dst / "scenarios")
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
    )
    meta = compute_vif_shortlist(root, meta, seed=seed)
    write_metadata(root, meta)
    log.info("Built %s: predictors kept after VIF<5: %s", version, meta.selected_predictors)
    return root


def add_scenario(version: str, gcm: str, ssp: str, period: str, bioclim_root: Path) -> str:
    """Register a CMIP6 future bioclim stack (same grid) for the 'what-if' scenario toggle."""
    root = bioclim_root / version
    meta = BioclimMetadata.load(root / METADATA_FILENAME)
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


def build_chelsa(
    version: str,
    match_version: str,
    bioclim_root: Path,
    seed: int = 42,
    src_dir: Path | None = None,
) -> Path:
    """CHELSA V2.1 BIO1–19 (1981–2010) in WorldClim units, average-resampled onto the grid of
    `match_version` and masked to its land cells, as an independent cross-check stack.

    Files are downloaded one at a time (≈5 GB in total) unless `src_dir` holds them already.
    """
    root = bioclim_root / version
    if (root / STACK_FILENAME).exists():
        raise FileExistsError(f"{root} already exists — pick a new version tag")
    target = BioclimStack.open_version(bioclim_root, match_version)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        stack_tmp = tmp_p / "stack.tif"
        profile = {
            "driver": "GTiff",
            "width": target.width,
            "height": target.height,
            "count": len(BIOCLIM_BANDS),
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
        with rasterio.open(stack_tmp, "w", **profile) as dst:
            for i, band in enumerate(BIOCLIM_BANDS, start=1):
                fname = CHELSA_FILE.format(i=i)
                src_path = (
                    src_dir / fname
                    if src_dir is not None
                    else _download(f"{CHELSA_BASE}/{fname}", tmp_p / fname)
                )
                _resample_chelsa_band(src_path, band, target, dst, i)
                dst.set_band_description(i, band)
                if src_dir is None:
                    src_path.unlink()
        rio_copy(stack_tmp, root / STACK_FILENAME, driver="COG", **COG_OPTIONS)
    meta = BioclimMetadata(
        version=version,
        source=(
            f"CHELSA V2.1 1981–2010 ({CHELSA_BASE}), average-resampled to the {match_version} "
            "grid and land mask, converted to WorldClim units"
        ),
        resolution=target.metadata.resolution,
        crs="EPSG:4326",
        bands=BIOCLIM_BANDS,
        created=datetime.now(UTC).isoformat(),
        sha256=sha256_file(root / STACK_FILENAME),
        citation=CHELSA_CITATION,
        aligned_to=match_version,
    )
    meta = compute_vif_shortlist(root, meta, seed=seed)
    write_metadata(root, meta)
    log.info("Built CHELSA cross-check stack %s on the %s grid", version, match_version)
    return root


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
    c = sub.add_parser("build-chelsa", help="CHELSA V2.1 cross-check stack on a version's grid")
    c.add_argument("--version", required=True)
    c.add_argument("--match", default=settings.bioclim_version)
    c.add_argument("--src-dir", type=Path, help="directory with the CHELSA_bio*_V.2.1.tif files")
    args = p.parse_args()
    if args.cmd == "build":
        build(args.version, args.resolution, settings.bioclim_root, settings.random_seed)
    elif args.cmd == "add-hires":
        add_hires(args.version, args.resolution, settings.bioclim_root, args.src)
    elif args.cmd == "build-chelsa":
        build_chelsa(
            args.version, args.match, settings.bioclim_root, settings.random_seed, args.src_dir
        )
    elif args.cmd == "derive":
        core = tuple(c.strip() for c in args.core.split(",") if c.strip())
        derive(args.src_version, args.version, settings.bioclim_root, core, settings.random_seed)
    elif args.cmd == "add-scenario":
        add_scenario(args.version, args.gcm, args.ssp, args.period, settings.bioclim_root)
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
