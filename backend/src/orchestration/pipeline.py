"""Species pipeline: the incremental fetch → conditional retrain → publish lifecycle.

Implements the reference contract from CLAUDE.md verbatim in `SpeciesPipeline.run`:

    if species_not_in_registry(taxon_key):
        full_pipeline_run(taxon_key)
    else:
        delta = fetch_incremental_gbif(taxon_key, since=last_gbif_fetch_ts)
        if delta.count == 0:
            serve_cached_results(taxon_key)
        else:
            append_to_occurrence_store(delta)
            update_watermark(taxon_key, now())
            if (delta.count / total_count > 0.05) or delta_outside_suitability_envelope(delta):
                retrain_model(taxon_key)
                regenerate_rasters_and_bulletin(taxon_key)
            else:
                flag_minor_update_pending(taxon_key)

Two user-driven additions sit *outside* that decision (they never change the threshold):
  * training is gated on a user-confirmed native-range polygon (constraint 6);
  * a user edit to the confirmed polygon sets `retrain_needed`, honoured on the next run.

Asset steps are executed through a `StepRunner` so the same logic runs directly (tests) or
as Prefect tasks (orchestration/flows.py): raw_occurrences → features → model →
suitability_raster → bulletin.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, TypeVar

import elapid
import numpy as np
import pandas as pd
from shapely.geometry import shape

from src.config import Settings
from src.connectors.gbif_client import download_citation, inat_citation
from src.domain import (
    BIOCLIM_BANDS,
    ModelVersionRecord,
    NativeRange,
    OccurrenceRecord,
    SpeciesRecord,
    SpeciesStatus,
    utcnow,
)
from src.features.background_sampler import (
    BackgroundSample,
    buffer_region,
    cell_dedupe,
    distance_thin,
    sample_buffered_background,
    sample_target_group_background,
)
from src.features.bioclim_store import dvc_pointer_md5
from src.features.raster_sampler import (
    METADATA_FILENAME,
    BioclimMetadata,
    BioclimStack,
    sample_raster_file,
)
from src.modeling.crosscheck import climate_crosscheck
from src.modeling.maxent_trainer import (
    TRANSFERABILITY_CAVEAT,
    TrainingConfig,
    TrainingData,
    select_model_type,
    train_maxent,
    training_presence_mask,
)
from src.modeling.native_range import (
    effective_range_labels,
    propose_native_range,
    validate_geojson_polygon,
)
from src.modeling.projection import ScenarioOutputs, project_model, project_scenario
from src.modeling.severity_index import (
    SeverityConfig,
    SeverityInputs,
    compute_severity,
    spread_rate_cells_per_year,
)
from src.persistence.artifact_store import ArtifactStore
from src.persistence.model_registry import ModelRegistry
from src.persistence.repository import Repository
from src.persistence.training_snapshot import write_training_snapshot

log = logging.getLogger(__name__)
T = TypeVar("T")

TRACKED_PACKAGES = ("elapid", "pygbif", "pyinaturalist", "rasterio", "scikit-learn", "numpy")


class StepRunner(Protocol):
    def __call__(self, name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T: ...


def direct_runner(name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return fn(*args, **kwargs)


ProgressFn = Callable[[str, float, str | None], None]


def _no_progress(stage: str, fraction: float, message: str | None) -> None:
    log.info("[%s %.0f%%] %s", stage, fraction * 100, message or "")


class GbifLike(Protocol):
    download_enabled: bool

    def get_taxon(self, taxon_key: int) -> Any: ...
    def download_occurrences(self, taxon_key: int, workdir: Path) -> Any: ...
    def search_occurrences(
        self,
        taxon_key: int,
        since: datetime | None = None,
        until: datetime | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[OccurrenceRecord]: ...
    def fetch_effort_density(
        self, taxon_key: int, bbox: tuple[float, float, float, float], max_pixel_deg: float
    ) -> Any: ...


class InatLike(Protocol):
    def find_taxon(self, scientific_name: str) -> Any: ...
    def get_observations(
        self,
        inat_taxon_id: int,
        taxon_key: int,
        updated_since: datetime | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[OccurrenceRecord]: ...


class BulletinLike(Protocol):
    def generate(self, taxon_key: int, model_version: int) -> dict[str, str]: ...


class Decision(StrEnum):
    INGESTED_AWAITING_REVIEW = "ingested_awaiting_native_range_review"
    AWAITING_REVIEW = "awaiting_native_range_review"
    TRAINED = "trained"
    CACHED = "served_cached_results"
    MINOR_UPDATE = "minor_update_pending"
    RETRAINED = "retrained"


@dataclass
class PipelineOutcome:
    taxon_key: int
    decision: Decision
    n_new_records: int = 0
    model_version: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "taxon_key": self.taxon_key,
            "decision": str(self.decision),
            "n_new_records": self.n_new_records,
            "model_version": self.model_version,
            "details": self.details,
        }


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


@dataclass
class SpeciesPipeline:
    repo: Repository
    gbif: GbifLike
    inat: InatLike | None
    artifacts: ArtifactStore
    registry: ModelRegistry
    settings: Settings
    bulletin: BulletinLike | None = None
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    step: StepRunner = direct_runner
    progress: ProgressFn = _no_progress
    _last_fetch_started: datetime | None = field(default=None, init=False, repr=False)
    # (download key, DOI) of a DOI-backed full re-sync, recorded with the watermark.
    _last_download: tuple[str, str | None] | None = field(default=None, init=False, repr=False)
    # False when the last iNaturalist fetch failed: its watermark must then stay put, or the
    # observations it missed would be skipped by every later incremental fetch.
    _inat_fetch_ok: bool = field(default=True, init=False, repr=False)

    # ================================================================ contract
    def run(
        self, taxon_key: int, force_retrain: bool = False, full_resync: bool = False
    ) -> PipelineOutcome:
        """One run per species at a time: a second request (a double click, confirm-and-train
        while a check is in flight, the nightly sweep) waits, then sees the first run's result
        — instead of both training the same model version."""
        with self._exclusive(taxon_key):
            return self._run(taxon_key, force_retrain, full_resync)

    def _exclusive(self, taxon_key: int) -> AbstractContextManager[None]:
        return self.repo.species_lock(
            taxon_key,
            on_wait=lambda: self.progress(
                "raw_occurrences", 0.01, "Waiting for another run of this species to finish"
            ),
        )

    def _run(self, taxon_key: int, force_retrain: bool, full_resync: bool) -> PipelineOutcome:
        """`full_resync` re-fetches every record (not just the watermark delta) and appends
        whatever is missing — e.g. after a capped or interrupted ingest. The retrain decision
        is unchanged: only records actually inserted count as new."""
        species = self.repo.get_species(taxon_key)
        if species is None:
            return self.full_pipeline_run(taxon_key)

        delta = self.step("raw_occurrences", self.fetch_incremental, species, full_resync)
        if len(delta) == 0:
            if self._needs_training(species, force_retrain):
                self.update_watermark(taxon_key, delta_fetch_ts=self._last_fetch_started)
                return self._train_if_confirmed(
                    taxon_key, trigger=self._trigger(species, force_retrain)
                )
            return self.serve_cached_results(species)

        prev_total = species.n_occurrences_total
        n_new = self.append_to_occurrence_store(taxon_key, delta)
        self.update_watermark(taxon_key, delta_fetch_ts=self._last_fetch_started)
        if n_new == 0:  # everything returned was already stored (re-interpreted records)
            if self._needs_training(species, force_retrain):
                return self._train_if_confirmed(
                    taxon_key, trigger=self._trigger(species, force_retrain)
                )
            return self.serve_cached_results(species)

        ratio = n_new / prev_total if prev_total else float("inf")
        outside = self.delta_outside_suitability_envelope(species, delta)
        if (
            ratio > self.settings.retrain_delta_fraction
            or outside
            or self._needs_training(species, force_retrain)
        ):
            trigger = (
                f"delta {n_new} records ({ratio:.1%} > {self.settings.retrain_delta_fraction:.0%})"
                if ratio > self.settings.retrain_delta_fraction
                else (
                    "new records outside suitability envelope"
                    if outside
                    else self._trigger(species, force_retrain)
                )
            )
            out = self._train_if_confirmed(taxon_key, trigger=trigger)
            out.n_new_records = n_new
            if out.decision == Decision.TRAINED:
                out.decision = Decision.RETRAINED
            return out
        return self.flag_minor_update_pending(taxon_key, n_new, ratio)

    def _needs_training(self, species: SpeciesRecord, force: bool) -> bool:
        return force or species.retrain_needed or species.model_version == 0

    @staticmethod
    def _trigger(species: SpeciesRecord, force: bool) -> str:
        if force:
            return "manual retrain request"
        if species.model_version == 0:
            return "initial training"
        return "native range edited / retrain flagged"

    # ============================================================ full run
    def full_pipeline_run(self, taxon_key: int) -> PipelineOutcome:
        self.progress("raw_occurrences", 0.02, "Resolving taxon")
        taxon = self.gbif.get_taxon(taxon_key)
        inat_taxon = None
        if self.inat is not None and self.settings.inat_enabled:
            try:
                inat_taxon = self.inat.find_taxon(taxon.canonical_name or taxon.scientific_name)
            except Exception as exc:  # iNaturalist is supplementary; never block on it
                log.warning("iNaturalist taxon lookup failed: %s", exc)
        self.repo.create_species(
            SpeciesRecord(
                taxon_key=taxon_key,
                scientific_name=taxon.scientific_name,
                canonical_name=taxon.canonical_name,
                common_name=getattr(inat_taxon, "common_name", None),
                inat_taxon_id=getattr(inat_taxon, "inat_taxon_id", None),
                thumbnail_url=getattr(inat_taxon, "thumbnail_url", None),
                status=SpeciesStatus.INGESTING,
            )
        )
        try:
            n = self.step("raw_occurrences", self._initial_ingest, taxon_key)
        except Exception as exc:
            # Leave a recoverable state: the next run re-fetches everything (since=None).
            self.repo.update_species(
                taxon_key, status=SpeciesStatus.FAILED, last_error=f"{type(exc).__name__}: {exc}"
            )
            raise
        self.progress("raw_occurrences", 0.35, f"Stored {n} occurrence records")

        nr = self.ensure_native_range_draft(taxon_key)
        if nr.status != "confirmed":
            self.repo.update_species(taxon_key, status=SpeciesStatus.AWAITING_NATIVE_RANGE_REVIEW)
            return PipelineOutcome(
                taxon_key,
                Decision.INGESTED_AWAITING_REVIEW,
                n_new_records=n,
                details={"native_range_source": nr.source},
            )
        out = self.train_and_publish(taxon_key, trigger="initial training")
        out.n_new_records = n
        return out

    def _initial_ingest(self, taxon_key: int) -> int:
        started = utcnow()
        download_key = doi = None
        if self.gbif.download_enabled:
            self.progress("raw_occurrences", 0.05, "Requesting DOI-backed GBIF download")
            res = self.gbif.download_occurrences(taxon_key, self._download_dir(taxon_key))
            records, download_key, doi = res.records, res.download_key, res.doi
        else:
            log.warning("GBIF credentials missing — using paged search (no download DOI)")
            records = self.gbif.search_occurrences(taxon_key, on_progress=self._gbif_progress)
        n = self.repo.append_occurrences(records)
        n += self._ingest_inat(taxon_key, since=None)
        counts = self.repo.occurrence_counts(taxon_key)
        self.repo.update_species(
            taxon_key,
            last_gbif_fetch_ts=started,
            **self._inat_watermark(started),
            last_gbif_download_key=download_key,
            last_gbif_download_doi=doi,
            n_occurrences_total=counts["total"],
            n_occurrences_native=counts["native"],
        )
        return n

    def _download_dir(self, taxon_key: int) -> Path:
        return self.artifacts.local_path(str(self.artifacts.species_dir(taxon_key))) / "gbif"

    def _inat_watermark(self, ts: datetime) -> dict[str, datetime]:
        return {"last_inat_fetch_ts": ts} if self._inat_fetch_ok else {}

    def _ingest_inat(self, taxon_key: int, since: datetime | None) -> int:
        self._inat_fetch_ok = True
        species = self.repo.get_species(taxon_key)
        if (
            self.inat is None
            or not self.settings.inat_enabled
            or not species
            or not (species.inat_taxon_id)
        ):
            return 0
        try:
            obs = self.inat.get_observations(
                species.inat_taxon_id,
                taxon_key,
                updated_since=since,
                on_progress=self._inat_progress,
            )
        except Exception as exc:
            log.warning("iNaturalist fetch failed (continuing with GBIF only): %s", exc)
            self._inat_fetch_ok = False
            return 0
        already_in_gbif = self.repo.inat_crossref_ids(taxon_key)
        fresh = [o for o in obs if o.record_id not in already_in_gbif]
        return self.repo.append_occurrences(fresh)

    def _gbif_progress(self, fetched: int, total: int) -> None:
        """Per-page GBIF progress (also the heartbeat the stale-job reaper relies on)."""
        self.progress(
            "raw_occurrences",
            0.05 + 0.25 * fetched / max(total, 1),
            f"GBIF: {fetched:,} of {total:,} records",
        )

    def _inat_progress(self, fetched: int, total: int) -> None:
        self.progress(
            "raw_occurrences",
            0.3 + 0.05 * fetched / max(total, 1),
            f"iNaturalist: {fetched:,} of {total:,} research-grade observations",
        )

    # ====================================================== incremental
    def fetch_incremental(
        self, species: SpeciesRecord, full: bool = False
    ) -> list[OccurrenceRecord]:
        self._last_fetch_started = utcnow()
        self._last_download = None
        self._inat_fetch_ok = True
        if full and self.gbif.download_enabled:
            # Periodic full refresh: the DOI-backed download is complete (no search cap) and
            # citable, so it replaces the species' current download key/DOI.
            self.progress("raw_occurrences", 0.05, "Requesting DOI-backed GBIF download")
            res = self.gbif.download_occurrences(
                species.taxon_key, self._download_dir(species.taxon_key)
            )
            delta = list(res.records)
            self._last_download = (res.download_key, res.doi)
        else:
            self.progress(
                "raw_occurrences",
                0.05,
                "Re-fetching all records" if full else "Checking GBIF for new records",
            )
            delta = list(
                self.gbif.search_occurrences(
                    species.taxon_key,
                    since=None if full else species.last_gbif_fetch_ts,
                    on_progress=self._gbif_progress,
                )
            )
        if self.inat is not None and self.settings.inat_enabled and species.inat_taxon_id:
            try:
                inat = self.inat.get_observations(
                    species.inat_taxon_id,
                    species.taxon_key,
                    updated_since=None if full else species.last_inat_fetch_ts,
                    on_progress=self._inat_progress,
                )
                crossref = self.repo.inat_crossref_ids(species.taxon_key) | {
                    r.inat_crossref_id for r in delta if r.inat_crossref_id
                }
                delta.extend(o for o in inat if o.record_id not in crossref)
            except Exception as exc:
                log.warning("iNaturalist incremental fetch failed: %s", exc)
                self._inat_fetch_ok = False
        return delta

    def append_to_occurrence_store(self, taxon_key: int, delta: list[OccurrenceRecord]) -> int:
        return self.repo.append_occurrences(delta)

    def update_watermark(self, taxon_key: int, delta_fetch_ts: datetime | None) -> None:
        ts = delta_fetch_ts or utcnow()
        counts = self.repo.occurrence_counts(taxon_key)
        download: dict[str, str | None] = {}
        if self._last_download is not None:
            key, doi = self._last_download
            download = {"last_gbif_download_key": key, "last_gbif_download_doi": doi}
        self.repo.update_species(
            taxon_key,
            last_gbif_fetch_ts=ts,
            **self._inat_watermark(ts),
            **download,
            n_occurrences_total=counts["total"],
            n_occurrences_native=counts["native"],
        )

    def serve_cached_results(self, species: SpeciesRecord) -> PipelineOutcome:
        self.update_watermark(species.taxon_key, self._last_fetch_started)
        self.progress("done", 1.0, "No new records — serving cached results")
        return PipelineOutcome(
            species.taxon_key, Decision.CACHED, model_version=species.model_version or None
        )

    def flag_minor_update_pending(
        self, taxon_key: int, n_new: int, ratio: float
    ) -> PipelineOutcome:
        species = self.repo.get_species(taxon_key)
        assert species is not None
        self.repo.update_species(
            taxon_key,
            n_pending_records=species.n_pending_records + n_new,
            status=SpeciesStatus.MINOR_UPDATE_PENDING,
        )
        self.progress("done", 1.0, f"{n_new} new records ({ratio:.1%}) — minor update pending")
        return PipelineOutcome(
            taxon_key,
            Decision.MINOR_UPDATE,
            n_new_records=n_new,
            model_version=species.model_version or None,
            details={"delta_fraction": ratio},
        )

    def delta_outside_suitability_envelope(
        self, species: SpeciesRecord, delta: list[OccurrenceRecord]
    ) -> bool:
        """True if ≥ N new records fall in cells the current model classifies as unsuitable."""
        if not species.model_version or not delta:
            return False
        mv = self.repo.get_model_version(species.taxon_key, species.model_version)
        if mv is None or "suitability" not in mv.artifacts:
            return False
        path = self.artifacts.local_path(mv.artifacts["suitability"])
        if not path.exists():
            return False
        suit = sample_raster_file(path, [r.longitude for r in delta], [r.latitude for r in delta])
        threshold = float(mv.metrics.get("threshold", 0.0))
        n_outside = int(np.sum(suit[~np.isnan(suit)] < threshold))
        return n_outside >= self.settings.envelope_min_outside_points

    # ======================================================== native range
    def ensure_native_range_draft(self, taxon_key: int) -> NativeRange:
        existing = self.repo.get_native_range(taxon_key)
        if existing is not None:
            return existing
        curated = self.settings.native_range_root / f"{taxon_key}.geojson"
        if curated.exists():
            import json

            fc = json.loads(curated.read_text(encoding="utf-8"))
            feat = fc["features"][0] if fc.get("type") == "FeatureCollection" else fc
            props = feat.get("properties") or {}
            # A curator can mark a reference polygon confirmed explicitly; otherwise it is a
            # draft like any other and must be reviewed in the UI.
            status = "confirmed" if props.get("status") == "confirmed" else "draft"
            return self.repo.save_native_range(
                NativeRange(
                    taxon_key=taxon_key,
                    geometry=validate_geojson_polygon(feat["geometry"]),
                    status=status,
                    source=f"curated:{props.get('source', curated.name)}",
                    note=props.get("note"),
                    confirmed_ts=utcnow() if status == "confirmed" else None,
                )
            )
        occ = self.repo.load_occurrences(taxon_key)
        proposal = propose_native_range(occ)
        return self.repo.save_native_range(
            NativeRange(
                taxon_key=taxon_key,
                geometry=proposal.geometry,
                status="draft",
                source=proposal.source,
                note=proposal.note,
            )
        )

    def _train_if_confirmed(self, taxon_key: int, trigger: str) -> PipelineOutcome:
        nr = self.ensure_native_range_draft(taxon_key)
        if nr.status != "confirmed":
            self.repo.update_species(taxon_key, status=SpeciesStatus.AWAITING_NATIVE_RANGE_REVIEW)
            return PipelineOutcome(taxon_key, Decision.AWAITING_REVIEW)
        return self.train_and_publish(taxon_key, trigger=trigger)

    # ======================================================== train + publish
    def train_and_publish(self, taxon_key: int, trigger: str) -> PipelineOutcome:
        """retrain_model + regenerate_rasters_and_bulletin."""
        species = self.repo.get_species(taxon_key)
        assert species is not None
        nr = self.repo.get_native_range(taxon_key)
        if nr is None or nr.status != "confirmed":
            raise RuntimeError("Training requires a user-confirmed native-range polygon")
        self.repo.update_species(taxon_key, status=SpeciesStatus.TRAINING, last_error=None)
        try:
            stack = BioclimStack.open_version(
                self.settings.bioclim_root, self.settings.bioclim_version
            )
            occ = self.step("features", self.extract_features, taxon_key, stack)
            result, data_info = self.step("model", self.fit_model, occ, nr, stack)
            version = species.model_version + 1
            out = self.step(
                "suitability_raster",
                self.publish_model,
                species,
                version,
                occ,
                nr,
                stack,
                result,
                data_info,
                trigger,
            )
        except Exception as exc:
            # Keep the retrain owed: the delta that triggered it is already stored and the
            # watermark advanced, so without the flag the next run would see no delta and keep
            # serving the stale model.
            self.repo.update_species(
                taxon_key,
                status=SpeciesStatus.FAILED,
                last_error=f"{type(exc).__name__}: {exc}",
                retrain_needed=True,
            )
            raise
        if self.bulletin is not None:
            try:
                self.step("bulletin", self.bulletin.generate, taxon_key, version)
            except Exception as exc:  # the model is published even if PDF rendering fails
                log.exception("Bulletin generation failed")
                out.details["bulletin_error"] = f"{type(exc).__name__}: {exc}"
        self.progress("done", 1.0, f"Model v{version} published")
        return out

    def extract_features(self, taxon_key: int, stack: BioclimStack) -> pd.DataFrame:
        self.progress("features", 0.4, "Sampling bioclim predictors at occurrences")
        occ = self.repo.load_occurrences(taxon_key)
        feats = stack.sample(occ["longitude"], occ["latitude"], BIOCLIM_BANDS)
        missing = occ["bio1"].isna().to_numpy() & ~feats["bio1"].isna().to_numpy()
        if missing.any():
            self.repo.fill_features(
                taxon_key,
                occ.loc[missing, ["source", "record_id"]].reset_index(drop=True),
                feats.loc[missing].reset_index(drop=True),
                stack.version,
            )
        out = occ.drop(columns=list(BIOCLIM_BANDS)).join(feats)
        out["bioclim_version"] = stack.version
        # Per-species GeoParquet-compatible feature table (lon/lat + predictors).
        path = self.artifacts.local_path(self.artifacts.features_uri(taxon_key))
        path.parent.mkdir(parents=True, exist_ok=True)
        out.assign(event_date=out["event_date"].astype("string")).to_parquet(path, index=False)
        # A fresh table supersedes one archived while the species was inactive.
        self.artifacts.local_path(self.artifacts.archived_features_uri(taxon_key)).unlink(
            missing_ok=True
        )
        return out

    def fit_model(
        self, occ: pd.DataFrame, nr: NativeRange, stack: BioclimStack
    ) -> tuple[Any, dict[str, Any]]:
        cfg = self.training_config
        predictors = list(stack.metadata.selected_predictors or BIOCLIM_BANDS)
        occ = occ[~occ[predictors].isna().any(axis=1)].reset_index(drop=True)
        labels = effective_range_labels(occ, nr.geometry)
        model_type = select_model_type(labels)
        pres = occ[training_presence_mask(labels, model_type)].reset_index(drop=True)

        keep = cell_dedupe(pres["longitude"], pres["latitude"], stack.transform)
        pres = pres.iloc[keep].reset_index(drop=True)
        thin_seed = cfg.seed
        keep = distance_thin(pres["longitude"], pres["latitude"], cfg.thin_km, seed=thin_seed)
        pres = pres.iloc[keep].reset_index(drop=True)
        self.progress("model", 0.5, f"Model {model_type}: {len(pres)} thinned presences")

        bg_seed = cfg.seed + 1
        bg, background_info = self.sample_background(nr.taxon_key, pres, stack, predictors, bg_seed)
        data = TrainingData(
            predictors=predictors,
            presence_lon=pres["longitude"].to_numpy(),
            presence_lat=pres["latitude"].to_numpy(),
            presence_x=pres[predictors].reset_index(drop=True),
            background_lon=bg.lon,
            background_lat=bg.lat,
            background_x=bg.features,
            background_method=bg.method,
        )
        self.progress("model", 0.55, "Spatial block cross-validation + tuning")
        result = train_maxent(data, model_type, cfg)
        info = {
            "labels": labels,
            "occ_valid": occ,
            "training_presences": pres,
            "training_data": data,
            "seeds": {
                "thinning": thin_seed,
                "background_sampling": bg_seed,
                "cv_fold_assignment": cfg.seed,
                "maxent_random_state": cfg.seed,
                "permutation_importance": cfg.seed,
            },
            "n_records_by_effective_label": labels.value_counts().to_dict(),
            "background": background_info,
        }
        return result, info

    def sample_background(
        self,
        taxon_key: int,
        pres: pd.DataFrame,
        stack: BioclimStack,
        predictors: list[str],
        seed: int,
    ) -> tuple[BackgroundSample, dict[str, Any]]:
        """Target-group background when configured and the effort data supports it, else
        uniform buffered background. The returned record (method, target group, effort
        query, any fallback reason) is logged with the model version."""
        cfg = self.training_config
        info: dict[str, Any] = {
            "requested_method": cfg.background_method,
            "buffer_km": cfg.background_buffer_km,
        }
        region = buffer_region(pres["longitude"], pres["latitude"], cfg.background_buffer_km)
        if cfg.background_method == "target_group":
            try:
                bg = self._target_group_background(taxon_key, region, stack, predictors, seed, info)
            except Exception as exc:  # effort data is an enhancement; never block training
                log.warning("Target-group background unavailable: %s", exc)
                info["fallback_reason"] = (
                    f"GBIF sampling-effort data unavailable ({type(exc).__name__}: {exc})"
                )
                bg = None
            if bg is not None:
                info["method"] = bg.method
                return bg, info
            self.progress("model", 0.53, f"Buffered background: {info['fallback_reason']}")
        bg = sample_buffered_background(
            pres["longitude"],
            pres["latitude"],
            stack,
            cfg.n_background,
            cfg.background_buffer_km,
            seed,
            predictors,
        )
        info["method"] = bg.method
        return bg, info

    def _target_group_background(
        self,
        taxon_key: int,
        region: Any,
        stack: BioclimStack,
        predictors: list[str],
        seed: int,
        info: dict[str, Any],
    ) -> BackgroundSample | None:
        cfg = self.training_config
        rank = cfg.target_group_rank.lower()
        group = self.gbif.get_taxon(taxon_key).higher_taxon(rank)
        if group is None:
            info["fallback_reason"] = f"taxon has no {rank} in the GBIF backbone"
            return None
        group_key, group_name = group
        self.progress("model", 0.52, f"Fetching GBIF sampling effort for {rank} {group_name}")
        effort = self.gbif.fetch_effort_density(
            group_key, tuple(region.bounds), max_pixel_deg=abs(stack.res[0])
        )
        info["target_group"] = {
            "rank": rank,
            "taxon_key": group_key,
            "name": group_name,
            "n_records": effort.n_records,
            "n_pixels": len(effort.lon),
            "map_zoom": effort.zoom,
            "pixel_deg": effort.pixel_deg,
            "n_tiles": effort.n_tiles,
            "query": effort.query,
            "fetched_ts": utcnow().isoformat(),
        }
        if not effort.lon:
            info["fallback_reason"] = f"no GBIF records of {rank} {group_name} in the region"
            return None
        bg = sample_target_group_background(
            effort.lon,
            effort.lat,
            stack,
            cfg.n_background,
            seed,
            predictors,
            region=region,
            method=f"target_group_{rank}",
        )
        if len(bg.lon) < cfg.min_target_group_cells:
            info["fallback_reason"] = (
                f"only {len(bg.lon)} cells with {rank} {group_name} records in the background "
                f"region (< {cfg.min_target_group_cells})"
            )
            return None
        return bg

    def publish_model(
        self,
        species: SpeciesRecord,
        version: int,
        occ: pd.DataFrame,
        nr: NativeRange,
        stack: BioclimStack,
        result: Any,
        info: dict[str, Any],
        trigger: str,
    ) -> PipelineOutcome:
        taxon_key = species.taxon_key
        vdir_uri = str(self.artifacts.version_dir(taxon_key, version))
        vdir = self.artifacts.local_path(vdir_uri)
        vdir.mkdir(parents=True, exist_ok=True)

        model_path = vdir / "model.pkl"
        elapid.save_object(
            {
                "model": result.model,
                "predictors": result.predictors,
                "threshold": result.threshold,
                "model_type": result.model_type,
                "mess_reference": result.mess_reference,
                "presence_envelope": result.presence_envelope,
            },
            str(model_path),
        )

        # Immutable, hashed copy of exactly what this version was trained on.
        data: TrainingData = info["training_data"]
        snapshot = write_training_snapshot(
            vdir,
            info["training_presences"],
            data.background_lon,
            data.background_lat,
            data.background_x,
            data.predictors,
            info["occ_valid"],
            info["labels"],
        )
        snapshot_meta = {
            name: f.describe(self.artifacts.relative_uri(f.path)) for name, f in snapshot.items()
        }

        self.progress("suitability_raster", 0.7, "Projecting suitability + MESS globally")
        scenario_stacks = {
            name: BioclimStack.open_version(self.settings.bioclim_root, stack.version, name)
            for name in stack.metadata.scenarios
        }
        occ_valid: pd.DataFrame = info["occ_valid"]
        proj = project_model(
            result,
            stack,
            vdir,
            shape(nr.geometry),
            occ_valid["longitude"].to_numpy(),
            occ_valid["latitude"].to_numpy(),
            scenario_stacks=scenario_stacks,
        )

        labels: pd.Series = info["labels"]
        years = pd.to_datetime(occ_valid["event_date"], errors="coerce").dt.year
        sev_cfg = SeverityConfig.from_dict(species.severity_config)
        spread = spread_rate_cells_per_year(
            occ_valid.assign(year=years),
            (labels != "native").to_numpy(),
            sev_cfg.spread_window_years,
            datetime.now(UTC).year,
        )
        s = proj.summary
        sev_inputs = SeverityInputs(
            candidate_mean_suitability=s["candidate_mean_suitability"],
            candidate_fraction_of_nonnative_land=s["candidate_fraction_of_nonnative_land"],
            candidate_envelope_fraction=s["candidate_envelope_fraction"],
            candidate_mess_ok_fraction=s["candidate_mess_ok_fraction"],
            new_cells_per_year=spread,
            model_type=result.model_type,
        )
        severity = compute_severity(sev_inputs, sev_cfg)

        artifacts = {
            "model": self.artifacts.relative_uri(model_path),
            "suitability": self.artifacts.relative_uri(proj.suitability),
            "mess": self.artifacts.relative_uri(proj.mess),
            "zones": self.artifacts.relative_uri(proj.zones),
            "extrapolation": self.artifacts.relative_uri(proj.extrapolation),
            "training_data": snapshot_meta["training_data"]["path"],
            "occurrence_manifest": snapshot_meta["occurrence_manifest"]["path"],
            "scenarios": {
                name: self._scenario_artifacts(out) for name, out in proj.scenarios.items()
            },
        }
        refreshed = self.repo.get_species(taxon_key)
        assert refreshed is not None
        reproducibility = {
            "bioclim_version_used": stack.version,
            "bioclim_source": stack.metadata.source,
            "bioclim_sha256": stack.metadata.sha256,
            "bioclim_dvc_md5": dvc_pointer_md5(self.settings.bioclim_root, stack.version),
            "gbif_download_key": refreshed.last_gbif_download_key,
            "gbif_download_doi": refreshed.last_gbif_download_doi,
            "gbif_citation": download_citation(
                refreshed.last_gbif_download_doi,
                refreshed.last_gbif_download_key,
                taxon_key=taxon_key,
            ),
            "inat_citation": (
                inat_citation(refreshed.inat_taxon_id)
                if refreshed.inat_taxon_id and (occ_valid["source"] == "iNaturalist").any()
                else None
            ),
            "package_versions": package_versions(),
            "random_seeds": info["seeds"],
            "training_config": self.training_config.as_dict(),
            "background": info["background"],
            "native_range_source": nr.source,
            "native_range_confirmed_ts": nr.confirmed_ts.isoformat() if nr.confirmed_ts else None,
            "training_data": snapshot_meta["training_data"],
            "occurrence_manifest": snapshot_meta["occurrence_manifest"],
        }
        metrics: dict[str, Any] = {
            **{k: _finite(v) for k, v in result.cv_summary.items()},
            "train_auc": _finite(result.train_auc),
            "threshold": result.threshold,
            "threshold_rule": "10th percentile training presence",
            "cv_method": (
                f"spatial block CV (elapid.GeographicKFold, k={self.training_config.n_folds})"
            ),
            "n_cv_folds": len(result.folds),
            "folds": [
                {**f.__dict__, "auc": _finite(f.auc), "tss": _finite(f.tss), "cbi": _finite(f.cbi)}
                for f in result.folds
            ],
            "tuning": [
                {k: _finite(v) if isinstance(v, float) else v for k, v in t.items()}
                for t in result.tuning
            ],
            "feature_classes": result.feature_classes,
            "beta_multiplier": result.beta_multiplier,
            "predictors": result.predictors,
            "variable_importance": result.importance,
            "n_presence": result.n_presence,
            "n_background": result.n_background,
            "background_method": result.background_method,
            "n_records_by_effective_label": {
                k: int(v) for k, v in info["n_records_by_effective_label"].items()
            },
            "model_type": result.model_type,
            "confidence_label": result.confidence_label,
            "transferability_caveat": TRANSFERABILITY_CAVEAT,
            "projection": s,
            "reproducibility": reproducibility,
        }

        run_id = self.registry.log_model_version(
            taxon_key,
            version,
            params={
                "model_type": result.model_type,
                "feature_classes": result.feature_classes,
                "beta_multiplier": result.beta_multiplier,
                "predictors": ",".join(result.predictors),
                "background_method": result.background_method,
                "trigger": trigger,
                **{f"seed_{k}": v for k, v in info["seeds"].items()},
                "bioclim_version": stack.version,
                "gbif_download_doi": refreshed.last_gbif_download_doi or "none",
                "training_data_sha256": snapshot_meta["training_data"]["sha256"],
                "occurrence_manifest_sha256": snapshot_meta["occurrence_manifest"]["sha256"],
                **{f"pkg_{k}": v for k, v in reproducibility["package_versions"].items()},
            },
            metrics={
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
            tags={"model_type": result.model_type, "confidence": result.confidence_label},
            artifact_paths=[
                model_path,
                snapshot["training_data"].path,
                snapshot["occurrence_manifest"].path,
                proj.suitability,
                proj.mess,
                proj.zones,
                proj.extrapolation,
            ],
        )
        self.repo.add_model_version(
            ModelVersionRecord(
                taxon_key=taxon_key,
                model_version=version,
                model_type=result.model_type,
                trigger=trigger,
                model_path=artifacts["model"],
                artifacts=artifacts,
                metrics=metrics,
                severity=severity,
                mlflow_run_id=run_id,
            )
        )
        counts = self.repo.occurrence_counts(taxon_key)
        self.repo.update_species(
            taxon_key,
            model_version=version,
            model_path=artifacts["model"],
            model_trained_ts=utcnow(),
            model_metrics=metrics,
            bioclim_version_used=stack.version,
            retrain_needed=False,
            n_pending_records=0,
            n_occurrences_total=counts["total"],
            n_occurrences_native=counts["native"],
            status=SpeciesStatus.UP_TO_DATE,
        )
        return PipelineOutcome(
            taxon_key,
            Decision.TRAINED,
            model_version=version,
            details={
                "trigger": trigger,
                "model_type": result.model_type,
                "severity": severity["score_0_100"],
            },
        )

    # ======================================================== scenarios
    def _scenario_artifacts(self, out: ScenarioOutputs) -> dict[str, Any]:
        return {
            "suitability": self.artifacts.relative_uri(out.suitability),
            "extrapolation": self.artifacts.relative_uri(out.extrapolation),
            "extrapolated_land_fraction": out.extrapolated_land_fraction,
        }

    def project_scenarios(self, taxon_key: int, only_missing: bool = True) -> dict[str, Any]:
        """Project the *current* stored model onto the climate scenarios registered for the
        bioclim version it was trained on — no retraining. Scenario outputs are derived
        artifacts added to the existing model version."""
        with self._exclusive(taxon_key):
            return self._project_scenarios(taxon_key, only_missing)

    def _project_scenarios(self, taxon_key: int, only_missing: bool) -> dict[str, Any]:
        species = self.repo.get_species(taxon_key)
        if species is None or not species.model_version:
            return {"taxon_key": taxon_key, "projected": []}
        mv = self.repo.get_model_version(taxon_key, species.model_version)
        assert mv is not None
        version = mv.metrics["reproducibility"]["bioclim_version_used"]
        stack = BioclimStack.open_version(self.settings.bioclim_root, version)
        existing: dict[str, Any] = dict(mv.artifacts.get("scenarios", {}))
        todo = [
            n
            for n in stack.metadata.scenarios
            # legacy entries (plain suitability path) lack the paired extrapolation mask
            if not only_missing or not isinstance(existing.get(n), dict)
        ]
        if not todo:
            return {"taxon_key": taxon_key, "projected": []}
        saved = elapid.load_object(str(self.artifacts.local_path(mv.artifacts["model"])))
        model = SimpleNamespace(
            model=saved["model"],
            predictors=saved["predictors"],
            mess_reference=saved["mess_reference"],
        )
        vdir = self.artifacts.local_path(
            str(self.artifacts.version_dir(taxon_key, mv.model_version))
        )
        for i, name in enumerate(todo):
            self.progress("scenarios", i / len(todo), f"Projecting v{mv.model_version} onto {name}")
            sc_stack = BioclimStack.open_version(self.settings.bioclim_root, version, name)
            existing[name] = self._scenario_artifacts(
                self.step("scenario_raster", project_scenario, model, sc_stack, vdir, name)
            )
        self.repo.update_model_artifacts(
            taxon_key, mv.model_version, {**mv.artifacts, "scenarios": existing}
        )
        self.progress("done", 1.0, f"Projected {len(todo)} scenario(s)")
        return {"taxon_key": taxon_key, "model_version": mv.model_version, "projected": todo}

    def _current_model(self, taxon_key: int) -> tuple[ModelVersionRecord, dict[str, Any]]:
        species = self.repo.get_species(taxon_key)
        if species is None or not species.model_version:
            raise ValueError(f"No trained model for taxon {taxon_key}")
        mv = self.repo.get_model_version(taxon_key, species.model_version)
        assert mv is not None
        saved = elapid.load_object(str(self.artifacts.local_path(mv.artifacts["model"])))
        return mv, saved

    # ======================================================== high-res region
    def project_hires(
        self, taxon_key: int, bbox: tuple[float, float, float, float]
    ) -> dict[str, Any]:
        """Project the current model at high resolution (e.g. 30″) inside `bbox`, with its
        own MESS extrapolation mask — no retraining. Replaces the species' previous region."""
        with self._exclusive(taxon_key):
            return self._project_hires(taxon_key, bbox)

    def _project_hires(
        self, taxon_key: int, bbox: tuple[float, float, float, float]
    ) -> dict[str, Any]:
        mv, saved = self._current_model(taxon_key)
        version = mv.metrics["reproducibility"]["bioclim_version_used"]
        resolution = self.hires_resolution(version)
        if resolution is None:
            raise ValueError(f"No high-resolution stack registered for {version}")
        stack = BioclimStack.open_version(
            self.settings.bioclim_root, version, hires=resolution
        ).subset(bbox)
        n_cells = stack.width * stack.height
        if n_cells == 0:
            raise ValueError("The region does not overlap the climate grid")
        if n_cells > self.settings.hires_max_cells:
            raise ValueError(
                f"Region has {n_cells:,} cells at {resolution} (max "
                f"{self.settings.hires_max_cells:,}); choose a smaller area"
            )
        self.progress(
            "hires_raster",
            0.1,
            f"Projecting v{mv.model_version} at {resolution} ({n_cells:,} cells)",
        )
        model = SimpleNamespace(
            model=saved["model"],
            predictors=saved["predictors"],
            mess_reference=saved["mess_reference"],
        )
        vdir = self.artifacts.local_path(
            str(self.artifacts.version_dir(taxon_key, mv.model_version))
        )
        out = self.step("hires_raster", project_scenario, model, stack, vdir, f"hires_{resolution}")
        t = stack.transform
        entry = {
            "resolution": resolution,
            "bbox": [t.c, t.f + stack.height * t.e, t.c + stack.width * t.a, t.f],
            **self._scenario_artifacts(out),
            "n_cells": n_cells,
            "created_ts": utcnow().isoformat(),
        }
        self.repo.update_model_artifacts(
            taxon_key, mv.model_version, {**mv.artifacts, "hires": entry}
        )
        self.progress("done", 1.0, f"High-resolution ({resolution}) region projected")
        return {"taxon_key": taxon_key, "model_version": mv.model_version, **entry}

    def hires_resolution(self, bioclim_version: str) -> str | None:
        """Finest high-resolution stack registered for a bioclim version, if any."""
        path = self.settings.bioclim_root / bioclim_version / METADATA_FILENAME
        return BioclimMetadata.load(path).finest_hires() if path.exists() else None

    # ======================================================== climate cross-check
    def crosscheck_climate(self, taxon_key: int) -> dict[str, Any]:
        """Refit the current model version's exact training set on the configured
        cross-check climate source (e.g. CHELSA) and store the comparison with the version."""
        with self._exclusive(taxon_key):
            return self._crosscheck_climate(taxon_key)

    def _crosscheck_climate(self, taxon_key: int) -> dict[str, Any]:
        alt_version = self.settings.crosscheck_bioclim_version
        if not alt_version:
            raise ValueError(
                "No cross-check climate stack configured (SDM_CROSSCHECK_BIOCLIM_VERSION)"
            )
        mv, saved = self._current_model(taxon_key)
        repro = mv.metrics["reproducibility"]
        if alt_version == repro["bioclim_version_used"]:
            raise ValueError(f"v{mv.model_version} was trained on {alt_version} itself")
        snapshot = repro.get("training_data")
        if not snapshot:
            raise ValueError(
                f"v{mv.model_version} predates training snapshots; retrain before cross-checking"
            )
        training = pd.read_parquet(self.artifacts.local_path(snapshot["path"]))
        alt_stack = BioclimStack.open_version(self.settings.bioclim_root, alt_version)
        cfg = replace(
            self.training_config,
            tune=False,
            feature_classes=mv.metrics["feature_classes"],
            beta_multiplier=mv.metrics["beta_multiplier"],
            seed=int(repro.get("training_config", {}).get("seed", self.training_config.seed)),
        )
        self.progress(
            "crosscheck", 0.1, f"Refitting v{mv.model_version} on {alt_version} predictors"
        )
        report = self.step(
            "crosscheck",
            climate_crosscheck,
            training,
            saved["model"],
            float(saved["threshold"]),
            mv.metrics,
            mv.model_type,
            alt_stack,
            cfg,
        )
        report["primary_bioclim_version"] = repro["bioclim_version_used"]
        report["created_ts"] = utcnow().isoformat()
        vdir = self.artifacts.local_path(
            str(self.artifacts.version_dir(taxon_key, mv.model_version))
        )
        path = vdir / f"crosscheck_{alt_version}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["path"] = self.artifacts.relative_uri(path)
        crosschecks = {**mv.artifacts.get("crosschecks", {}), alt_version: report}
        self.repo.update_model_artifacts(
            taxon_key, mv.model_version, {**mv.artifacts, "crosschecks": crosschecks}
        )
        self.progress("done", 1.0, f"Cross-check vs {alt_version}: {report['verdict']}")
        return {
            "taxon_key": taxon_key,
            "model_version": mv.model_version,
            "alt_bioclim_version": alt_version,
            "verdict": report["verdict"],
        }


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None
