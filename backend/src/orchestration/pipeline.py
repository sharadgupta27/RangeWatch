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

Alongside the contract, and never changing it:
  * explicit GBIF absences are kept as an evaluation-only reference dataset (they are not
    occurrences and never count towards the retrain delta);
  * records failing an excluded data-quality test (features/quality.py) are left out of
    training; every flag is stored in the version's quality_flags.parquet snapshot;
  * every new model version is scored on the species' reference datasets (uploaded surveys,
    GBIF absences), so successive versions can be compared on independent data;
  * presences of uploads with role "training" join the training set (after the same quality
    tests, minus records duplicating a GBIF/iNaturalist one). Adding or removing such an
    upload is a user edit like the native-range polygon: it sets `retrain_needed`;
  * optional sampling-bias handling (features/sampling_bias.py, off by default): presence
    observation weights (quality tier x local density) and a target-group effort covariate
    held constant for prediction. Both are logged per version and stored in the snapshot;
  * expert record annotations (U11) are applied after the quality tests (exclude / cultivated
    / verified / range-label overrides) and snapshotted per version; adding one sets
    `retrain_needed` like a polygon edit;
  * optional algorithms (U3 ensemble with pseudo-absences, §4.4 integrated model), all
    threshold rules, response curves, limiting-factor / agreement / change / survey-priority
    rasters, an inferred-absence check, and a regional study area (U2);
  * early-detection alerts (U10) on new records, classified by prior sampling effort (B10).
    The retrain decision above is unchanged unless the *proposed* effort-aware mode is
    switched on (settings.retrain_trigger_mode, pending review).
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
import shapely
from shapely.geometry import box, shape

from src.config import Settings
from src.connectors.gbif_client import (
    EFFORT_BASIS_OF_RECORD,
    GBIF_ABSENCES_DATASET_ID,
    download_citation,
    inat_citation,
)
from src.domain import (
    BIOCLIM_BANDS,
    UPLOAD_SOURCE_PREFIX,
    Alert,
    ModelVersionRecord,
    NativeRange,
    OccurrenceRecord,
    ReferenceDataset,
    ReferenceRecord,
    SpeciesRecord,
    SpeciesStatus,
    utcnow,
)
from src.features.annotations import (
    apply_exclusions,
    apply_range_labels,
    effective_annotations,
)
from src.features.background_sampler import (
    BackgroundSample,
    buffer_region,
    cell_dedupe,
    distance_to_nearest_km,
    sample_buffered_background,
    sample_completeness_absences,
    sample_disk_pseudo_absences,
    sample_profile_pseudo_absences,
    sample_target_group_background,
)
from src.features.bioclim_store import dvc_pointer_md5
from src.features.quality import (
    DEFAULT_EXCLUDE,
    QualityConfig,
    QualityResult,
    assess_quality,
    overlap_with_occurrences,
)
from src.features.quality_refs import QualityReferences
from src.features.raster_sampler import (
    METADATA_FILENAME,
    BioclimMetadata,
    BioclimStack,
    sample_bias_layer,
    sample_raster_file,
)
from src.features.sampling_bias import (
    EFFORT_COVARIATE,
    conditioning_value,
    effort_values,
    presence_weights,
)
from src.features.thinning import (
    VARIOGRAM_MAX_POINTS,
    autocorrelation_range_km,
    environmental_thin,
    max_retention_thin,
)
from src.modeling.crosscheck import climate_crosscheck
from src.modeling.evaluation import auc, independent_evaluation
from src.modeling.maxent_trainer import (
    TRANSFERABILITY_CAVEAT,
    TrainingConfig,
    TrainingData,
    select_model_type,
    train_model,
    training_presence_mask,
)
from src.modeling.native_range import (
    effective_range_labels,
    propose_native_range,
    validate_geojson_polygon,
)
from src.modeling.projection import (
    EFFORT_RASTER,
    GCM_ENSEMBLE_PREFIX,
    PRIORITY_RASTER,
    ScenarioOutputs,
    change_map,
    effort_grid,
    effort_zone_summary,
    gcm_ensemble,
    priority_map,
    project_model,
    project_scenario,
    same_grid,
    scenario_groups,
    write_effort_raster,
)
from src.modeling.severity_index import (
    SeverityConfig,
    SeverityInputs,
    compute_severity,
    spread_rate_cells_per_year,
)
from src.monitoring.alerts import (
    MAX_RECORDS_PER_ALERT,
    alert_summary,
    class_counts,
    deliver,
    effort_classes,
)
from src.persistence.artifact_store import ArtifactStore
from src.persistence.model_registry import ModelRegistry
from src.persistence.repository import Repository
from src.persistence.training_snapshot import (
    write_annotations_snapshot,
    write_quality_snapshot,
    write_training_snapshot,
)

log = logging.getLogger(__name__)
T = TypeVar("T")

TRACKED_PACKAGES = ("elapid", "pygbif", "pyinaturalist", "rasterio", "scikit-learn", "numpy")
INDEPENDENT_EVALUATION_FILE = "independent_evaluation.json"
GLOBAL_BBOX = (-180.0, -90.0, 180.0, 90.0)


SHARED_BIAS_DIR = "shared_bias"


def shared_bias_path(rank: str, group_key: int) -> str:
    """Artifact path of a target group's shared bias estimate (B9)."""
    return f"{SHARED_BIAS_DIR}/{rank.lower()}_{int(group_key)}.json"


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
    def search_absences(
        self,
        taxon_key: int,
        since: datetime | None = None,
        until: datetime | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ReferenceRecord]: ...


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
        prior = self.repo.load_occurrences(taxon_key) if self.settings.alerts_enabled else None
        n_new = self.append_to_occurrence_store(taxon_key, delta)
        self.update_watermark(taxon_key, delta_fetch_ts=self._last_fetch_started)
        early = (
            self._early_detection(species, delta, prior) if n_new and prior is not None else None
        )
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
            if early:
                out.details["early_detection"] = early
            return out
        out = self.flag_minor_update_pending(taxon_key, n_new, ratio)
        if early:
            out.details["early_detection"] = early
        return out

    def _needs_training(self, species: SpeciesRecord, force: bool) -> bool:
        return force or species.retrain_needed or species.model_version == 0

    @staticmethod
    def _trigger(species: SpeciesRecord, force: bool) -> str:
        if force:
            return "manual retrain request"
        if species.model_version == 0:
            return "initial training"
        return (
            "native range, training uploads, record annotations or study region edited / "
            "retrain flagged"
        )

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
        self.ingest_absences(taxon_key)
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
        self.ingest_absences(species.taxon_key, full=full)
        return delta

    def ingest_absences(self, taxon_key: int, full: bool = False) -> int:
        """Store the species' explicit GBIF absences (occurrenceStatus = ABSENT) in its
        evaluation-only reference dataset. They are not occurrences: they never enter
        training or the retrain delta. Failure never blocks the presence pipeline; the
        dataset keeps its own watermark, so missed absences are fetched next time."""
        if not self.settings.gbif_absences_enabled:
            return 0
        ds = self.repo.get_reference_dataset(taxon_key, GBIF_ABSENCES_DATASET_ID)
        started = utcnow()
        since = None if ds is None or full else ds.last_fetch_ts
        try:
            records = self.gbif.search_absences(taxon_key, since=since)
        except Exception as exc:
            log.warning("GBIF absence fetch failed (continuing): %s", exc)
            return 0
        if ds is None:
            ds = self.repo.save_reference_dataset(
                ReferenceDataset(
                    taxon_key=taxon_key,
                    dataset_id=GBIF_ABSENCES_DATASET_ID,
                    kind="gbif_absences",
                    name="GBIF explicit absences",
                    citation=(
                        "GBIF.org occurrence search, occurrenceStatus=ABSENT, "
                        f"taxonKey={taxon_key} (search API; not DOI-citable)"
                    ),
                    note="Survey records reporting the species as not found. An absence is "
                    "a site, not a grid cell: the species may still occur elsewhere in the "
                    "cell, so specificity computed from absences is a lower bound.",
                )
            )
        n = self.repo.append_reference_records(records)
        self.repo.save_reference_dataset(
            replace(ds, last_fetch_ts=started, **self._reference_counts(taxon_key, ds.dataset_id))
        )
        return n

    def _reference_counts(self, taxon_key: int, dataset_id: str) -> dict[str, int]:
        recs = self.repo.load_reference_records(taxon_key, dataset_id)
        status = recs["occurrence_status"] if not recs.empty else pd.Series(dtype=str)
        return {
            "n_present": int((status == "present").sum()),
            "n_absent": int((status == "absent").sum()),
        }

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
        """True if ≥ N new records fall in cells the current model classifies as unsuitable.

        With the proposed (pending review) effort-aware mode only records in cells the
        target group had already sampled count (B10); the default mode is the reviewed
        contract and ignores effort."""
        res = self._delta_suitability(species, delta)
        if res is None:
            return False
        suit, threshold, classes = res
        outside = ~np.isnan(suit) & (suit < threshold)
        if self.settings.retrain_trigger_mode == "effort_aware":
            outside &= classes != "discovery"
        return int(outside.sum()) >= self.settings.envelope_min_outside_points

    def _delta_suitability(
        self, species: SpeciesRecord, delta: list[OccurrenceRecord]
    ) -> tuple[np.ndarray, float, np.ndarray] | None:
        """Current suitability, threshold and B10 effort class at each delta record."""
        if not species.model_version or not delta:
            return None
        mv = self.repo.get_model_version(species.taxon_key, species.model_version)
        if mv is None or "suitability" not in mv.artifacts:
            return None
        path = self.artifacts.local_path(mv.artifacts["suitability"])
        if not path.exists():
            return None
        lon, lat = [r.longitude for r in delta], [r.latitude for r in delta]
        suit = sample_raster_file(path, lon, lat)
        threshold = float(mv.metrics.get("threshold", 0.0))
        effort_uri = mv.artifacts.get("effort")
        classes = effort_classes(
            self.artifacts.local_path(effort_uri) if effort_uri else None,
            lon,
            lat,
            self.settings.effort_sampled_min_records,
        )
        return suit, threshold, classes

    # ======================================================== early detection (U10)
    def _early_detection(
        self, species: SpeciesRecord, delta: list[OccurrenceRecord], prior: pd.DataFrame
    ) -> dict[str, Any] | None:
        """Alerts for new records outside the envelope or far from every known record.
        Never blocks ingestion and never changes the retrain decision."""
        try:
            return self._raise_alerts(species, delta, prior)
        except Exception as exc:
            log.exception("Early-detection check failed")
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _raise_alerts(
        self, species: SpeciesRecord, delta: list[OccurrenceRecord], prior: pd.DataFrame
    ) -> dict[str, Any] | None:
        taxon_key = species.taxon_key
        known = set(
            zip(prior["source"].astype(str), prior["record_id"].astype("int64"), strict=True)
        )
        alerted = {
            (r["source"], int(r["record_id"]))
            for a in self.repo.list_alerts(taxon_key, limit=500)
            for r in a.records
        }
        new = [
            r
            for r in delta
            if (r.source, int(r.record_id)) not in known
            and (r.source, int(r.record_id)) not in alerted
        ]
        if not new:
            return None
        lon = np.array([r.longitude for r in new], dtype="float64")
        lat = np.array([r.latitude for r in new], dtype="float64")
        res = self._delta_suitability(species, new)
        if res is not None:
            suit, threshold, classes = res
        else:
            suit, threshold = np.full(len(new), np.nan), None
            classes = np.full(len(new), "unknown", dtype=object)
        flags: dict[str, np.ndarray] = {}
        extra: dict[str, np.ndarray] = {}
        if threshold is not None:
            flags["outside_envelope"] = ~np.isnan(suit) & (suit < threshold)
        if len(prior):
            dist = distance_to_nearest_km(lon, lat, prior["longitude"], prior["latitude"])
            nr = self.repo.get_native_range(taxon_key)
            native = (
                shapely.contains_xy(shape(nr.geometry), lon, lat)
                if nr is not None
                else np.zeros(len(new), dtype=bool)
            )
            flags["outside_known_range"] = (dist > self.settings.alert_known_range_km) & ~native
            extra["distance_km"] = dist
        name = species.canonical_name or species.scientific_name
        subs = self.repo.list_subscriptions(taxon_key)
        out: dict[str, Any] = {"n_new_records_checked": len(new), "alerts": []}
        for kind, mask in flags.items():
            out[kind] = int(mask.sum())
            if not mask.any():
                continue
            idx = np.flatnonzero(mask)
            counts = class_counts(classes[idx])
            records = []
            for i in idx[:MAX_RECORDS_PER_ALERT]:
                r = new[i]
                records.append(
                    {
                        "source": r.source,
                        "record_id": int(r.record_id),
                        "longitude": r.longitude,
                        "latitude": r.latitude,
                        "event_date": r.event_date.isoformat() if r.event_date else None,
                        "suitability": None if np.isnan(suit[i]) else round(float(suit[i]), 4),
                        **(
                            {"distance_km": round(float(extra["distance_km"][i]), 1)}
                            if "distance_km" in extra
                            else {}
                        ),
                        "effort_class": str(classes[i]),
                    }
                )
            detail = (
                f"suitability < threshold {threshold:.3g} of model v{species.model_version}"
                if kind == "outside_envelope" and threshold is not None
                else f"> {self.settings.alert_known_range_km:g} km from any earlier record, "
                "outside the native range"
            )
            alert = self.repo.add_alert(
                Alert(
                    taxon_key=taxon_key,
                    kind=kind,  # type: ignore[arg-type]
                    n_records=len(idx),
                    records=records,
                    summary=alert_summary(kind, len(idx), name, counts, detail),
                    model_version=species.model_version or None,
                    effort_classes=counts,
                )
            )
            if subs:
                self.repo.update_alert_deliveries(
                    alert.alert_id, deliver(alert, subs, name, self.settings)
                )
            out["alerts"].append(str(alert.alert_id))
            out[f"{kind}_effort_classes"] = counts
        out["retrain_trigger_mode"] = self.settings.retrain_trigger_mode
        return out

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
            stack, region = self.training_stack(species)
            occ = self.step("features", self.extract_features, taxon_key, stack)
            result, data_info = self.step("model", self.fit_model, occ, nr, stack, region)
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
        try:
            evaluated = self._evaluate_references(taxon_key, versions=[version])
            if evaluated["datasets"]:
                out.details["independent_evaluation"] = evaluated["datasets"]
        except Exception as exc:  # evaluation on reference data never blocks publishing
            log.exception("Independent evaluation failed")
            out.details["independent_evaluation_error"] = f"{type(exc).__name__}: {exc}"
        if self.bulletin is not None:
            try:
                self.step("bulletin", self.bulletin.generate, taxon_key, version)
            except Exception as exc:  # the model is published even if PDF rendering fails
                log.exception("Bulletin generation failed")
                out.details["bulletin_error"] = f"{type(exc).__name__}: {exc}"
        self.progress("done", 1.0, f"Model v{version} published")
        return out

    def training_stack(self, species: SpeciesRecord) -> tuple[BioclimStack, dict[str, Any] | None]:
        """The climate grid a model is trained and projected on: the global stack, or (U2)
        the species' study region, optionally at a registered high resolution."""
        base = BioclimStack.open_version(self.settings.bioclim_root, self.settings.bioclim_version)
        region = species.study_region
        if not region:
            return base, None
        resolution = region.get("resolution")
        stack = (
            BioclimStack.open_version(
                self.settings.bioclim_root, self.settings.bioclim_version, hires=resolution
            )
            if resolution
            else base
        )
        stack = stack.subset(tuple(region["bbox"]))
        n_cells = stack.width * stack.height
        if n_cells == 0:
            raise ValueError("The study region does not overlap the climate grid")
        if n_cells > self.settings.hires_max_cells:
            raise ValueError(
                f"Study region has {n_cells:,} cells (max {self.settings.hires_max_cells:,}); "
                "choose a smaller area or a coarser resolution"
            )
        stack.hires_resolution = resolution  # type: ignore[attr-defined]
        t = stack.transform
        return stack, {
            "bbox": [float(v) for v in region["bbox"]],
            "resolution": resolution or stack.metadata.resolution,
            "grid_bbox": [t.c, t.f + stack.height * t.e, t.c + stack.width * t.a, t.f],
            "n_cells": n_cells,
            "cell_deg": abs(t.a),
        }

    def extract_features(self, taxon_key: int, stack: BioclimStack) -> pd.DataFrame:
        self.progress("features", 0.4, "Sampling bioclim predictors at occurrences")
        occ = self.repo.load_occurrences(taxon_key)
        feats = stack.sample(occ["longitude"], occ["latitude"], BIOCLIM_BANDS)
        missing = occ["bio1"].isna().to_numpy() & ~feats["bio1"].isna().to_numpy()
        # The stored bio* columns hold the version's own resolution, never a regional
        # high-resolution stack's values.
        if missing.any() and not getattr(stack, "hires_resolution", None):
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
        self,
        occ: pd.DataFrame,
        nr: NativeRange,
        stack: BioclimStack,
        region: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        cfg = self.training_config
        predictors = list(stack.metadata.selected_predictors or BIOCLIM_BANDS)
        clip = box(*region["bbox"]) if region else None
        algorithm: dict[str, Any] = {"requested": cfg.algorithm, "used": cfg.algorithm}
        survey: pd.DataFrame | None = None
        if cfg.algorithm == "integrated":
            survey, training_uploads = self._survey_records(nr.taxon_key, stack, predictors, clip)
            n_abs = 0 if survey is None else int((survey["y"] == 0).sum())
            n_pres = 0 if survey is None else int((survey["y"] == 1).sum())
            if n_abs < cfg.min_survey_absences or n_pres < 5:
                algorithm.update(
                    used="maxent",
                    fallback_reason=f"integrated model needs training uploads with ≥ "
                    f"{cfg.min_survey_absences} absences and ≥ 5 presences (have {n_abs} / "
                    f"{n_pres}); trained MaxEnt instead",
                )
                survey = None
        run_cfg = cfg if algorithm["used"] == cfg.algorithm else replace(cfg, algorithm="maxent")
        if run_cfg.algorithm != "integrated":
            occ, training_uploads = self._with_training_uploads(nr.taxon_key, occ, stack)
        n_outside_region = 0
        if clip is not None:
            inside = shapely.contains_xy(
                clip, occ["longitude"].to_numpy("float64"), occ["latitude"].to_numpy("float64")
            )
            n_outside_region = int((~inside).sum())
            occ = occ[inside].reset_index(drop=True)
        n_records = len(occ)
        occ = occ[~occ[predictors].isna().any(axis=1)].reset_index(drop=True)
        self.progress("model", 0.45, f"Data-quality tests on {len(occ):,} records")
        quality = self.assess_quality(occ, predictors, stack)
        quality.summary["n_outside_climate_grid"] = n_records - len(occ)
        quality_frame = quality.frame(occ)
        annotations = effective_annotations(self.repo.list_annotations(nr.taxon_key))
        excluded, annotation_info = apply_exclusions(occ, quality.excluded, annotations)
        occ = occ.assign(quality_tier=quality.tier)[~excluded].reset_index(drop=True)
        labels = effective_range_labels(occ, nr.geometry)
        labels, n_overrides = apply_range_labels(occ, labels, annotations)
        annotation_info["n_label_overrides"] = n_overrides
        model_type = select_model_type(labels)
        pres = occ[training_presence_mask(labels, model_type)].reset_index(drop=True)

        keep = cell_dedupe(pres["longitude"], pres["latitude"], stack.transform)
        pres = pres.iloc[keep].reset_index(drop=True)
        thin_seed = cfg.seed
        pres, thinning_info = self._thin(pres, stack, predictors, clip)
        self.progress("model", 0.5, f"Model {model_type}: {len(pres)} thinned presences")

        bg_seed = cfg.seed + 1
        explicit_bias = run_cfg.algorithm in ("integrated", "ppm")
        bias_method = (
            cfg.bias_covariate
            if cfg.bias_covariate != "none"
            else "target_group_effort" if explicit_bias else "none"
        )
        effort_filters, match_info = self._effort_filters(pres)
        bg, background_info = self.sample_background(
            nr.taxon_key,
            pres,
            stack,
            predictors,
            bg_seed,
            clip=clip,
            uniform_override=(run_cfg.algorithm if explicit_bias else None),
            effort_filters=effort_filters,
            match_info=match_info,
        )
        pa_seed = cfg.seed + 2
        pa, pa_info = (
            self._pseudo_absences(nr, pres, bg, stack, predictors, pa_seed, clip)
            if run_cfg.needs_pseudo_absences
            else (None, None)
        )

        weights, weight_info = presence_weights(
            pres["longitude"],
            pres["latitude"],
            pres["quality_tier"].tolist() if "quality_tier" in pres else None,
            cfg.presence_weighting,
            dict(cfg.tier_weights),
            cfg.density_neighbors,
        )
        presence_x = pres[predictors].reset_index(drop=True)
        background_x = bg.features[predictors].reset_index(drop=True)
        absence_x = None if pa is None else pa.features[predictors].reset_index(drop=True)
        bias, bias_info = self._bias_covariate(
            nr.taxon_key,
            pres,
            bg,
            stack,
            method=bias_method,
            extra=pa,
            effort_filters=effort_filters,
            match_info=match_info,
        )
        bias_columns: dict[str, float] = {}
        static, static_info = self._static_bias_layers(pres, bg, stack, pa)
        if static_info:
            bias_info["static_layers"] = static_info
        for col, cov in [*([(EFFORT_COVARIATE, bias)] if bias is not None else []), *static]:
            presence_x[col] = cov["presence"]
            background_x[col] = cov["background"]
            pres[col] = cov["presence"]
            if absence_x is not None:
                absence_x[col] = cov["extra"]
            bias_columns[col] = cov["value"]
        if bias is not None and run_cfg.algorithm == "integrated":
            bias_info.update(
                condition="estimated",
                conditioning_value=None,
                note="coefficient estimated jointly with the survey data (integrated "
                "model) and dropped for prediction",
            )
        offsets, ppm_info = (
            self._ppm_offsets(bias_columns, bias_info) if run_cfg.algorithm == "ppm" else ({}, None)
        )
        if ppm_info:
            algorithm["point_process"] = ppm_info
        if weights is not None:
            pres["sample_weight"] = weights
        data = TrainingData(
            predictors=predictors,
            presence_lon=pres["longitude"].to_numpy(),
            presence_lat=pres["latitude"].to_numpy(),
            presence_x=presence_x,
            background_lon=bg.lon,
            background_lat=bg.lat,
            background_x=background_x,
            background_method=bg.method,
            presence_weight=weights,
            bias_covariates=bias_columns,
            absence_lon=None if pa is None else pa.lon,
            absence_lat=None if pa is None else pa.lat,
            absence_x=absence_x,
            survey_lon=None if survey is None else survey["longitude"].to_numpy("float64"),
            survey_lat=None if survey is None else survey["latitude"].to_numpy("float64"),
            survey_x=None if survey is None else survey[predictors].reset_index(drop=True),
            survey_y=None if survey is None else survey["y"].to_numpy("float64"),
            offset_coefficients=offsets,
        )
        self.progress(
            "model",
            0.55,
            "Spatial block cross-validation + tuning"
            + ("" if run_cfg.algorithm == "maxent" else f" ({run_cfg.algorithm})"),
        )
        result = train_model(data, model_type, run_cfg)
        inferred = self._inferred_absence_check(nr, pres, stack, predictors, result, clip)
        info = {
            "labels": labels,
            "occ_valid": occ,
            "training_presences": pres,
            "training_data": data,
            "seeds": {
                "thinning": thin_seed,
                **(
                    {"thinning_variogram": cfg.seed + 4}
                    if "autocorrelation" in thinning_info
                    else {}
                ),
                "background_sampling": bg_seed,
                "cv_fold_assignment": cfg.seed,
                "maxent_random_state": cfg.seed,
                "permutation_importance": cfg.seed,
                "quality_outlier_fit": cfg.seed,
                **({"pseudo_absence_sampling": pa_seed} if pa is not None else {}),
                **(
                    {"inferred_absence_sampling": cfg.seed + 3}
                    if inferred.get("status") == "evaluated"
                    else {}
                ),
            },
            "n_records_by_effective_label": labels.value_counts().to_dict(),
            "background": background_info,
            "quality": quality.summary,
            "quality_frame": quality_frame,
            "training_uploads": training_uploads,
            "presence_weights": weight_info,
            "bias_covariates": {
                **bias_info,
                **({"importance": result.extra.get("bias_importance", {})} if bias_columns else {}),
            },
            "annotations": annotation_info,
            "annotations_frame": annotations,
            "algorithm": algorithm,
            "pseudo_absences": pa_info,
            "pseudo_absence_sample": pa,
            "survey": survey,
            "inferred_absences": inferred,
            "study_region": ({**region, "n_records_outside": n_outside_region} if region else None),
            "thinning": thinning_info,
        }
        return result, info

    # ------------------------------------------------------ thinning, bias helpers
    def _thin(
        self, pres: pd.DataFrame, stack: BioclimStack, predictors: list[str], clip: Any
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Thin the (cell-deduplicated) presences with the configured method (B3 / B4);
        the default is the fixed-distance greedy thinning at `thin_km`."""
        cfg = self.training_config
        info: dict[str, Any] = {"method": cfg.thin_method, "n_before": len(pres), "seed": cfg.seed}
        lon, lat = pres["longitude"].to_numpy("float64"), pres["latitude"].to_numpy("float64")
        if cfg.thin_method == "environmental":
            keep, extra = environmental_thin(
                pres[predictors], cfg.env_thin_bins, cfg.seed, cfg.env_thin_axes
            )
            info.update(extra)
        else:
            km = cfg.thin_km
            if cfg.thin_method == "autocorrelation" and len(pres):
                try:
                    region = buffer_region(lon, lat, cfg.background_buffer_km)
                    if clip is not None:
                        region = shapely.intersection(region, clip)
                    ref = sample_buffered_background(
                        lon,
                        lat,
                        stack,
                        VARIOGRAM_MAX_POINTS,
                        cfg.background_buffer_km,
                        cfg.seed + 4,
                        predictors,
                        region=region,
                    )
                    ac = autocorrelation_range_km(
                        ref.lon, ref.lat, ref.features[predictors], cfg.seed + 4
                    )
                    km = float(min(max(ac["range_km"], cfg.thin_km), cfg.thin_max_km))
                    info["autocorrelation"] = {
                        **ac,
                        "clamped_to_km": [cfg.thin_km, cfg.thin_max_km],
                        "capped": ac["range_km"] > cfg.thin_max_km,
                    }
                except Exception as exc:  # never block training on the variogram
                    log.warning("Autocorrelation range unavailable: %s", exc)
                    info["fallback_reason"] = (
                        f"autocorrelation range unavailable ({exc}); thinned at thin_km"
                    )
            keep, reps = max_retention_thin(lon, lat, km, cfg.seed, cfg.thin_reps)
            info.update(distance_km=round(km, 2), **reps)
        pres = pres.iloc[keep].reset_index(drop=True)
        info["n_after"] = len(pres)
        return pres, info

    def _effort_filters(self, pres: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
        """B1: target-group effort restricted to the presences' data sources (each basis of
        record with ≥ 5 % of the presences) and time window (5th–95th percentile year)."""
        cfg = self.training_config
        kwargs: dict[str, Any] = {}
        info: dict[str, Any] = {}
        if cfg.target_group_match_source:
            share = pres.get("basis_of_record", pd.Series(dtype=object)).dropna()
            share = share.value_counts(normalize=True)
            basis = sorted(
                str(b) for b, f in share.items() if b in EFFORT_BASIS_OF_RECORD and f >= 0.05
            )
            if basis:
                kwargs["basis_of_record"] = basis
                info["basis_of_record"] = basis
            else:
                info["basis_of_record_note"] = "no known basis of record: all sources used"
        if cfg.target_group_match_years:
            years = pd.to_datetime(
                pres.get("event_date", pd.Series(dtype=object)), errors="coerce"
            ).dt.year.dropna()
            if len(years) >= 5:
                window = (int(np.percentile(years, 5)), int(np.percentile(years, 95)))
                kwargs["years"] = window
                info["years"] = list(window)
            else:
                info["years_note"] = f"only {len(years)} dated presences: all years used"
        if info:
            info["reference"] = "Phillips et al. 2009; Botella et al. 2020"
        return kwargs, info

    def _static_bias_layers(
        self,
        pres: pd.DataFrame,
        bg: BackgroundSample,
        stack: BioclimStack,
        extra: BackgroundSample | None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
        """Registered static bias rasters (travel time, roads, …) at the training points,
        each held at its most-effort end (`bias_condition`) for prediction."""
        cfg = self.training_config
        cols: list[tuple[str, dict[str, Any]]] = []
        infos: list[dict[str, Any]] = []
        lon = np.r_[
            pres["longitude"].to_numpy("float64"),
            np.asarray(bg.lon, "float64"),
            np.asarray(extra.lon, "float64") if extra is not None else np.zeros(0),
        ]
        lat = np.r_[
            pres["latitude"].to_numpy("float64"),
            np.asarray(bg.lat, "float64"),
            np.asarray(extra.lat, "float64") if extra is not None else np.zeros(0),
        ]
        n_pres, n_bg = len(pres), len(bg.lon)
        for name in cfg.bias_static_layers:
            try:
                values, entry = sample_bias_layer(
                    self.settings.bioclim_root, stack.version, name, lon, lat
                )
            except (KeyError, FileNotFoundError, OSError) as exc:
                infos.append(
                    {
                        "name": name,
                        "status": "unavailable",
                        "reason": f"not registered for {stack.version} ({type(exc).__name__})",
                    }
                )
                continue
            missing = np.isnan(values)
            fill = float(np.nanmedian(values[n_pres : n_pres + n_bg])) if n_bg else 0.0
            values = np.where(missing, fill, values)
            ref = values[: n_pres + n_bg]
            if entry["direction"] == "increasing":
                value = conditioning_value(ref, cfg.bias_condition)
            else:
                value = -conditioning_value(-ref, cfg.bias_condition)
            col = f"bias_{name}"
            cols.append(
                (
                    col,
                    {
                        "presence": values[:n_pres],
                        "background": values[n_pres : n_pres + n_bg],
                        "extra": values[n_pres + n_bg :],
                        "value": value,
                    },
                )
            )
            infos.append(
                {
                    "name": name,
                    "covariate": col,
                    "status": "used",
                    "direction": entry["direction"],
                    "transform": entry.get("transform", "none"),
                    "condition": cfg.bias_condition,
                    "conditioning_value": round(float(value), 4),
                    "n_missing_filled": int(missing.sum()),
                    "presence_mean": round(float(values[:n_pres].mean()), 4) if n_pres else None,
                    "background_mean": (
                        round(float(values[n_pres : n_pres + n_bg].mean()), 4) if n_bg else None
                    ),
                    "citation": entry.get("citation"),
                    "sha256": entry.get("sha256"),
                }
            )
        return cols, infos

    def shared_bias_estimate(
        self, rank: str | None, group_key: int | None
    ) -> dict[str, Any] | None:
        """The stored shared bias estimate of a target group (B9), if any."""
        if not rank or group_key is None:
            return None
        path = self.artifacts.local_path(shared_bias_path(rank, group_key))
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _ppm_offsets(
        self, bias_columns: dict[str, float], bias_info: dict[str, Any]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Fixed coefficients of the point process (B8 offset / B9 shared bias)."""
        mode = self.training_config.ppm_bias
        info: dict[str, Any] = {"bias_mode_requested": mode, "bias_mode": mode}
        if EFFORT_COVARIATE not in bias_columns:
            info.update(
                bias_mode="none",
                note="no target-group effort available: fitted without an offset",
            )
            return {}, info
        if mode == "estimated":
            return {}, info
        if mode == "shared":
            group = bias_info.get("target_group") or {}
            shared = self.shared_bias_estimate(group.get("rank"), group.get("taxon_key"))
            coefs = {
                c: float(v)
                for c, v in ((shared or {}).get("coefficients") or {}).items()
                if c in bias_columns
            }
            if coefs:
                info["shared_bias"] = {
                    k: shared[k]
                    for k in ("estimate_id", "created_ts", "target_group", "coefficients")
                    if k in shared
                }
                info["shared_bias"]["n_species"] = len(shared.get("species", []))
                bias_info.update(
                    condition="offset",
                    note="coefficients fixed at the target group's shared bias estimate (B9) "
                    "and dropped for prediction",
                )
                return coefs, info
            info.update(
                bias_mode="offset",
                fallback_reason="no shared bias estimate for this target group: used the "
                "effort offset (coefficient 1)",
            )
        bias_info.update(
            condition="offset",
            note="log(effort + 1) as an offset with coefficient 1 (B8), dropped for prediction",
        )
        return {EFFORT_COVARIATE: 1.0}, info

    # ------------------------------------------------------ pseudo / inferred absences
    def _region_effort(self, taxon_key: int, bbox: tuple[float, ...], stack: BioclimStack) -> Any:
        """Target-group GBIF effort over a box (raises when unavailable)."""
        rank = self.training_config.target_group_rank.lower()
        group = self.gbif.get_taxon(taxon_key).higher_taxon(rank)
        if group is None:
            raise ValueError(f"taxon has no {rank} in the GBIF backbone")
        eff = self.gbif.fetch_effort_density(group[0], tuple(bbox), max_pixel_deg=abs(stack.res[0]))
        return eff, {"rank": rank, "taxon_key": group[0], "name": group[1]}

    def _pseudo_absences(
        self,
        nr: NativeRange,
        pres: pd.DataFrame,
        bg: BackgroundSample,
        stack: BioclimStack,
        predictors: list[str],
        seed: int,
        clip: Any,
    ) -> tuple[BackgroundSample, dict[str, Any]]:
        cfg = self.training_config
        strategy = cfg.pseudo_absence_strategy
        lon, lat = pres["longitude"].to_numpy("float64"), pres["latitude"].to_numpy("float64")
        region = buffer_region(lon, lat, cfg.background_buffer_km)
        if clip is not None:
            region = shapely.intersection(region, clip)
        info: dict[str, Any] = {
            "strategy": strategy,
            "n_requested": cfg.n_pseudo_absences,
            "buffer_km": cfg.background_buffer_km,
            "seed": seed,
            "reference": "Barbet-Massin et al. 2012",
        }
        self.progress("model", 0.535, f"Pseudo-absences ({strategy})")

        def disk() -> BackgroundSample:
            info["min_distance_km"] = cfg.pa_min_distance_km
            return sample_disk_pseudo_absences(
                lon,
                lat,
                stack,
                cfg.n_pseudo_absences,
                cfg.background_buffer_km,
                cfg.pa_min_distance_km,
                seed,
                predictors,
                region=region,
            )

        if strategy == "background":
            sample = bg
        elif strategy == "environmental_profile":
            info["envelope"] = "elapid.NicheEnvelopeModel, 2.5–97.5 percentiles, intersection"
            info["caveat"] = "excluding the presences' envelope can inflate accuracy"
            sample = sample_profile_pseudo_absences(
                lon,
                lat,
                pres[predictors],
                stack,
                cfg.n_pseudo_absences,
                cfg.background_buffer_km,
                seed,
                predictors,
                region=region,
            )
        elif strategy == "completeness":
            sample = None
            try:
                eff, group = self._region_effort(nr.taxon_key, region.bounds, stack)
                native = shapely.intersection(shape(nr.geometry), region)
                sample, counts = sample_completeness_absences(
                    eff.lon,
                    eff.lat,
                    eff.count,
                    lon,
                    lat,
                    stack,
                    cfg.n_pseudo_absences,
                    seed,
                    predictors,
                    cfg.completeness_min_records,
                    cfg.pa_min_distance_km,
                    region=native,
                )
                info.update(
                    counts,
                    target_group=group,
                    min_records=cfg.completeness_min_records,
                    min_distance_km=cfg.pa_min_distance_km,
                    zone="native range within the buffer (invaded-range absences may only mean "
                    "'not arrived yet')",
                    query=eff.query,
                )
                if len(sample.lon) < 50:
                    info["fallback_reason"] = (
                        f"only {len(sample.lon)} well-surveyed cells without the species"
                    )
                    sample = None
            except Exception as exc:
                log.warning("Completeness pseudo-absences unavailable: %s", exc)
                info["fallback_reason"] = f"effort data unavailable ({type(exc).__name__}: {exc})"
            if sample is None:
                sample = disk()
        else:
            sample = disk()
        info.update(method=sample.method, n=len(sample.lon))
        return sample, info

    def _inferred_absence_check(
        self,
        nr: NativeRange,
        pres: pd.DataFrame,
        stack: BioclimStack,
        predictors: list[str],
        result: Any,
        clip: Any,
    ) -> dict[str, Any]:
        """§3.2: score the model on inferred absences — well-surveyed native-range cells
        without the species. A plausibility check, not independent evaluation: the
        presences are the training presences."""
        cfg = self.training_config
        if not self.settings.inferred_absence_check:
            return {"status": "disabled"}
        info: dict[str, Any] = {
            "min_records": cfg.completeness_min_records,
            "min_distance_km": cfg.pa_min_distance_km,
            "zone": "confirmed native range",
            "note": "probable absences: cells where the target group is well recorded but the "
            "species never was; assumes comparable detectability. Presences are the training "
            "presences, so this is a plausibility check, not independent evaluation.",
        }
        try:
            native = shape(nr.geometry)
            if clip is not None:
                native = shapely.intersection(native, clip)
            if native.is_empty:
                return {**info, "status": "insufficient", "reason": "native range outside region"}
            eff, group = self._region_effort(nr.taxon_key, native.bounds, stack)
            sample, counts = sample_completeness_absences(
                eff.lon,
                eff.lat,
                eff.count,
                pres["longitude"],
                pres["latitude"],
                stack,
                2000,
                cfg.seed + 3,
                predictors,
                cfg.completeness_min_records,
                cfg.pa_min_distance_km,
                region=native,
            )
        except Exception as exc:
            return {**info, "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        info.update(counts, target_group=group)
        if len(sample.lon) < 20:
            return {
                **info,
                "status": "insufficient",
                "reason": f"{len(sample.lon)} well-surveyed native-range cells without the "
                "species (need ≥ 20)",
            }
        a = np.asarray(result.model.predict(sample.features[predictors]), "float64").ravel()
        p_ = np.asarray(result.model.predict(pres[predictors]), "float64").ravel()
        return {
            **info,
            "status": "evaluated",
            "n": len(a),
            "threshold": result.threshold,
            "specificity": round(float(np.mean(a < result.threshold)), 4),
            "mean_suitability": round(float(a.mean()), 4),
            "auc_training_presence_vs_inferred": round(auc(p_, a), 4),
        }

    def _survey_records(
        self, taxon_key: int, stack: BioclimStack, predictors: list[str], clip: Any
    ) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
        """Presence–absence records of training uploads for the integrated model."""
        frames: list[pd.DataFrame] = []
        summary: list[dict[str, Any]] = []
        for ds in self.repo.list_reference_datasets(taxon_key):
            if ds.role != "training":
                continue
            recs = self.repo.load_reference_records(taxon_key, ds.dataset_id)
            feats = stack.sample(recs["longitude"], recs["latitude"], predictors)
            ok = ~feats.isna().any(axis=1).to_numpy()
            if clip is not None:
                ok &= shapely.contains_xy(
                    clip,
                    recs["longitude"].to_numpy("float64"),
                    recs["latitude"].to_numpy("float64"),
                )
            y = (recs["occurrence_status"] == "present").to_numpy().astype("float64")
            summary.append(
                {
                    "dataset_id": ds.dataset_id,
                    "name": ds.name,
                    "file_name": ds.file_name,
                    "file_sha256": ds.file_sha256,
                    "citation": ds.citation,
                    "n_presences": int(y.sum()),
                    "n_overlap_excluded": 0,
                    "n_added": 0,
                    "n_absences_not_used": 0,
                    "used_as": "survey",
                    "n_survey_presences": int(y[ok].sum()),
                    "n_survey_absences": int((1 - y[ok]).sum()),
                }
            )
            if ok.any():
                frames.append(
                    pd.DataFrame(
                        {
                            "dataset_id": ds.dataset_id,
                            "record_id": recs["record_id"].astype("int64").to_numpy()[ok],
                            "longitude": recs["longitude"].astype("float64").to_numpy()[ok],
                            "latitude": recs["latitude"].astype("float64").to_numpy()[ok],
                            "y": y[ok],
                        }
                    ).join(feats[ok].reset_index(drop=True))
                )
        return (pd.concat(frames, ignore_index=True) if frames else None), summary

    def _bias_covariate(
        self,
        taxon_key: int,
        pres: pd.DataFrame,
        bg: BackgroundSample,
        stack: BioclimStack,
        method: str | None = None,
        extra: BackgroundSample | None = None,
        effort_filters: dict[str, Any] | None = None,
        match_info: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Target-group GBIF effort at the presences, background (and `extra`, e.g. pseudo-
        absence) points (log1p records per effort pixel) and the value it is held at for
        prediction. Unavailable effort data never blocks training: the model is then fitted
        without the covariate, and the reason is logged."""
        cfg = self.training_config
        method = method or cfg.bias_covariate
        info: dict[str, Any] = {"method": method}
        if method == "none":
            return None, info
        info.update(
            covariate=EFFORT_COVARIATE,
            transform="log1p(target-group records per effort pixel)",
            condition=cfg.bias_condition,
            reference="Warton et al. 2013; Fithian et al. 2015",
        )
        rank = cfg.target_group_rank.lower()
        ex_lon = np.asarray(extra.lon, "float64") if extra is not None else np.zeros(0)
        ex_lat = np.asarray(extra.lat, "float64") if extra is not None else np.zeros(0)
        lon = np.r_[
            pres["longitude"].to_numpy(dtype="float64"), np.asarray(bg.lon, "float64"), ex_lon
        ]
        lat = np.r_[
            pres["latitude"].to_numpy(dtype="float64"), np.asarray(bg.lat, "float64"), ex_lat
        ]
        try:
            group = self.gbif.get_taxon(taxon_key).higher_taxon(rank)
            if group is None:
                return None, {**info, "status": "unavailable", "reason": f"taxon has no {rank}"}
            group_key, group_name = group
            self.progress("model", 0.54, f"GBIF effort covariate for {rank} {group_name}")
            pad = abs(stack.res[0])
            bbox = (
                max(-180.0, float(lon.min()) - pad),
                max(-90.0, float(lat.min()) - pad),
                min(180.0, float(lon.max()) + pad),
                min(90.0, float(lat.max()) + pad),
            )
            eff = self.gbif.fetch_effort_density(
                group_key, bbox, max_pixel_deg=abs(stack.res[0]), **(effort_filters or {})
            )
        except Exception as exc:
            log.warning("Effort covariate unavailable: %s", exc)
            return None, {
                **info,
                "status": "unavailable",
                "reason": f"GBIF sampling-effort data unavailable ({type(exc).__name__}: {exc})",
            }
        if not eff.lon:
            return None, {
                **info,
                "status": "unavailable",
                "reason": f"no GBIF records of {rank} {group_name} in the region",
            }
        values = effort_values(eff.lon, eff.lat, eff.count, eff.pixel_deg, lon, lat)
        n_pres, n_bg = len(pres), len(bg.lon)
        value = conditioning_value(values[: n_pres + n_bg], cfg.bias_condition)
        info.update(
            status="used",
            conditioning_value=round(value, 4),
            target_group={
                "rank": rank,
                "taxon_key": group_key,
                "name": group_name,
                **({"matched": match_info} if match_info else {}),
            },
            n_records=int(eff.n_records),
            pixel_deg=eff.pixel_deg,
            map_zoom=eff.zoom,
            query=eff.query,
            fetched_ts=utcnow().isoformat(),
            presence_mean=round(float(values[:n_pres].mean()), 4) if n_pres else None,
            background_mean=(
                round(float(values[n_pres : n_pres + n_bg].mean()), 4) if n_bg else None
            ),
            background_note="background drawn uniformly in the buffer: the covariate models "
            "observer bias instead of a target-group background",
        )
        return {
            "presence": values[:n_pres],
            "background": values[n_pres : n_pres + n_bg],
            "extra": values[n_pres + n_bg :],
            "value": value,
        }, info

    def _with_training_uploads(
        self, taxon_key: int, occ: pd.DataFrame, stack: BioclimStack
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """Append the presences of uploads with role "training" to the occurrence frame, as
        rows with source "upload:<dataset_id>" and predictors sampled from the stack. Records
        duplicating a GBIF/iNaturalist record (the independence rule) are left out, so a
        survey also published to GBIF is not counted twice. Absences are not used: MaxEnt is
        a presence-background model."""
        datasets = [d for d in self.repo.list_reference_datasets(taxon_key) if d.role == "training"]
        summary: list[dict[str, Any]] = []
        frames: list[pd.DataFrame] = []
        for ds in datasets:
            recs = self.repo.load_reference_records(taxon_key, ds.dataset_id)
            is_pres = (recs["occurrence_status"] == "present").to_numpy()
            pres = recs[is_pres].reset_index(drop=True)
            dup = overlap_with_occurrences(
                pres["longitude"],
                pres["latitude"],
                pres["event_date"],
                occ["longitude"],
                occ["latitude"],
                occ["event_date"],
                distance_m=self.settings.overlap_distance_m,
                max_days=self.settings.overlap_max_days,
            )
            use = pres[~dup].reset_index(drop=True)
            summary.append(
                {
                    "dataset_id": ds.dataset_id,
                    "name": ds.name,
                    "file_name": ds.file_name,
                    "file_sha256": ds.file_sha256,
                    "citation": ds.citation,
                    "n_presences": len(pres),
                    "n_overlap_excluded": int(dup.sum()),
                    "n_added": len(use),
                    "n_absences_not_used": int((~is_pres).sum()),
                }
            )
            if use.empty:
                continue
            feats = stack.sample(use["longitude"], use["latitude"], BIOCLIM_BANDS)
            rows = pd.DataFrame(
                {
                    "source": UPLOAD_SOURCE_PREFIX + ds.dataset_id,
                    "record_id": use["record_id"].astype("int64"),
                    "longitude": use["longitude"].astype("float64"),
                    "latitude": use["latitude"].astype("float64"),
                    "event_date": use["event_date"],
                    "coordinate_uncertainty_m": pd.to_numeric(
                        use["coordinate_uncertainty_m"], errors="coerce"
                    ),
                    "basis_of_record": use["basis_of_record"],
                    "country_code": use["country_code"],
                    "range_label": "unknown",
                    "inat_crossref_id": None,
                    "issues": [[] for _ in range(len(use))],
                    "details": None,
                    "ingested_ts": use["ingested_ts"],
                }
            )
            rows = rows.join(feats.reset_index(drop=True))
            rows["bioclim_version"] = stack.version
            frames.append(rows)
        if frames:
            occ = pd.concat([occ, *frames], ignore_index=True)
        return occ, summary

    def quality_config(self) -> QualityConfig:
        exclude = self.settings.quality_exclude
        return QualityConfig(
            exclude=tuple(exclude) if exclude is not None else DEFAULT_EXCLUDE,
            baseline_start_year=self.settings.quality_baseline_start_year,
        )

    def assess_quality(
        self, occ: pd.DataFrame, predictors: list[str], stack: BioclimStack
    ) -> QualityResult:
        """Data-quality tests on records with climate data (see features/quality.py)."""
        refs = QualityReferences.load(
            self.settings.quality_reference_root, self.settings.quality_reference_version
        )
        return assess_quality(
            occ,
            predictors,
            cell_deg=abs(stack.res[0]),
            refs=refs,
            cfg=self.quality_config(),
            seed=self.training_config.seed,
        )

    def sample_background(
        self,
        taxon_key: int,
        pres: pd.DataFrame,
        stack: BioclimStack,
        predictors: list[str],
        seed: int,
        clip: Any = None,
        uniform_override: str | None = None,
        effort_filters: dict[str, Any] | None = None,
        match_info: dict[str, Any] | None = None,
    ) -> tuple[BackgroundSample, dict[str, Any]]:
        """Target-group background when configured and the effort data supports it, else
        uniform buffered background. The returned record (method, target group, effort
        query, any fallback reason) is logged with the model version. `clip` restricts the
        draw to a study region; `uniform_override` forces a uniform background (an integrated
        model's quadrature points must be uniform)."""
        cfg = self.training_config
        info: dict[str, Any] = {
            "requested_method": cfg.background_method,
            "buffer_km": cfg.background_buffer_km,
        }
        region = buffer_region(pres["longitude"], pres["latitude"], cfg.background_buffer_km)
        if clip is not None:
            region = shapely.intersection(region, clip)
            info["clipped_to_study_region"] = True
        override = cfg.bias_covariate if cfg.bias_covariate != "none" else uniform_override
        if override:
            # Modelling observer bias explicitly replaces mirroring it in the background.
            info["bias_covariate_override"] = override
        if cfg.background_method == "target_group" and not override:
            try:
                bg = self._target_group_background(
                    taxon_key, region, stack, predictors, seed, info, effort_filters, match_info
                )
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
            region=region,
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
        effort_filters: dict[str, Any] | None = None,
        match_info: dict[str, Any] | None = None,
    ) -> BackgroundSample | None:
        cfg = self.training_config
        rank = cfg.target_group_rank.lower()
        weighting = cfg.target_group_weighting
        group = self.gbif.get_taxon(taxon_key).higher_taxon(rank)
        if group is None:
            info["fallback_reason"] = f"taxon has no {rank} in the GBIF backbone"
            return None
        group_key, group_name = group
        self.progress("model", 0.52, f"Fetching GBIF sampling effort for {rank} {group_name}")
        effort = self.gbif.fetch_effort_density(
            group_key,
            tuple(region.bounds),
            max_pixel_deg=abs(stack.res[0]),
            **(effort_filters or {}),
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
            "weighting": weighting,
            **({"matched": match_info} if match_info else {}),
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
            method=f"target_group_{rank}"
            + ("" if weighting == "unique_cells" else f"_{weighting}"),
            counts=effort.count,
            weighting=weighting,
        )
        n_cells = bg.n_cells if bg.n_cells is not None else len(bg.lon)
        info["target_group"]["n_background_cells"] = n_cells
        if n_cells < cfg.min_target_group_cells:
            info["fallback_reason"] = (
                f"only {n_cells} cells with {rank} {group_name} records in the background "
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
                "extrapolation_reference": result.extrapolation_reference,
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
            data.model_columns,
            info["occ_valid"],
            info["labels"],
            extra_rows=self._extra_training_rows(data, info),
        )
        snapshot["quality_flags"] = write_quality_snapshot(vdir, info["quality_frame"])
        annotations: pd.DataFrame = info.get("annotations_frame", pd.DataFrame())
        if len(annotations):
            snapshot["annotations"] = write_annotations_snapshot(vdir, annotations)
        snapshot_meta = {
            name: f.describe(self.artifacts.relative_uri(f.path)) for name, f in snapshot.items()
        }

        self.progress(
            "suitability_raster", 0.7, "Projecting suitability + extrapolation diagnostics globally"
        )
        region = info.get("study_region")
        scenario_stacks = self._scenario_stacks(stack, region)
        occ_valid: pd.DataFrame = info["occ_valid"]
        proj = project_model(
            result,
            stack,
            vdir,
            shape(nr.geometry),
            occ_valid["longitude"].to_numpy(),
            occ_valid["latitude"].to_numpy(),
            scenario_stacks=scenario_stacks,
            limiting_factor=self.training_config.limiting_factor,
        )

        effort_path, effort_info, effort = self._effort_layer(
            taxon_key, info["background"], stack, vdir, proj.zones
        )
        change_path, change_info = self._change_previous(
            taxon_key, version, proj.suitability, result.threshold, vdir
        )
        priority_path, priority_info = self._survey_priority(proj, result.threshold, effort, vdir)

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
            candidate_consensus_ok_fraction=(s.get("extrapolation") or {}).get(
                "candidate_consensus_ok_fraction"
            ),
            candidate_consensus_majority_fraction=(s.get("extrapolation") or {}).get(
                "candidate_consensus_majority_fraction"
            ),
            cbi_mean=_finite(result.cv_summary.get("cbi_mean", float("nan"))),
        )
        severity = compute_severity(sev_inputs, sev_cfg)

        artifacts = {
            "model": self.artifacts.relative_uri(model_path),
            "suitability": self.artifacts.relative_uri(proj.suitability),
            "mess": self.artifacts.relative_uri(proj.mess),
            "zones": self.artifacts.relative_uri(proj.zones),
            "extrapolation": self.artifacts.relative_uri(proj.extrapolation),
            **{k: self.artifacts.relative_uri(path) for k, path in proj.diagnostics.items()},
            "training_data": snapshot_meta["training_data"]["path"],
            "occurrence_manifest": snapshot_meta["occurrence_manifest"]["path"],
            "quality_flags": snapshot_meta["quality_flags"]["path"],
            **({"effort": self.artifacts.relative_uri(effort_path)} if effort_path else {}),
            **{k: self.artifacts.relative_uri(path) for k, path in proj.products.items()},
            **(
                {"change_previous": self.artifacts.relative_uri(change_path)} if change_path else {}
            ),
            **({"priority": self.artifacts.relative_uri(priority_path)} if priority_path else {}),
            "scenarios": {
                name: self._scenario_artifacts(out, (proj.suitability, result.threshold))
                for name, out in proj.scenarios.items()
            },
        }
        artifacts["scenarios"].update(
            self._gcm_ensembles(artifacts["scenarios"], (proj.suitability, result.threshold), vdir)
        )
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
            "data_quality": {**info["quality"], "snapshot": snapshot_meta["quality_flags"]},
            "training_uploads": info.get("training_uploads", []),
            "presence_weights": info.get("presence_weights"),
            "bias_covariates": info.get("bias_covariates"),
            "algorithm": info.get("algorithm"),
            "pseudo_absences": info.get("pseudo_absences"),
            "annotations": {
                **(info.get("annotations") or {}),
                **(
                    {"snapshot": snapshot_meta["annotations"]}
                    if "annotations" in snapshot_meta
                    else {}
                ),
            },
            "study_region": region,
            "thinning": info.get("thinning"),
        }
        metrics: dict[str, Any] = {
            **{k: _finite(v) for k, v in result.cv_summary.items()},
            "train_auc": _finite(result.train_auc),
            "threshold": result.threshold,
            "threshold_rule": result.threshold_rule,
            "thresholds": result.thresholds,
            "cv_method": result.cv_method
            or f"spatial block CV (elapid.GeographicKFold, k={self.training_config.n_folds})",
            "algorithm": result.algorithm,
            **({"ensemble": result.extra["ensemble"]} if "ensemble" in result.extra else {}),
            **({"integrated": result.extra["integrated"]} if "integrated" in result.extra else {}),
            **(
                {"point_process": result.extra["point_process"]}
                if "point_process" in result.extra
                else {}
            ),
            "response_curves": result.response_curves,
            "inferred_absences": info.get("inferred_absences"),
            **({"change_previous": change_info} if change_info else {}),
            **({"survey_priority": priority_info} if priority_info else {}),
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
            "sampling_effort": effort_info,
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
                "quality_flags_sha256": snapshot_meta["quality_flags"]["sha256"],
                "quality_version": info["quality"]["version"],
                "quality_reference_version": (info["quality"].get("reference") or {}).get(
                    "version", "none"
                ),
                "target_group_weighting": self.training_config.target_group_weighting,
                "presence_weighting": self.training_config.presence_weighting,
                "bias_covariate": self.training_config.bias_covariate,
                "bias_covariate_status": (info.get("bias_covariates") or {}).get("status", "none"),
                "algorithm": result.algorithm,
                "thin_method": self.training_config.thin_method,
                "thin_distance_km": (info.get("thinning") or {}).get("distance_km", "n/a"),
                "threshold_rule": self.training_config.threshold_rule,
                "pseudo_absence_strategy": (info.get("pseudo_absences") or {}).get(
                    "strategy", "none"
                ),
                "n_records_annotated": (info.get("annotations") or {}).get(
                    "n_records_annotated", 0
                ),
                "study_region": json.dumps(region["bbox"]) if region else "global",
                **{f"pkg_{k}": v for k, v in reproducibility["package_versions"].items()},
            },
            metrics={
                **{
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                },
                "n_quality_excluded": info["quality"]["n_excluded"],
            },
            tags={"model_type": result.model_type, "confidence": result.confidence_label},
            artifact_paths=[
                model_path,
                snapshot["training_data"].path,
                snapshot["occurrence_manifest"].path,
                snapshot["quality_flags"].path,
                proj.suitability,
                proj.mess,
                proj.zones,
                proj.extrapolation,
                *proj.diagnostics.values(),
                *proj.products.values(),
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

    def _extra_training_rows(self, data: TrainingData, info: dict[str, Any]) -> pd.DataFrame | None:
        """Pseudo-absences (ensembles) and survey records (integrated) for the snapshot."""
        parts = []
        if data.absence_x is not None and data.absence_lon is not None:
            parts.append(
                pd.DataFrame(
                    {
                        "role": "pseudo_absence",
                        "longitude": data.absence_lon,
                        "latitude": data.absence_lat,
                    }
                ).join(data.absence_x[data.model_columns].reset_index(drop=True))
            )
        survey: pd.DataFrame | None = info.get("survey")
        if survey is not None and len(survey):
            parts.append(
                pd.DataFrame(
                    {
                        "role": np.where(survey["y"] == 1, "survey_presence", "survey_absence"),
                        "source": UPLOAD_SOURCE_PREFIX + survey["dataset_id"].astype(str),
                        "record_id": survey["record_id"].astype("int64"),
                        "longitude": survey["longitude"],
                        "latitude": survey["latitude"],
                    }
                ).join(survey[data.predictors].reset_index(drop=True))
            )
        return pd.concat(parts, ignore_index=True) if parts else None

    def _scenario_stacks(
        self, stack: BioclimStack, region: dict[str, Any] | None
    ) -> dict[str, BioclimStack]:
        """Scenario stacks on the training grid (regional models: the same box; regional
        high-resolution models have no scenario stacks at that resolution)."""
        if getattr(stack, "hires_resolution", None):
            return {}
        out = {}
        for name in stack.metadata.scenarios:
            sc = BioclimStack.open_version(self.settings.bioclim_root, stack.version, name)
            out[name] = sc.subset(tuple(region["bbox"])) if region else sc
        return out

    def _change_previous(
        self, taxon_key: int, version: int, suitability: Path, threshold: float, vdir: Path
    ) -> tuple[Path | None, dict[str, Any] | None]:
        """U6: gain / loss / stable between this version's binary map and the previous one's
        (each at its own threshold), when both are on the same grid."""
        if version <= 1:
            return None, None
        prev = self.repo.get_model_version(taxon_key, version - 1)
        base = {"from_version": version - 1, "to_version": version}
        if prev is None or "suitability" not in prev.artifacts:
            return None, {**base, "status": "unavailable", "reason": "no previous projection"}
        prev_path = self.artifacts.local_path(prev.artifacts["suitability"])
        if not prev_path.exists():
            return None, {**base, "status": "unavailable", "reason": "previous raster missing"}
        if not same_grid(prev_path, suitability):
            return None, {
                **base,
                "status": "unavailable",
                "reason": "the versions were projected on different grids (study region, "
                "resolution or bioclim version changed)",
            }
        path = vdir / "change_previous.tif"
        summary = change_map(
            prev_path, float(prev.metrics.get("threshold", 0.0)), suitability, threshold, path
        )
        return path, {**base, "status": "available", **summary}

    def _survey_priority(
        self, proj: Any, threshold: float, effort: tuple[np.ndarray, float] | None, vdir: Path
    ) -> tuple[Path | None, dict[str, Any] | None]:
        """U12: where field surveys add most (suitable, little-sampled, uncertain cells)."""
        if not self.settings.priority_layer_enabled:
            return None, None
        if effort is None:
            return None, {"status": "unavailable", "reason": "no sampling-effort layer"}
        unc_kind, unc_path = (
            ("agreement", proj.products["agreement"])
            if "agreement" in proj.products
            else (
                ("consensus", proj.diagnostics["consensus"])
                if "consensus" in proj.diagnostics
                else ("mess", proj.extrapolation)
            )
        )
        try:
            grid, pixel_deg = effort
            path = vdir / PRIORITY_RASTER
            summary = priority_map(
                proj.suitability, threshold, grid, pixel_deg, path, unc_path, unc_kind
            )
        except Exception as exc:  # a derived product never blocks publishing
            log.exception("Survey-priority layer failed")
            return None, {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        return path, {"status": "available", **summary}

    # ======================================================== sampling effort
    def _effort_layer(
        self,
        taxon_key: int,
        background: dict[str, Any],
        stack: BioclimStack,
        vdir: Path,
        zones: Path,
    ) -> tuple[Path | None, dict[str, Any], tuple[np.ndarray, float] | None]:
        """Global GBIF sampling effort of the target group: a map layer, plus the share of
        each range zone where the group has no records at all (there, an absence of
        presences is uninformative), plus the grid for the survey-priority layer. Never
        blocks publishing."""
        if not self.settings.effort_layer_enabled:
            return None, {"status": "disabled"}, None
        rank = self.training_config.target_group_rank.lower()
        try:
            tg = background.get("target_group") or {}
            if tg:
                group_key, group_name, rank = int(tg["taxon_key"]), tg["name"], tg["rank"]
            else:
                group = self.gbif.get_taxon(taxon_key).higher_taxon(rank)
                if group is None:
                    return (
                        None,
                        {
                            "status": "unavailable",
                            "reason": f"taxon has no {rank} in the GBIF backbone",
                        },
                        None,
                    )
                group_key, group_name = group
            self.progress(
                "suitability_raster", 0.85, f"Global GBIF sampling effort for {rank} {group_name}"
            )
            eff = self.gbif.fetch_effort_density(
                group_key, GLOBAL_BBOX, max_pixel_deg=abs(stack.res[0])
            )
            grid = effort_grid(eff.lon, eff.lat, eff.count, eff.pixel_deg)
            path = write_effort_raster(grid, eff.pixel_deg, vdir / EFFORT_RASTER)
            summary = effort_zone_summary(zones, stack, grid, eff.pixel_deg)
        except Exception as exc:
            log.warning("Sampling-effort layer unavailable: %s", exc)
            return None, {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}, None
        return (
            path,
            {
                "status": "available",
                "target_group": {"rank": rank, "taxon_key": group_key, "name": group_name},
                "n_records": int(eff.n_records),
                "max_records_per_pixel": int(grid.max()) if grid.size else 0,
                "pixel_deg": eff.pixel_deg,
                "map_zoom": eff.zoom,
                "n_tiles": eff.n_tiles,
                "query": eff.query,
                "fetched_ts": utcnow().isoformat(),
                **summary,
            },
            (grid, eff.pixel_deg),
        )

    # ======================================================== independent evaluation
    def evaluate_references(self, taxon_key: int, all_versions: bool = True) -> dict[str, Any]:
        """Re-check every reference dataset's independence and score model versions on it
        (all versions by default, so a new upload immediately compares them)."""
        with self._exclusive(taxon_key):
            versions = (
                [mv.model_version for mv in self.repo.list_model_versions(taxon_key)]
                if all_versions
                else None
            )
            self.progress("evaluation", 0.1, "Checking independence and scoring model versions")
            out = self._evaluate_references(taxon_key, versions=versions)
            self.progress("done", 1.0, f"Evaluated {len(out['model_versions'])} model version(s)")
            return out

    def _evaluate_references(
        self, taxon_key: int, versions: list[int] | None = None
    ) -> dict[str, Any]:
        datasets = self.repo.list_reference_datasets(taxon_key)
        if not datasets:
            return {"taxon_key": taxon_key, "datasets": [], "model_versions": []}
        occ = self.repo.load_occurrences(taxon_key)
        records: dict[str, pd.DataFrame] = {}
        independent: dict[str, np.ndarray] = {}
        overlapping: dict[str, np.ndarray] = {}
        for ds in datasets:
            recs = self.repo.load_reference_records(taxon_key, ds.dataset_id)
            records[ds.dataset_id] = recs
            if ds.kind == "upload":
                dup = overlap_with_occurrences(
                    recs["longitude"],
                    recs["latitude"],
                    recs["event_date"],
                    occ["longitude"],
                    occ["latitude"],
                    occ["event_date"],
                    distance_m=self.settings.overlap_distance_m,
                    max_days=self.settings.overlap_max_days,
                )
                overlap = {
                    "n_checked": len(recs),
                    "n_overlap": int(dup.sum()),
                    "fraction": round(float(dup.mean()), 4) if len(recs) else 0.0,
                    "distance_m": self.settings.overlap_distance_m,
                    "max_days": self.settings.overlap_max_days,
                    "n_occurrences_compared": len(occ),
                    "checked_ts": utcnow().isoformat(),
                    "rule": "a GBIF/iNaturalist record within the distance and, when both "
                    "are dated, within the day window; such records are left out of the "
                    "independent evaluation",
                }
                self.repo.save_reference_dataset(replace(ds, overlap=overlap))
            else:
                dup = np.zeros(len(recs), dtype=bool)
            overlapping[ds.dataset_id] = dup
            keep = ~dup
            if ds.role == "training":
                # Its presences trained the model; only its absences are independent of it.
                keep &= (recs["occurrence_status"] == "absent").to_numpy()
            independent[ds.dataset_id] = keep

        species = self.repo.get_species(taxon_key)
        if versions is None:
            versions = [species.model_version] if species and species.model_version else []
        done: list[int] = []
        for version in versions:
            mv = self.repo.get_model_version(taxon_key, version)
            if mv is None or "suitability" not in mv.artifacts:
                continue
            suit_path = self.artifacts.local_path(mv.artifacts["suitability"])
            if not suit_path.exists():
                continue
            # Versions trained before the extrapolation consensus have no such raster.
            extra_uri = mv.artifacts.get("extrapolation")
            extra_path = self.artifacts.local_path(extra_uri) if extra_uri else None
            bg_pred = self._background_predictions(mv, suit_path)
            threshold = float(mv.metrics.get("threshold", 0.0))
            entries: dict[str, Any] = {}
            for ds in datasets:
                recs = records[ds.dataset_id]
                keep = independent[ds.dataset_id]
                entries[ds.dataset_id] = {
                    "dataset_id": ds.dataset_id,
                    "name": ds.name,
                    "kind": ds.kind,
                    "role": ds.role,
                    "records_scored": "absences" if ds.role == "training" else "all",
                    "model_version": version,
                    "evaluated_ts": utcnow().isoformat(),
                    "n_records": len(recs),
                    "n_excluded_overlap": int(overlapping[ds.dataset_id].sum()),
                    "metrics": self._score(recs[keep], suit_path, extra_path, bg_pred, threshold),
                    "metrics_all_records": (
                        self._score(recs, suit_path, extra_path, bg_pred, threshold)
                        if overlapping[ds.dataset_id].any() and ds.role != "training"
                        else None
                    ),
                }
            vdir = self.artifacts.local_path(str(self.artifacts.version_dir(taxon_key, version)))
            (vdir / INDEPENDENT_EVALUATION_FILE).write_text(
                json.dumps(entries, indent=2, default=str), encoding="utf-8"
            )
            fresh = self.repo.get_model_version(taxon_key, version)
            assert fresh is not None
            self.repo.update_model_artifacts(
                taxon_key, version, {**fresh.artifacts, "independent_evaluation": entries}
            )
            done.append(version)
        return {
            "taxon_key": taxon_key,
            "datasets": [ds.dataset_id for ds in datasets],
            "model_versions": done,
        }

    def _background_predictions(self, mv: ModelVersionRecord, suit_path: Path) -> np.ndarray:
        """Suitability at the version's own background points (from its training snapshot),
        the reference distribution for the independent CBI."""
        snap = (mv.metrics.get("reproducibility") or {}).get("training_data")
        if not snap:
            return np.array([], dtype="float64")
        path = self.artifacts.local_path(snap["path"])
        if not path.exists():
            return np.array([], dtype="float64")
        bg = pd.read_parquet(path, columns=["role", "longitude", "latitude"])
        bg = bg[bg["role"] == "background"]
        return sample_raster_file(suit_path, bg["longitude"], bg["latitude"]).astype("float64")

    @staticmethod
    def _score(
        recs: pd.DataFrame,
        suit_path: Path,
        extra_path: Path | None,
        bg_pred: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        pres = recs[recs["occurrence_status"] == "present"]
        absn = recs[recs["occurrence_status"] == "absent"]
        p = sample_raster_file(suit_path, pres["longitude"], pres["latitude"])
        a = sample_raster_file(suit_path, absn["longitude"], absn["latitude"])
        ext = (
            sample_raster_file(extra_path, pres["longitude"], pres["latitude"]) == 1
            if extra_path is not None and extra_path.exists()
            else None
        )
        return independent_evaluation(p, a, bg_pred, threshold, presence_extrapolated=ext)

    # ======================================================== scenarios
    def _scenario_artifacts(
        self, out: ScenarioOutputs, current: tuple[Path, float] | None = None
    ) -> dict[str, Any]:
        """Artifacts of one scenario projection; with `current` (present-climate suitability
        and threshold) also the present → scenario change map (U6), same threshold."""
        entry: dict[str, Any] = {
            "suitability": self.artifacts.relative_uri(out.suitability),
            "extrapolation": self.artifacts.relative_uri(out.extrapolation),
            "extrapolated_land_fraction": out.extrapolated_land_fraction,
        }
        if out.consensus is not None:
            entry["consensus"] = self.artifacts.relative_uri(out.consensus)
            entry["diagnostics_land_fraction"] = out.diagnostics_land_fraction
        if current is not None and current[0].exists() and same_grid(current[0], out.suitability):
            name = out.suitability.name.removeprefix("suitability_")
            path = out.suitability.with_name(f"change_{name}")
            entry["change_summary"] = change_map(
                current[0], current[1], out.suitability, current[1], path
            )
            entry["change"] = self.artifacts.relative_uri(path)
        return entry

    def _gcm_ensembles(
        self, scenarios: dict[str, Any], current: tuple[Path, float], vdir: Path
    ) -> dict[str, Any]:
        """U7: one entry per SSP × period with ≥ 2 projected GCMs — mean suitability, the
        any-GCM extrapolation mask, the GCM agreement count and the present → ensemble change
        map, all at the model threshold. Entries are keyed `ensemble_<ssp>_<period>`."""
        projected = {
            name: entry
            for name, entry in scenarios.items()
            if isinstance(entry, dict)
            and entry.get("extrapolation")
            and not name.startswith(GCM_ENSEMBLE_PREFIX)
        }
        out: dict[str, Any] = {}
        for key, names in scenario_groups(projected).items():
            members = {
                n: (
                    self.artifacts.local_path(projected[n]["suitability"]),
                    self.artifacts.local_path(projected[n]["extrapolation"]),
                )
                for n in names
            }
            if not all(s.exists() and e.exists() for s, e in members.values()):
                continue
            self.progress("scenarios", 0.95, f"GCM ensemble {key.replace('_', ' ')}")
            try:
                ens = gcm_ensemble(members, current[1], vdir, key)
            except ValueError as exc:
                log.warning("GCM ensemble %s skipped: %s", key, exc)
                continue
            entry = self._scenario_artifacts(
                ScenarioOutputs(
                    ens.suitability,
                    ens.extrapolation,
                    ens.summary["extrapolated_any_gcm_land_fraction"],
                ),
                current,
            )
            entry["gcm_ensemble"] = {
                **ens.summary,
                "agreement": self.artifacts.relative_uri(ens.agreement),
            }
            out[GCM_ENSEMBLE_PREFIX + key] = entry
        return out

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
        groups = scenario_groups(
            n for n in [*existing, *todo] if not n.startswith(GCM_ENSEMBLE_PREFIX)
        )
        ensembles_missing = any(GCM_ENSEMBLE_PREFIX + k not in existing for k in groups)
        if not todo and not ensembles_missing:
            return {"taxon_key": taxon_key, "projected": []}
        saved = elapid.load_object(str(self.artifacts.local_path(mv.artifacts["model"])))
        model = SimpleNamespace(
            model=saved["model"],
            predictors=saved["predictors"],
            mess_reference=saved["mess_reference"],
            extrapolation_reference=saved.get("extrapolation_reference"),
        )
        vdir = self.artifacts.local_path(
            str(self.artifacts.version_dir(taxon_key, mv.model_version))
        )
        region = (mv.metrics.get("reproducibility") or {}).get("study_region")
        current = (
            self.artifacts.local_path(mv.artifacts["suitability"]),
            float(mv.metrics.get("threshold", 0.0)),
        )
        for i, name in enumerate(todo):
            self.progress("scenarios", i / len(todo), f"Projecting v{mv.model_version} onto {name}")
            sc_stack = BioclimStack.open_version(self.settings.bioclim_root, version, name)
            if region:
                sc_stack = sc_stack.subset(tuple(region["bbox"]))
            existing[name] = self._scenario_artifacts(
                self.step("scenario_raster", project_scenario, model, sc_stack, vdir, name),
                current,
            )
        # Ensembles are rebuilt whenever a member changes (stale ones are dropped first).
        existing = {k: v for k, v in existing.items() if not k.startswith(GCM_ENSEMBLE_PREFIX)}
        ensembles = self._gcm_ensembles(existing, current, vdir)
        existing.update(ensembles)
        self.repo.update_model_artifacts(
            taxon_key, mv.model_version, {**mv.artifacts, "scenarios": existing}
        )
        self.progress("done", 1.0, f"Projected {len(todo)} scenario(s)")
        return {
            "taxon_key": taxon_key,
            "model_version": mv.model_version,
            "projected": todo,
            "gcm_ensembles": sorted(ensembles),
        }

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
            extrapolation_reference=saved.get("extrapolation_reference"),
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
        if mv.metrics.get("algorithm", "maxent") != "maxent":
            raise ValueError(
                f"v{mv.model_version} is an {mv.metrics['algorithm']} model; the climate "
                "cross-check refits MaxEnt and supports MaxEnt versions only"
            )
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
