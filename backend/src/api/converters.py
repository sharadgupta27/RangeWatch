"""Domain → API-schema conversion helpers."""

from __future__ import annotations

from typing import Any

from src.api.schemas import (
    JobOut,
    ModelSummary,
    ModelVersionDetail,
    ModelVersionOut,
    NativeRangeOut,
    SeverityResult,
    SpeciesSummary,
)
from src.domain import Job, ModelVersionRecord, NativeRange, SpeciesRecord
from src.modeling.native_range import validate_geojson_polygon
from src.persistence.repository import Repository


def _severity(mv: ModelVersionRecord | None) -> SeverityResult | None:
    return SeverityResult.model_validate(mv.severity) if mv and mv.severity else None


def species_summary(repo: Repository, sp: SpeciesRecord) -> SpeciesSummary:
    mv = repo.get_model_version(sp.taxon_key, sp.model_version) if sp.model_version else None
    sev = mv.severity if mv else None
    return SpeciesSummary(
        taxon_key=sp.taxon_key,
        scientific_name=sp.scientific_name,
        common_name=sp.common_name,
        thumbnail_url=sp.thumbnail_url,
        status=sp.status,
        n_occurrences_total=sp.n_occurrences_total,
        n_occurrences_native=sp.n_occurrences_native,
        n_pending_records=sp.n_pending_records,
        model_version=sp.model_version,
        model_trained_ts=sp.model_trained_ts,
        last_gbif_fetch_ts=sp.last_gbif_fetch_ts,
        retrain_needed=sp.retrain_needed,
        severity_score=sev["score_0_100"] if sev else None,
        model_type=mv.model_type if mv else None,
    )


def model_summary(repo: Repository, sp: SpeciesRecord) -> ModelSummary | None:
    if not sp.model_version:
        return None
    mv = repo.get_model_version(sp.taxon_key, sp.model_version)
    if mv is None:
        return None
    m = mv.metrics
    return ModelSummary(
        model_version=mv.model_version,
        model_type=mv.model_type,
        confidence_label=m["confidence_label"],
        trained_ts=mv.trained_ts,
        trigger=mv.trigger,
        auc_mean=m.get("auc_mean"),
        cbi_mean=m.get("cbi_mean"),
        tss_mean=m.get("tss_mean"),
        transferability_caveat=m["transferability_caveat"],
        severity=_severity(mv),
    )


def native_range_out(nr: NativeRange) -> NativeRangeOut:
    return NativeRangeOut(
        taxon_key=nr.taxon_key,
        geometry=validate_geojson_polygon(nr.geometry),
        status=nr.status,
        source=nr.source,
        note=nr.note,
        updated_ts=nr.updated_ts,
        confirmed_ts=nr.confirmed_ts,
    )


def job_out(job: Job) -> JobOut:
    return JobOut.model_validate(job)


def _version_fields(mv: ModelVersionRecord) -> dict[str, Any]:
    m = mv.metrics
    return {
        "model_version": mv.model_version,
        "model_type": mv.model_type,
        "confidence_label": m["confidence_label"],
        "trained_ts": mv.trained_ts,
        "trigger": mv.trigger,
        **{
            k: m.get(k)
            for k in ("auc_mean", "auc_std", "cbi_mean", "cbi_std", "tss_mean", "tss_std")
        },
        "n_presence": m["n_presence"],
        "n_background": m["n_background"],
        "n_cv_folds": m["n_cv_folds"],
        "feature_classes": m["feature_classes"],
        "beta_multiplier": m["beta_multiplier"],
        "severity_score": (mv.severity or {}).get("score_0_100"),
        "mlflow_run_id": mv.mlflow_run_id,
        "reproducibility": m["reproducibility"],
    }


def model_version_out(mv: ModelVersionRecord) -> ModelVersionOut:
    return ModelVersionOut.model_validate(_version_fields(mv))


def model_version_detail(mv: ModelVersionRecord) -> ModelVersionDetail:
    m = mv.metrics
    return ModelVersionDetail.model_validate(
        {
            **_version_fields(mv),
            "threshold": m["threshold"],
            "threshold_rule": m["threshold_rule"],
            "cv_method": m["cv_method"],
            "predictors": m["predictors"],
            "variable_importance": m["variable_importance"],
            "folds": m["folds"],
            "tuning": m["tuning"],
            "projection": m["projection"],
            "transferability_caveat": m["transferability_caveat"],
            "severity": mv.severity,
            "n_records_by_effective_label": m["n_records_by_effective_label"],
        }
    )
