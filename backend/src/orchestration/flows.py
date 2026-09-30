"""Prefect flows: the species asset DAG and the scheduled monitoring sweep.

    raw_occurrences → features → model → suitability_raster → bulletin

Each asset step of `SpeciesPipeline` runs as a named Prefect task (retries for network-bound
steps), so the Prefect UI shows asset freshness and failures per species run. Flows are
started by Celery workers: on demand from the API, and nightly via the Celery beat schedule
(`sdm.refresh_all_species` -> `refresh_all_species_flow`) for continuous monitoring. Beat is
the single scheduler; do not also add a Prefect deployment schedule, or sweeps run twice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from src.orchestration.factory import build_pipeline, get_repository, job_progress

T = TypeVar("T")

_RETRIES = {"raw_occurrences": 3}


def prefect_step(name: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """StepRunner that executes each asset step as a Prefect task."""
    # Inputs are DataFrames / raster handles / fitted models: never hash or persist them.
    t = task(
        fn,
        name=name,
        retries=_RETRIES.get(name, 0),
        retry_delay_seconds=60,
        persist_result=False,
        cache_policy=NO_CACHE,
    )
    return t(*args, **kwargs)


@flow(name="species-pipeline", log_prints=True)
def species_pipeline_flow(
    taxon_key: int,
    force_retrain: bool = False,
    job_id: str | None = None,
    full_resync: bool = False,
) -> dict[str, Any]:
    pipeline = build_pipeline(progress=job_progress(job_id), step=prefect_step)
    outcome = pipeline.run(taxon_key, force_retrain=force_retrain, full_resync=full_resync)
    get_run_logger().info("taxon %s → %s", taxon_key, outcome.decision)
    return outcome.to_dict()


@flow(name="bulletin")
def bulletin_flow(taxon_key: int, model_version: int | None = None) -> dict[str, str]:
    pipeline = build_pipeline()
    species = pipeline.repo.get_species(taxon_key)
    if species is None or not species.model_version:
        raise ValueError(f"No trained model for taxon {taxon_key}")
    assert pipeline.bulletin is not None
    return pipeline.bulletin.generate(taxon_key, model_version or species.model_version)


@flow(name="project-scenarios")
def scenarios_flow(taxon_key: int, job_id: str | None = None) -> dict[str, Any]:
    pipeline = build_pipeline(progress=job_progress(job_id), step=prefect_step)
    return pipeline.project_scenarios(taxon_key)


@flow(name="validation-suite")
def validation_flow(
    taxon_keys: list[int] | None = None, train: bool = True, job_id: str | None = None
) -> dict[str, Any]:
    from src.orchestration.validation_suite import run_suite

    report = run_suite(build_pipeline(progress=job_progress(job_id)), taxon_keys, train=train)
    return report["summary"]


@flow(name="refresh-all-species")
def refresh_all_species_flow() -> list[dict[str, Any]]:
    """Monitoring sweep: incremental check (and conditional retrain) for every species."""
    results = []
    for sp in get_repository().list_species():
        try:
            results.append(species_pipeline_flow(sp.taxon_key))
            # Keep "what-if" layers complete when scenarios were added to the bioclim layer.
            build_pipeline(step=prefect_step).project_scenarios(sp.taxon_key)
        except Exception as exc:  # one failing species must not stop the sweep
            get_run_logger().error("taxon %s failed: %s", sp.taxon_key, exc)
            results.append({"taxon_key": sp.taxon_key, "error": str(exc)})
    return results
