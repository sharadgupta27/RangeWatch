"""FastAPI dependency providers (overridden in tests)."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from src.config import Settings, get_settings
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import Repository


class JobQueue(Protocol):
    def enqueue_pipeline(
        self, job_id: str, taxon_key: int, force_retrain: bool, full_resync: bool = False
    ) -> None: ...
    def enqueue_bulletin(self, job_id: str, taxon_key: int, model_version: int | None) -> None: ...
    def enqueue_scenarios(self, job_id: str, taxon_key: int) -> None: ...
    def enqueue_validation(
        self, job_id: str, taxon_keys: list[int] | None, train: bool
    ) -> None: ...


class CeleryJobQueue:
    """Enqueue by task name so the API process never imports the modeling stack."""

    def __init__(self, redis_url: str) -> None:
        from celery import Celery

        self._app = Celery("sdm", broker=redis_url, backend=redis_url)

    def enqueue_pipeline(
        self, job_id: str, taxon_key: int, force_retrain: bool, full_resync: bool = False
    ) -> None:
        self._app.send_task(
            "sdm.run_species_pipeline",
            kwargs={
                "job_id": job_id,
                "taxon_key": taxon_key,
                "force_retrain": force_retrain,
                "full_resync": full_resync,
            },
        )

    def enqueue_bulletin(self, job_id: str, taxon_key: int, model_version: int | None) -> None:
        self._app.send_task(
            "sdm.generate_bulletin",
            kwargs={"job_id": job_id, "taxon_key": taxon_key, "model_version": model_version},
        )

    def enqueue_scenarios(self, job_id: str, taxon_key: int) -> None:
        self._app.send_task(
            "sdm.project_scenarios", kwargs={"job_id": job_id, "taxon_key": taxon_key}
        )

    def enqueue_validation(self, job_id: str, taxon_keys: list[int] | None, train: bool) -> None:
        self._app.send_task(
            "sdm.run_validation_suite",
            kwargs={"job_id": job_id, "taxon_keys": taxon_keys, "train": train},
        )


class TaxonSearch(Protocol):
    def suggest_species(self, query: str, limit: int = 10) -> list: ...


class InatSearch(Protocol):
    def autocomplete(self, query: str, limit: int = 10) -> list: ...


def settings_dep() -> Settings:
    return get_settings()


@lru_cache
def repository_dep() -> Repository:
    from src.persistence.postgres_repository import PostgresRepository

    return PostgresRepository.from_url(get_settings().database_url)


@lru_cache
def queue_dep() -> JobQueue:
    return CeleryJobQueue(get_settings().redis_url)


def artifacts_dep() -> ArtifactStore:
    s = get_settings()
    return ArtifactStore(s.artifact_root, s.titiler_artifact_root)


@lru_cache
def gbif_search_dep() -> TaxonSearch:
    from src.connectors.gbif_client import GbifClient

    return GbifClient()


@lru_cache
def inat_search_dep() -> InatSearch | None:
    if not get_settings().inat_enabled:
        return None
    from src.connectors.inaturalist_client import InatClient

    return InatClient()
