"""MLflow model registry integration: every trained model gets a run, metrics and artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class ModelRegistry(Protocol):
    def log_model_version(
        self,
        taxon_key: int,
        model_version: int,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        artifact_paths: list[Path],
    ) -> str | None: ...


class MlflowModelRegistry:
    def __init__(self, tracking_uri: str | None) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def log_model_version(
        self,
        taxon_key: int,
        model_version: int,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        artifact_paths: list[Path],
    ) -> str | None:
        mlflow = self._mlflow
        mlflow.set_experiment(f"sdm-taxon-{taxon_key}")
        with mlflow.start_run(run_name=f"v{model_version}") as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})  # drop NaN
            mlflow.set_tags(
                {**tags, "taxon_key": str(taxon_key), "model_version": str(model_version)}
            )
            for p in artifact_paths:
                if p.exists():
                    mlflow.log_artifact(str(p))
            run_id = run.info.run_id
        try:
            mlflow.register_model(f"runs:/{run_id}/model.pkl", f"sdm_{taxon_key}")
        except Exception as exc:  # registry may be unavailable with a file store
            log.warning("MLflow model registration skipped: %s", exc)
        return run_id


class NullModelRegistry:
    """Used in tests / when MLflow is intentionally disabled."""

    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []

    def log_model_version(
        self,
        taxon_key: int,
        model_version: int,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        artifact_paths: list[Path],
    ) -> str | None:
        self.logged.append(
            {
                "taxon_key": taxon_key,
                "model_version": model_version,
                "params": params,
                "metrics": metrics,
                "tags": tags,
            }
        )
        return None
