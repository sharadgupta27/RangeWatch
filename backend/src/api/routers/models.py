"""Model lineage endpoints (feeds the frontend's model-lineage table)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.converters import model_version_detail, model_version_out
from src.api.deps import repository_dep
from src.api.schemas import ModelVersionDetail, ModelVersionOut
from src.persistence.repository import Repository

router = APIRouter(prefix="/species/{taxon_key}/models", tags=["models"])
RepoDep = Annotated[Repository, Depends(repository_dep)]


@router.get("", response_model=list[ModelVersionOut], operation_id="listModelVersions")
def list_models(taxon_key: int, repo: RepoDep) -> list[ModelVersionOut]:
    """All model versions, newest first, with full reproducibility metadata."""
    if repo.get_species(taxon_key) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")
    return [model_version_out(mv) for mv in repo.list_model_versions(taxon_key)]


@router.get("/{model_version}", response_model=ModelVersionDetail, operation_id="getModelVersion")
def get_model(taxon_key: int, model_version: int, repo: RepoDep) -> ModelVersionDetail:
    mv = repo.get_model_version(taxon_key, model_version)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model version not found")
    return model_version_detail(mv)
