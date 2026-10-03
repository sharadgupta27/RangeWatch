"""FastAPI application. Pure JSON/OpenAPI service — it knows nothing about UI rendering."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import bulletin, jobs, layers, meta, models, species, validation
from src.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RangeWatch API",
        version="0.1.0",
        description=(
            "Species-keyed MLOps service for MaxEnt species distribution models: incremental "
            "GBIF/iNaturalist ingestion, conditional retraining, model lineage, map-layer "
            "descriptors (titiler / pg_tileserv) and automated bulletins."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in (
        meta.router,
        species.router,
        models.router,
        layers.router,
        bulletin.router,
        jobs.router,
        validation.router,
    ):
        app.include_router(r)
    return app


logging.basicConfig(level=logging.INFO)
app = create_app()
