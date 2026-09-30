# CLAUDE.md

Guidance for Claude Code (or any AI coding agent) working in this repository. This project builds an **MLOps-driven species distribution modeling (SDM) and invasion-risk dashboard** with a **React frontend** and a **Python/FastAPI backend**. Read this file fully before making changes; it encodes architectural decisions, non-negotiable scientific constraints, and coding conventions that keep the project reproducible and consistent across sessions.

---

## Project Summary

A web dashboard where a user selects a species from GBIF/iNaturalist, and the system automatically:
1. Fetches occurrence data (GBIF primary, iNaturalist secondary).
2. Extracts bioclimatic features from a pre-cached global raster stack (WorldClim v2.1 / CHELSA).
3. Trains a MaxEnt-equivalent SDM (`elapid`) per species.
4. Projects native range and candidate invasion/expansion zones globally.
5. Persists occurrences, features, and model artifacts keyed by `taxon_key`, so repeat runs fetch **only new GBIF records** and retrain **only when warranted**.
6. Generates an automated PDF bulletin (maps, severity index, expansion outlook) per species.

This is an operationalized MLOps pipeline (data versioning → drift/delta detection → conditional retraining → model registry → automated reporting) layered on top of established ecological methods — it is **not** a reinvention of MaxEnt or bioclim science. See "Prior Art" below before proposing new modeling approaches.

**Frontend is a React SPA, not Streamlit/Dash.** Streamlit was considered and rejected for this project — it cannot deliver the visual polish, layout control, or map interaction performance (large point datasets, custom overlays, synchronized dual maps) this dashboard requires. The backend is a pure JSON/OpenAPI service; it has no knowledge of how the UI renders.

---

## Architecture (do not restructure without discussion)

```
React SPA (TanStack Router/Query/Table + MapLibre + deck.gl + shadcn/ui)
    → typed API client (generated from FastAPI's OpenAPI schema)
    → FastAPI backend → orchestration (Prefect/Dagster) + Celery/Redis for long jobs
    → connectors (pygbif, pyinaturalist) → feature extraction (rasterio/xarray)
    → modeling (elapid MaxEnt) → persistence (PostGIS + object storage + MLflow)
    → titiler (raster tiles) / pg_tileserv (vector tiles) → back to frontend map
    → bulletin generator (Jinja2 + WeasyPrint, server-rendered PDF)
```

Core separation of concerns, always preserve this:
- **Global static layer**: bioclim rasters, downloaded once, versioned (`bioclim_v1`, `bioclim_v2`, ...). Never re-download per species.
- **Per-species dynamic layer**: occurrences, extracted features, model, predictions — all keyed by GBIF `taxon_key`.
- **Frontend/backend boundary**: the React app never calls GBIF, iNaturalist, or reads raster files directly. All data reaches the browser as JSON (via FastAPI), vector tiles (via `pg_tileserv`), or raster tiles (via `titiler`). Do not add direct external-API calls from frontend code.

---

## Repository Layout

```
sdm-dashboard/
├── data/
│   ├── bioclim/                # cached global COG rasters, versioned
│   └── native_range_polygons/  # curated/editable reference polygons
├── backend/
│   ├── src/
│   │   ├── connectors/
│   │   │   ├── gbif_client.py
│   │   │   └── inaturalist_client.py
│   │   ├── features/
│   │   │   ├── raster_sampler.py
│   │   │   └── background_sampler.py
│   │   ├── modeling/
│   │   │   ├── maxent_trainer.py
│   │   │   ├── evaluation.py       # AUC, TSS, CBI, MESS
│   │   │   └── severity_index.py
│   │   ├── orchestration/
│   │   │   └── flows.py            # Prefect/Dagster DAGs
│   │   ├── bulletin/
│   │   │   ├── templates/
│   │   │   └── generator.py
│   │   └── api/
│   │       ├── main.py             # FastAPI app
│   │       └── routers/            # species, jobs, models, bulletin endpoints
│   ├── db/
│   │   └── schema.sql
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── routes/                 # TanStack Router file-based routes
│   │   ├── components/
│   │   │   ├── ui/                 # shadcn/ui primitives (owned in-repo)
│   │   │   ├── maps/                # MapLibre + deck.gl components
│   │   │   └── charts/              # Recharts/visx components
│   │   ├── api/                    # generated OpenAPI client + TanStack Query hooks
│   │   ├── store/                  # TanStack Store (ephemeral UI state only)
│   │   └── styles/                  # Tailwind config/theme
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
├── docker-compose.yml
└── README.md
```

When adding a new backend module, place it under the matching `backend/src/` subpackage. When adding a new frontend feature, place it under the matching `frontend/src/` subdirectory. Do not add top-level scripts outside this structure.

---

## Tech Stack (pinned choices — do not substitute without strong justification)

### Backend & Pipeline

| Concern | Library/Tool | Notes |
|---|---|---|
| GBIF access | `pygbif` | Use `occurrences.search` for incremental pulls, `occurrences.download` (DOI-backed) for first-time full fetches |
| iNaturalist access | `pyinaturalist` | `get_taxa` for search/autocomplete, `get_observations(quality_grade='research')` for supplementary data |
| Raster I/O | `rasterio`, `rioxarray`, `xarray` | All bioclim sampling goes through these; no manual GDAL calls |
| SDM engine | `elapid.MaxentModel` | scikit-learn-compatible; do not hand-roll a Maxent feature transform |
| Spatial DB | PostgreSQL + PostGIS | GiST index on `geom`, B-tree on `(taxon_key, ingested_ts)` |
| Model registry | MLflow | Every trained model gets a version, metrics, and artifact path logged |
| Orchestration | Prefect or Dagster | Pipeline = DAG of assets: `raw_occurrences → features → model → suitability_raster → bulletin` |
| Async job queue | Celery + Redis | Long-running downloads/training never block FastAPI request threads |
| API framework | **FastAPI** | Must expose a complete OpenAPI schema — this is the contract the frontend's typed client is generated from |
| Raster tile serving | **titiler** | Dynamic COG tiling for suitability/bioclim rasters — frontend never downloads full GeoTIFFs |
| Vector tile serving | **pg_tileserv** | Serves PostGIS occurrence/polygon data as vector tiles for the map, not raw GeoJSON dumps at low zoom |
| Bulletin | Jinja2 → WeasyPrint | Server-rendered PDF; not part of the frontend bundle |
| Bioclim source | WorldClim v2.1 (primary), CHELSA (optional cross-check) | Cache as multiband Cloud-Optimized GeoTIFF, 19 bands |
| Static-layer data versioning | **DVC** (S3 remote) | Only `data/bioclim/<version>/` — git tracks the `.dvc` pointers. Per-species data is *not* in DVC (PostGIS is its system of record; each model version stores a hashed training-data snapshot instead) |

### Frontend

| Concern | Library/Tool | Notes |
|---|---|---|
| App framework | React 18+ with Vite | Client-rendered SPA; consider TanStack Start later only if SSR/SEO becomes a requirement |
| Routing | **TanStack Router** | Fully type-safe; route params (`/species/:taxonKey`) drive data loading |
| Server-state/caching | **TanStack Query** | All API calls go through Query hooks — no ad hoc `fetch`/`useEffect` data loading |
| Data grids | **TanStack Table** | Model-lineage panel, occurrence lists, species registry table |
| Base map | **MapLibre GL JS** via `react-map-gl`/`react-maplibre` | Open-source vector tile rendering, no per-tile licensing cost |
| Large-point/raster overlays | **deck.gl** (`DeckGL` component layered on MapLibre) | Use for occurrence point clouds, hexbin aggregation, heatmaps, suitability bitmap overlays — never render thousands of DOM markers |
| Component/styling system | **shadcn/ui** + **Tailwind CSS** | Components are copied into `frontend/src/components/ui/` and owned in-repo — customize freely, do not treat as an opaque package |
| Charts | **Recharts** (default) / **visx** (custom severity gauge) | Declarative, React-native |
| Forms/validation | **TanStack Form** + **Zod** | Species search, native-range polygon editor, severity-weight config |
| API client typing | **openapi-typescript** | Regenerate on every backend schema change — never hand-write API types |
| Ephemeral UI state | **TanStack Store** or React context | Only for non-persisted state (selected layer, map viewport); everything else lives server-side |

**Do not introduce Streamlit, Dash, Gradio, or any other Python-rendered UI framework into this repository.** The UI layer is React-only.

---

## Non-Negotiable Scientific Constraints

These rules exist because the SDM/invasion-transferability literature is explicit about failure modes. Do not relax them for convenience, regardless of frontend framework.

1. **Always report the transferability caveat.** Native-range-only MaxEnt models show only moderate transferability to invaded ranges (mean AUC ≈ 0.7 in cross-continental studies). Never present a native-trained model's global projection as a confident invasion forecast without this caveat attached, both in the bulletin and in the frontend UI copy.
2. **Always compute and display a MESS (Multivariate Environmental Similarity Surface) or equivalent uncertainty layer** alongside any suitability projection extrapolated outside the training range. Low MESS = high extrapolation risk — this must reduce displayed confidence, not the severity score itself. In the frontend, this must be a toggleable deck.gl layer, not an optional afterthought.
3. **Prefer combined native+invaded training (Model B)** whenever the species already has documented introduced-range occurrences (`range_label = 'introduced'`). Only fall back to native-only extrapolation (Model A) when no invaded-range data exists, and label Model A outputs as lower-confidence in both the API response metadata and the UI.
4. **Use spatial block cross-validation, not random CV**, for model evaluation.
5. **Report Continuous Boyce Index (CBI) alongside AUC/TSS.**
6. **Never fully automate the native-range definition.** Always expose the assumed native-range polygon as user-editable in the frontend (e.g., a deck.gl editable-geometry layer) before training — do not silently trust an automated heuristic.
7. **Never present the severity index as a black-box number.** It is a documented weighted composite (`suitability`, `climate analogy`, `spread rate`, `ecological impact prior`) — weights must be configurable via the frontend settings form and explained in the bulletin's methods footnote.

---

## Incremental Fetch & Retrain Logic (reference implementation contract)

```python
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
```

This logic lives entirely in the backend/orchestration layer. The frontend's only responsibility is to poll/subscribe to job and registry status (via TanStack Query, `refetchInterval` or SSE) and reflect state changes (species status badges, "retrain pending" indicators) — it must never implement or duplicate this decision logic client-side.

Any change to the retrain-trigger threshold (currently 5% new-record delta, or new points falling outside the current suitability envelope) must be a documented, reviewed decision.

---

## Database Schema Conventions

- `species_registry`: one row per `taxon_key` (PK), tracks `last_gbif_fetch_ts`, `last_gbif_download_key`, `model_version`, `model_path`, `model_metrics` (jsonb), `retrain_needed`.
- `occurrences`: partitioned by `taxon_key`; PK is `gbif_id`/`inat_id`; includes `range_label` (`native`/`introduced`/`unknown`) and nullable `bio1..bio19` feature columns.
- Never mutate historical `occurrences` rows on retrain — append-only for occurrence data; only `species_registry` metadata rows are updated in place.
- All raster/model artifacts are referenced by path/URI in the DB, never stored as blobs in PostGIS.
- Occurrence and polygon data reach the frontend exclusively via `pg_tileserv` vector tiles — do not add an endpoint that dumps raw occurrence tables as JSON for map rendering (fine for the TanStack Table list views, not for map layers).

---

## Reproducibility Requirements

Every training run must log, at minimum:
- `bioclim_version_used`
- GBIF download key/DOI (cite per GBIF's citation guidelines)
- Pinned versions of `elapid`, `pygbif`, `pyinaturalist`, `rasterio`
- Random seeds used for background/pseudo-absence sampling and CV fold assignment
- The exact training set (`training_data.parquet`: presences + background with predictors) and
  the occurrence manifest with effective range labels, each with its SHA-256, stored per
  model version and logged to MLflow; plus the DVC hash of the bioclim version directory

This is required for every `model_version` entry in `species_registry.model_metrics`, and must be surfaced verbatim in the frontend's model-lineage `TanStack Table` panel — do not summarize away any of these fields in the UI.

---

## Prior Art (do not duplicate, do extend)

Closest existing tools, already reviewed — consult before proposing new modeling defaults or UI patterns:
- **Wallace EcoMod / Wallace 2** (R/Shiny): reproducible, GUI-guided SDM workflow. Reuse its validated Maxent feature-class/regularization defaults rather than re-deriving from scratch. Lacks: persistence, incremental retraining, iNaturalist integration, and a modern componentized UI.
- **BON in a Box – SDM indicator** (GEO BON): containerized MaxEnt pipeline with GBIF fetch, STAC environmental layers, uncertainty maps, DOI citation. Closest architectural analog for the per-run pipeline design. Lacks: species-keyed persistence, any dashboard UI.
- **biomod2**: ensemble SDM library, useful as a benchmarking cross-check, not a base for this project's ingestion/orchestration layer.

This project's distinct contribution is the **MLOps layer plus a production-grade React frontend**: continuous monitoring, incremental retraining, model versioning, dual-source ingestion, automated bulletin generation, and genuine visual/interaction polish — not new SDM algorithms.

---

## Coding Conventions

### Backend
- Python 3.11+, type-hinted, `ruff` for linting, `black` for formatting.
- All GBIF/iNaturalist API calls go through `backend/src/connectors/` wrappers — never call `pygbif`/`pyinaturalist` directly from modeling or API code.
- All raster sampling goes through `backend/src/features/raster_sampler.py` — never open bioclim GeoTIFFs ad hoc elsewhere.
- Any new model evaluation metric must be added to `backend/src/modeling/evaluation.py`, not inlined in `maxent_trainer.py`.
- Long-running jobs (downloads, training) must run through the orchestration layer (Prefect/Dagster flow or Celery task) — never block the FastAPI request thread.
- Every new/changed endpoint must keep the OpenAPI schema accurate (proper Pydantic response models) since the frontend's types are generated from it.

### Frontend
- TypeScript strict mode; no `any` in API-facing types — rely on the generated `openapi-typescript` client.
- All server data access goes through TanStack Query hooks in `frontend/src/api/` — no direct `fetch`/`axios` calls inside components.
- Map layers (deck.gl) are defined as composable layer-config functions, not inlined in page components, so they can be reused across the dual-map and bulletin-preview views.
- shadcn/ui components live in `frontend/src/components/ui/` and are edited directly when customization is needed — do not wrap them in another abstraction layer.
- Keep ephemeral UI state (selected layer, map viewport, active tab) out of TanStack Query cache; use TanStack Store or local component state instead.

### Testing
- Backend: unit tests for feature extraction and severity-index calculations under `backend/tests/`; mock external GBIF/iNaturalist API calls, never hit the live network in CI.
- Frontend: component tests for map layer configuration and TanStack Table columns; mock the generated API client, never call the real backend in unit tests.

---

## What NOT to Do

- Do not introduce Streamlit, Dash, Gradio, or any other Python-rendered dashboard framework — the frontend is React-only.
- Do not re-download the global bioclim raster stack per species run.
- Do not train on random CV folds for spatial occurrence data.
- Do not ship an invasion-risk map without an accompanying MESS/uncertainty layer, in both the frontend and the bulletin.
- Do not silently auto-finalize a native-range polygon without a user-review step in the UI.
- Do not introduce a new SDM algorithm/library without first checking whether `elapid` already supports the needed feature.
- Do not render large occurrence point sets (>10k points) as individual DOM markers or unaggregated `ScatterplotLayer` points at low zoom — use deck.gl aggregation layers instead.
- Do not hand-write TypeScript types for API responses — regenerate from the OpenAPI schema.
- Do not store secrets (API keys, DB credentials) in code — use environment variables / `.env` (git-ignored) on the backend, and never embed secrets in frontend bundles.
