# End-to-End Work Plan: Interactive Species Distribution & Invasion-Risk Dashboard

**Author context:** Designed for a reproducible-research, geospatial-data-science workflow (Python backend, React frontend, cloud-portable), targeting deployment as a web dashboard with incremental data refresh, model persistence, and automated bulletin generation. Framed as an **MLOps** engineering project — not a one-off analysis script.

---

## 1. Project Vision & Scope

A web dashboard where a user searches/selects a species (via GBIF and/or iNaturalist taxonomic search), and the system:

1. Fetches occurrence records for that species from GBIF (primary) and iNaturalist (secondary/complementary, via GBIF's iNaturalist-research-grade dataset or the iNaturalist API directly).
2. Extracts bioclimatic predictor values at occurrence locations from a **pre-downloaded, globally cached** bioclim raster stack (WorldClim v2.1 / CHELSA).
3. Trains a **MaxEnt-type** presence-background SDM for that species.
4. Projects the model globally to classify: (a) **native range** (where the species is currently established and climatically suitable), and (b) **potential invasive/expansion range** (climatically suitable areas outside the current known range).
5. Persists processed occurrence data, extracted features, and the trained model artifact keyed by species (taxon key), so that a repeat request only pulls **incremental/new GBIF records** since the last run, and retrains only if warranted.
6. Auto-generates a polished PDF/HTML **bulletin** per species with maps, invasion severity index, and expansion outlook narrative.

This is essentially an **operational MLOps pipeline for ecological niche modeling (ENM)**, exposed through a Python API and consumed by a dedicated **React** single-page application, backed by a lightweight orchestrated pipeline and a spatial database.

---

## 2. Related Work & Differentiation

Before building, it is worth positioning this project against existing tools, since a review of prior art shows the *scientific method* (MaxEnt + bioclim + GBIF) is very mature, but the *engineering pattern* you are proposing (a persistent, self-updating, MLOps-style service with a production-grade frontend) is not yet publicly available in one integrated product.

### 2.1 Existing Tools and Their Scope

| Tool | Core Capability | Data Sources | Architecture | Key Gap vs. This Project |
|---|---|---|---|---|
| **Wallace EcoMod** (CRAN `wallace` package) | GUI-guided, end-to-end SDM workflow: occurrence acquisition → environmental data → BIOCLIM/Maxent modeling → interactive mapping → exportable, reproducible R script [web:47][web:58] | GBIF, user-uploaded CSV | R/Shiny, single interactive session | No persistent model registry, no incremental data-fetch/retrain automation, no invasive-species-specific severity/expansion bulletin, R/Shiny UI (not a modern SPA) |
| **Wallace 2** (2023 redesign) | Adds multi-species sessions, climate-simulation/paleontological data modules, metadata tracking, cloud deployability [web:48][web:51] | GBIF, BIEN, user data | R/Shiny, modular "module contributor" architecture | Still session-based (analyst runs it); no "check-for-new-occurrences-and-retrain" loop; no iNaturalist connector; Shiny UI limits visual polish and componentization |
| **BON in a Box – SDM indicator** (GEO BON) | Pulls GBIF occurrences + STAC environmental layers, cleans data, generates pseudo-absences, runs MaxEnt via ENMeval, outputs a GBIF download **DOI**, suitability map, and **uncertainty map** [web:41] | GBIF, GEO BON STAC catalog | Containerized Web Processing Service (per-run indicator) | Closest architectural match to your "pipeline" concept — but region/run-scoped, not species-keyed and persistent; no dashboard UI at all |
| **biomod2** | R ensemble SDM library running up to 10 algorithms (including Maxent-equivalent) with ensemble forecasting [web:56][web:59] | User-supplied occurrence/environmental data | R package/library, not a hosted service | A modeling library, not a dashboard or data pipeline; no occurrence-fetching automation |
| **Galaxy Ecology / Wallace-in-Galaxy** | Wallace embedded inside the Galaxy bioinformatics workflow platform, with built-in GBIF fetch steps [web:53] | GBIF | Galaxy workflow engine | Reproducible batch execution, not a live, always-on, incrementally-updating service |
| **iMapInvasives** (NatureServe) | Cloud application for **tracking and managing** invasive species observations [web:52] | Manually submitted observation records | Cloud CRUD application | No SDM/Maxent modeling at all |
| Published single-species MaxEnt invasive studies (e.g., *Leptocybe invasa* [web:33], *Galinsoga parviflora* [web:38], *Sirex noctilio* [web:40], scarab beetles [web:37]) | Mature, validated science — AUC often > 0.9 in well-scoped regional invasive-risk studies | GBIF/field data, manually curated | Desktop `Maxent.jar` or R scripts, run once per paper | One-off academic analyses, not reusable software |
| Independently reported prototype (consultant LinkedIn profile) [web:43] | Describes an "automated SDM platform integrating Google Earth Engine, GBIF, and multiple ML architectures" | GBIF, GEE | Unknown/unpublished | Not open-sourced or documented; cannot be verified or reused |

### 2.2 What Remains Genuinely Unaddressed

1. **Persistent, species-keyed model lifecycle management** — no existing tool watches a species over time and conditionally retrains.
2. **Dual live-source ingestion (GBIF + iNaturalist)** — existing tools use GBIF only.
3. **Explicit native-vs-invasive dual modeling with a transparent severity index** — currently done manually, per paper.
4. **Automated, decision-ready bulletin generation** as a packaged deliverable.
5. **A modern, production-grade, componentized frontend** — every reviewed academic/ecology tool ships an R/Shiny or WPS interface, none use a contemporary React-based SPA with the visual and interaction quality expected of a commercial analytics product. This is a further differentiator worth naming explicitly, since Shiny apps are functional but visually and architecturally dated compared to a React + WebGL mapping stack.

### 2.3 Positioning Statement

Frame this project as **an MLOps operationalization layer, with a modern frontend, built on top of validated ecological methods** — reusing trusted components (GBIF via `pygbif`, Maxent-equivalent modeling via `elapid`, WorldClim/CHELSA bioclim layers, MESS-based transferability checks) rather than reinventing the underlying science, while contributing the missing engineering and UX layer: continuous monitoring, incremental retraining, model versioning, automated communication of results, and a genuinely polished, interactive interface. Cite Wallace 2 [web:48] and the BON in a Box SDM indicator [web:41] explicitly as the closest prior art in any write-up.

---

## 3. System Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                     FRONTEND — React SPA                          │
│  TanStack Router (routing) + TanStack Query (server-state/cache)  │
│  + TanStack Table (data grids) + MapLibre GL JS + deck.gl (maps)  │
│  + shadcn/ui + Tailwind CSS (component system) + Recharts/Visx    │
└───────────────┬───────────────────────────────────┬───────────────┘
                │ REST/JSON (OpenAPI-typed client)    │ SSE/WebSocket
┌───────────────▼───────────────┐   ┌────────────────▼─────────────┐
│      API LAYER — FastAPI       │   │   TASK QUEUE (async jobs)    │
│  Pydantic schemas, OpenAPI     │   │  Celery + Redis, or Prefect  │
│  spec auto-consumed by         │   │  deployments triggered by    │
│  frontend's typed client       │   │  API calls                   │
└───────┬───────────┬────────────┘   └──────────┬────────────────┘
        │           │                            │
┌───────▼───┐  ┌────▼─────┐              ┌───────▼────────┐
│ GBIF/iNat  │  │ Bioclim   │              │ Model Training  │
│ Connector  │  │ Feature   │              │ (elapid MaxEnt) │
│ (pygbif,   │  │ Extractor │              │ + evaluation    │
│ pyinat)    │  │ (rasterio)│              │ (AUC, CBI, TSS) │
└───────┬───┘  └────┬─────┘              └───────┬────────┘
        │           │                            │
┌───────▼───────────▼────────────────────────────▼────────────────┐
│                    PERSISTENCE LAYER                              │
│  PostgreSQL + PostGIS (occurrences, metadata, run log)            │
│  Object storage (raster stack, model .pkl, GeoTIFFs) served as    │
│  vector/raster tiles (pg_tileserv / titiler) to the frontend map  │
│  Parquet/GeoParquet cache for feature tables                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Bulletin Generator      │
                    │  (WeasyPrint / ReportLab │
                    │   + server-rendered      │
                    │   matplotlib/plotly)     │
                    └─────────────────────────┘
```

**Design principle:** separate the *global static layer* (bioclimatic rasters, downloaded once, versioned) from the *per-species dynamic layer* (occurrences, features, model, predictions), so re-runs are cheap and incremental. The React frontend is a **pure consumer** of a typed FastAPI contract — it never touches GBIF, iNaturalist, or raster files directly; all data access is mediated by the backend and served as GeoJSON, vector tiles, or Cloud-Optimized GeoTIFF tile endpoints.

---

## 4. Technology Stack

### 4.1 Frontend (React-based; replaces earlier Streamlit consideration)

Streamlit and Dash are excellent for internal analyst tools but are visually constrained and hard to make feel like a polished product — component styling is limited, layout control is coarse, and map interaction performance degrades with larger point datasets. For a dashboard meant to look and feel like a modern analytics product, a dedicated **React SPA** is the right choice.

| Concern | Recommended Tool | Rationale |
|---|---|---|
| App framework | **React 18+** with **Vite** (or **TanStack Start** if SSR/streaming is desired later) | Vite gives the fastest dev loop for a client-rendered dashboard; TanStack Start is the natural upgrade path if SEO/SSR is ever needed [web:68] |
| Routing | **TanStack Router** | Fully type-safe client-side routing; integrates natively with TanStack Query for route-level data loading [web:64][web:68] |
| Server-state & caching | **TanStack Query** | Handles caching, background refetch, and stale-while-revalidate for all API calls (species search, occurrence status, model metrics); avoids hand-rolled fetch/loading-state logic [web:62][web:69] |
| Data grids/tables | **TanStack Table** | Headless table logic (sorting, filtering, pagination) for the model-lineage panel, occurrence lists, and species registry views — full control over pixel-level styling [web:68][web:71] |
| Base map rendering | **MapLibre GL JS** via `react-map-gl`/`react-maplibre` | Open-source, no per-tile licensing cost, GPU-accelerated vector tiles; the current community consensus "best React map library" for 2026 [web:61][web:74] |
| Large-point/raster overlay rendering | **deck.gl** (`DeckGL` React component, layered on MapLibre via `MapboxOverlay`/`MapLibreOverlay`) | Purpose-built for rendering 100k+ occurrence points, hexbin aggregations, and heatmaps at GPU speed without DOM marker overhead [web:63][web:65][web:74] |
| Component/styling system | **shadcn/ui** + **Tailwind CSS** | Accessible, unstyled Radix-based primitives fully owned in-repo (not an opaque npm dependency), giving full design control for a "beautiful" custom look |
| Charts (severity gauge, variable importance, timelines) | **Recharts** or **visx** | Declarative, React-native charting; visx for fully custom gauge/severity visuals if Recharts presets are too generic |
| Forms/validation | **TanStack Form** + **Zod** | Type-safe validation for species search, native-range polygon editing, and severity-weight configuration forms |
| API client typing | **openapi-typescript** + generated fetch client, consumed by TanStack Query hooks | Keeps frontend types in lockstep with FastAPI's Pydantic schemas — no manual type drift |
| State for map/UI (non-server) | **TanStack Store** or React context, kept minimal | Only for ephemeral UI state (selected layer, active map bounds); all persisted data stays server-side |

### 4.2 Backend & Data Pipeline

| Layer | Recommended Tool | Rationale |
|---|---|---|
| Species search | **pygbif** `species.name_suggest` / `name_backbone`; **pyinaturalist** `get_taxa` | Resolve free-text to a canonical `taxonKey` / `taxon_id` |
| Occurrence retrieval | **pygbif** `occurrences.search` (quick, <100k records) and `occurrences.download` (bulk, DOI-backed) [web:2][web:9]; **pyinaturalist** `get_observations` for supplementary/fresher records [web:26] | GBIF download API is the authoritative, citable route; direct search API is fine for incremental small pulls |
| Bioclim data | **WorldClim v2.1** (30s/2.5m/5m/10m resolution GeoTIFFs) [web:21][web:28] or **CHELSA** (1 km, better in complex terrain) | Store once globally, reused across all species |
| Raster I/O & extraction | **rasterio**, **rioxarray**, **xarray**, **geopandas** | Point-in-raster sampling, reprojection, tiling |
| Raster/vector tile serving to frontend | **titiler** (dynamic COG tiling) + **pg_tileserv** (PostGIS vector tiles) | Lets the React/MapLibre frontend request map tiles directly by URL instead of downloading full GeoTIFFs or GeoJSON payloads client-side |
| SDM engine | **elapid** (`MaxentModel`) [web:15], fallback **maxnet** or scikit-learn ensemble for benchmarking | Elapid replicates Maxent features natively in Python, sklearn-compatible API |
| Background/pseudo-absence sampling | **elapid** `sample_bias_file`, target-group background, or buffer-based random background | Reduces sampling bias inherent to opportunistic citizen-science data |
| Model evaluation | AUC, TSS, Continuous Boyce Index (CBI) [web:20] | CBI is presence-only-appropriate, unlike plain AUC |
| API layer | **FastAPI** | Auto-generated OpenAPI spec consumed by the typed React client; native async support for long-running job status polling |
| Spatial DB | **PostgreSQL + PostGIS** | Efficient spatial indexing of occurrence points, incremental querying by `taxonKey` + last-fetch timestamp |
| Object/raster storage | **S3-compatible bucket** (MinIO for self-hosted) + **Cloud-Optimized GeoTIFF (COG)** | COGs allow partial/tiled reads for both `titiler` and direct analysis |
| Model artifact storage | **joblib/pickle** + **MLflow** model registry | Versioning, reproducibility, rollback |
| Orchestration | **Prefect** or **Dagster** | Handles the "check-for-new-data → retrain-if-needed" DAG |
| Task queue for long jobs | **Celery + Redis** | Keeps the FastAPI request thread responsive during downloads/training; frontend polls job status via TanStack Query or subscribes via SSE/WebSocket |
| Bulletin generation | **Jinja2** → **WeasyPrint**, or **ReportLab** | Reproducible, styled PDF report per species |
| Containerization/deploy | **Docker Compose** (dev) → **Kubernetes/Cloud Run** (prod); frontend built as static assets served via **Nginx** or a CDN | Portable across Hamburg university infra or cloud |

---

## 5. Data Layer Design

### 5.1 Global Bioclimatic Raster Store (downloaded once)
- Download all 19 WorldClim v2.1 BIO variables at a chosen resolution (recommend **2.5 arc-min** ≈ 4.5 km for global-scale invasion screening; keep **30 arc-sec** ≈ 1 km as an optional high-res tile set for regional zoom) [web:21][web:28].
- Optionally add CHELSA equivalents for cross-validation in topographically complex regions.
- Store as a stacked, **Cloud-Optimized GeoTIFF (COG)** multiband raster (`bioclim_global_2.5m.tif`, 19 bands) plus a companion metadata JSON (resolution, CRS = EPSG:4326, source, download date, checksum).
- Serve via **titiler** so the frontend map can request `{z}/{x}/{y}` tiles dynamically instead of shipping full rasters to the browser.
- Precompute a **variance inflation factor (VIF) / Pearson correlation matrix** across the 19 variables once, globally (typically 6–8 of 19 retained after VIF < 5 filtering) — improves transferability per [web:25].
- Version this layer (`bioclim_v1`, `bioclim_v2`...) — never re-extract per species without checking the raster version tag.

### 5.2 Species Metadata & Run Registry (PostGIS table `species_registry`)

| Column | Type | Purpose |
|---|---|---|
| `taxon_key` (PK) | integer | GBIF backbone taxon key |
| `scientific_name` | text | Canonical name |
| `inat_taxon_id` | integer | Cross-reference to iNaturalist |
| `last_gbif_fetch_ts` | timestamptz | Watermark for incremental pulls |
| `last_gbif_download_key` | text | GBIF download DOI/key for citation |
| `n_occurrences_native` | int | Count in currently defined native range |
| `n_occurrences_total` | int | Total stored |
| `bioclim_version_used` | text | Ties model to a raster version |
| `model_version` | int | Auto-incremented on retrain |
| `model_path` | text | Path/URI to serialized model |
| `model_trained_ts` | timestamptz | Last training time |
| `model_metrics` | jsonb | AUC, CBI, TSS, n_train, n_test |
| `retrain_needed` | boolean | Flag set by the incremental-check job |

### 5.3 Occurrence Store (PostGIS table `occurrences`, partitioned by `taxon_key`)

| Column | Type |
|---|---|
| `gbif_id` / `inat_id` (PK) | bigint |
| `taxon_key` | integer (FK) |
| `geom` | geometry(Point, 4326) |
| `event_date` | date |
| `coordinate_uncertainty_m` | float |
| `basis_of_record` | text |
| `source` | text (`GBIF`/`iNaturalist`) |
| `range_label` | text (`native`/`introduced`/`unknown`) |
| `ingested_ts` | timestamptz |
| `bio1..bio19` | float (extracted feature columns, nullable until extraction job runs) |

Indexing: GiST spatial index on `geom`; B-tree index on `(taxon_key, ingested_ts)`. This table is served to the frontend as vector tiles via `pg_tileserv`, not as raw JSON dumps, to keep map payloads small at low zoom levels.

### 5.4 Feature/Prediction Artifacts
- Per-species **GeoParquet** feature table (`features_<taxon_key>.parquet`).
- Per-species **suitability rasters**: `suitability_<taxon_key>_<model_version>.tif` (continuous 0–1) and `binary_<taxon_key>.tif`, both served via `titiler` for frontend display.
- Per-species **native mask** and **invasion-candidate mask**, stored as raster bands or dissolved vector polygons (served as GeoJSON for smaller polygon counts, vector tiles otherwise).

---

## 6. Modeling Pipeline

### 6.1 Species Resolution & Occurrence Retrieval
1. User types a common or scientific name in the React search box → debounced TanStack Query call to a `/species/search` FastAPI endpoint wrapping `pygbif.species.name_suggest()` and/or `pyinaturalist.get_taxa()`.
2. On selection, resolve to canonical `taxonKey` (GBIF backbone) — the single source of truth used everywhere downstream, embedded in the route as `/species/:taxonKey` via TanStack Router.
3. **First-time species:**
   - Backend triggers a GBIF occurrence download (`occ.download(f'taxonKey = {key}', format='SIMPLE_CSV')`) filtered by `hasCoordinate=True`, `hasGeospatialIssue=False`, coordinate uncertainty threshold [web:2][web:10].
   - Poll `occ.download_meta()` until `SUCCEEDED`, fetch via `occ.download_get()` — the DOI is stored and surfaced in the bulletin and in the frontend's "model lineage" panel [web:8].
   - Supplement with `pyinaturalist.get_observations(taxon_id=..., quality_grade='research')` [web:26].
   - Frontend shows job progress via a polled `/jobs/:jobId` endpoint (TanStack Query with `refetchInterval`) or an SSE stream.
4. **Repeat species (already in registry):**
   - Backend queries GBIF incrementally: `taxonKey = X AND lastInterpreted >= {last_gbif_fetch_ts}` via `occ.search` [web:9].
   - Insert only new `gbif_id`s (`ON CONFLICT DO NOTHING`), update watermark.
   - `Δn = new_records / n_occurrences_total`; set `retrain_needed = True` if `Δn > 5%` or new records fall outside the current suitability envelope.
   - Frontend's species-status badge (see Section 7) reflects this immediately via query invalidation.

### 6.2 Feature Extraction
- Sample the 19 bioclim bands for every occurrence and background point via `rasterio.sample()`/`xarray` against the cached global COG.
- Generate **background/pseudo-absence points** using target-group background sampling (preferred, corrects observer bias) or buffer-based random background (`elapid.sample_bias_file`, `elapid.sample_random`).
- Apply spatial thinning (`elapid.distance_thin`) to reduce spatial autocorrelation.
- Filter to a low-collinearity predictor subset (VIF < 5), precomputed globally or re-run per species if ranges differ substantially.

### 6.3 Model Training: Native Range vs. Invasion Projection
1. **Define native range**: GBIF's `establishmentMeans`/`degreeOfEstablishment` fields where populated, cross-checked against GISD/BIEN/USDA PLANTS or literature polygons; otherwise a user-editable heuristic polygon — editable directly on the React map via a deck.gl `EditableGeoJsonLayer`-style drawing tool.
2. **Model A (Native-trained)**: `elapid.MaxentModel()` on native-range occurrences + background; spatial block CV; report AUC, TSS, CBI [web:20].
3. **Project Model A globally** → continuous suitability surface; areas of high suitability outside the native polygon and occurrence buffer = **candidate invasion/expansion zones**.
4. **Model B (Combined native+invaded)**: trained when introduced-range occurrences exist; literature shows better predictive power and lower sensitivity to modeling choices than native-only extrapolation [web:25]. Presented as primary result when available; Model A flagged as lower-confidence exploratory layer with the known transferability caveat (mean AUC ≈ 0.7 cross-continentally) [web:18].
5. **Severity/Risk Index**:
   \[
   \text{Severity} = w_1 \cdot S_{\text{suitability}} + w_2 \cdot S_{\text{climate analogy}} + w_3 \cdot S_{\text{spread rate}} + w_4 \cdot S_{\text{ecological impact prior}}
   \]
   Weights configurable via a settings form (TanStack Form + Zod validation) in the frontend, documented in the bulletin methods section — never a black-box number.

### 6.4 Retraining Logic (Automation Rule)
```
IF species not in registry:
    full_pipeline_run()
ELSE:
    delta = fetch_incremental_gbif(taxon_key, since=last_gbif_fetch_ts)
    IF delta.count == 0:
        serve_cached_results()
    ELSE:
        append_to_occurrence_store(delta)
        IF (delta.count / total_count > 0.05) OR (delta has points outside current suitability envelope):
            retrain_model()
            regenerate_rasters_and_bulletin()
        ELSE:
            log_delta_only()  # keep serving prior model, flag "minor update pending"
```
Mirrors an asset-based orchestration pattern (Dagster's "software-defined assets": `raw_occurrences → features → model → suitability_raster → bulletin`), with the frontend simply reflecting current asset freshness state via API polling.

---

## 7. Dashboard UX / Innovation Features (React implementation notes)

1. **Species search-as-you-type** — TanStack Router search-param-driven autocomplete, debounced TanStack Query call, iNaturalist thumbnail + taxonomic breadcrumb rendered with shadcn/ui `Command` component.
2. **Dual-map synchronized view** — two `MapLibre`+`deck.gl` instances (or a single map with a swipe/compare control) sharing viewport state via a shared TanStack Store slice: left = observed occurrences (deck.gl `ScatterplotLayer`, colored by `range_label`), right = continuous suitability heatmap (deck.gl `BitmapLayer`/`HeatmapLayer` fed by titiler tiles), with a shadcn `Tabs`/`ToggleGroup` to switch "Native model / Combined model / MESS uncertainty."
3. **Time-slider "invasion timeline"** — a custom range slider (shadcn `Slider`) driving a deck.gl `TripsLayer`-style or filtered `ScatterplotLayer` animation over `event_date`, showing cumulative spread by year.
4. **"What-if" climate scenario toggle** — a `Select` component swapping the titiler tile source between current and CMIP6-future bioclim COGs, re-running the frontend's suitability layer request without a new model call.
5. **Confidence overlay** — MESS/uncertainty layer always toggle-visible alongside any invasion projection (deck.gl layer opacity/hatching pattern), directly addressing the transferability caveat [web:18].
6. **One-click "Generate Bulletin"** — button triggers a backend job (`POST /species/:taxonKey/bulletin`), frontend polls job status, then surfaces a download link for the generated PDF.
7. **Model lineage panel** — a TanStack Table listing `model_version`, training date, GBIF DOI, occurrence counts, and metrics, with column sorting/filtering out of the box.
8. **Species status badges** — small shadcn `Badge` components ("up to date," "N new records pending," "retrained on {date}") reactively updated via TanStack Query cache invalidation after each fetch/retrain job completes.
9. **Design system** — Tailwind + shadcn/ui gives full control over a cohesive dark/light theme, custom color-ramp legends for suitability maps, and consistent spacing/typography — the visual polish that Streamlit/Dash cannot match out of the box.

---

## 8. Automated Bulletin Generation

**Structure (1 PDF/HTML per species, generated server-side via Jinja2 → WeasyPrint — bulletin generation stays backend-rendered regardless of frontend framework, for print-quality consistency):**

1. **Header** — species name, taxon key, thumbnail image, date of generation, GBIF download DOI citation.
2. **Executive summary** — auto-generated narrative (severity index, expansion area in km², top contributing bioclim variables).
3. **Map panel 1** — native range occurrence map with suitability contours (rendered server-side with matplotlib/cartopy for the static PDF, distinct from the interactive frontend map).
4. **Map panel 2** — global suitability projection with candidate invasion zones and MESS uncertainty overlay.
5. **Severity gauge** — radial gauge chart with sub-components broken out.
6. **Variable importance chart** — bioclim variable contributions from Maxent's permutation importance.
7. **Model diagnostics table** — AUC, CBI, TSS, n_presence, n_background, spatial CV fold count.
8. **Expansion outlook narrative** — trend in new occurrences/year, top regions of concern, transferability caveat text.
9. **References/methods footnote** — GBIF DOI, WorldClim/CHELSA version, elapid/Maxent citation, transferability study links [web:17][web:18][web:20][web:25].

---

## 9. Implementation Phases & Milestones

| Phase | Deliverable | Est. Duration |
|---|---|---|
| 1. Data foundation | Download & cache global WorldClim/CHELSA stack as COG; build PostGIS schema; VIF-based predictor shortlist; stand up titiler/pg_tileserv | 1–2 weeks |
| 2. Connector layer | pygbif + pyinaturalist wrappers; species search/autocomplete endpoint; incremental-fetch logic with watermarking | 1 week |
| 3. Feature extraction pipeline | Raster sampling, background/pseudo-absence generation, spatial thinning | 1 week |
| 4. Core SDM engine | elapid MaxEnt training, spatial block CV, AUC/TSS/CBI evaluation, MLflow registry | 2 weeks |
| 5. Native-vs-invasive logic | Native-range polygon handling, Model A/B logic, MESS layer, severity index | 2 weeks |
| 6. Orchestration | Prefect/Dagster DAG for full pipeline + retrain-trigger rule; Celery+Redis for async jobs | 1 week |
| 7. FastAPI layer | OpenAPI-documented endpoints for species search, job status, model lineage, bulletin trigger | 1 week |
| 8. React frontend scaffold | Vite + TanStack Router/Query/Table setup, generated typed API client, shadcn/ui + Tailwind theme | 1–2 weeks |
| 9. Map & visualization layer | MapLibre + deck.gl integration, dual synchronized maps, time slider, MESS overlay | 2–3 weeks |
| 10. Bulletin generator | Jinja2/WeasyPrint templated PDF with all charts | 1 week |
| 11. Testing & validation | Run on 5–10 well-documented invasive species (e.g., *Lantana camara*, *Ailanthus altissima*, *Vespa velutina*) to sanity-check against known invasion literature | 1–2 weeks |
| 12. Deployment & docs | Dockerize backend + static frontend build; deploy (Cloud Run/K8s + Nginx/CDN); reproducibility documentation and API reference | 1 week |

**Total estimated timeline:** ~16–20 weeks for a full-featured MVP with a polished React frontend, solo developer; parallelizable to ~10–12 weeks with a backend/pipeline contributor and a frontend contributor working concurrently against the shared OpenAPI contract.

---

## 10. Key Technical Risks & Mitigations

- **Sampling bias in citizen-science data**: mitigate with target-group background sampling and spatial thinning; disclose bias explicitly in the bulletin.
- **Native range ambiguity**: automate a best-guess but always allow user override via the frontend's editable polygon layer — never present a fully automated native/invasive split as ground truth.
- **Overstated invasion-risk confidence**: enforce the MESS/uncertainty overlay and the AUC ≈ 0.7 transferability caveat as mandatory content, not optional [web:18].
- **Storage growth**: partition occurrence tables by `taxon_key`; archive/compress inactive species' feature parquet files after N months of no re-query.
- **GBIF rate limits / download queue delays**: use `occ.search` incremental pulls for routine updates, reserve full `occ.download` for first-time species or periodic full refreshes.
- **Reproducibility**: log `bioclim_version`, GBIF download key/DOI, pinned package versions, and random seeds per model version.
- **Reinventing existing science**: reuse Wallace's and BON in a Box's validated modeling defaults rather than re-deriving Maxent configuration choices from scratch [web:47][web:41].
- **Frontend/backend type drift**: generate the TypeScript API client directly from FastAPI's OpenAPI schema (`openapi-typescript`) on every backend change, so the React app never silently falls out of sync with the Pydantic contract.
- **Map performance at scale**: for species with very large occurrence counts (>100k points), always render via deck.gl aggregation layers (hexbin/heatmap) rather than individual `ScatterplotLayer` points at low zoom, switching to raw points only at high zoom/small viewport extents.

---

## 11. Suggested Repository Structure

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
│   │   │   ├── ui/                 # shadcn/ui primitives
│   │   │   ├── maps/                # MapLibre + deck.gl components
│   │   │   └── charts/              # Recharts/visx components
│   │   ├── api/                    # generated OpenAPI client + TanStack Query hooks
│   │   ├── store/                  # TanStack Store (ephemeral UI state)
│   │   └── styles/                  # Tailwind config/theme
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
├── docker-compose.yml
└── README.md
```

---

## References

- GBIF pygbif technical documentation [web:1][web:10]
- GBIF Occurrence Download API [web:2][web:9]
- GBIF citation guidelines (DOI-backed downloads) [web:8]
- pyinaturalist client documentation [web:26][web:29]
- elapid — Python MaxEnt SDM library [web:15]
- WorldClim v2.1 bioclimatic variables [web:21][web:28]
- Broennimann et al., "Ecological Niche Transferability Using Invasive Species as a Case Study," PLoS ONE, 2015 [web:17]
- "How well do species distribution models predict occurrences in exotic ranges?" Global Ecology and Biogeography, 2022 [web:18]
- "Species distribution model transferability and model grain size," Scientific Reports, 2018 (Continuous Boyce Index for MaxEnt) [web:20]
- "Improving Transferability of Introduced Species' Distribution Models," PMC, 2013 (combined native+invaded training) [web:25]
- Wallace EcoMod, CRAN package documentation [web:47][web:58]
- "wallace 2: a shiny app for modeling species niches and distributions redesigned to facilitate expansion via module contributions," 2023 [web:48][web:51]
- BON in a Box, Species Distribution Models indicator, GEO BON [web:41]
- biomod2 — Ensemble Platform for Species Distribution Modeling [web:56][web:59]
- Galaxy Training Network, "Species distribution modeling" ecology tutorial [web:53]
- iMapInvasives, NatureServe [web:52]
- MaxEnt invasive-species case studies: *Leptocybe invasa* [web:33]; *Galinsoga parviflora* [web:38]; *Sirex noctilio* [web:40]; invasive scarab beetles [web:37]
- CARTO, "Making deck.gl AI-Ready" (MapLibre as 2026 React mapping consensus) [web:61]
- deck.gl documentation — React integration and MapLibre interop [web:63][web:65]
- js-maps.com, "Best JavaScript Map Libraries for Interactive Maps in 2026" [web:74]
- TanStack ecosystem field guide (Query, Router, Table, Form, Store) [web:68]
- TanStack Query / Router official docs [web:62][web:64][web:69]
