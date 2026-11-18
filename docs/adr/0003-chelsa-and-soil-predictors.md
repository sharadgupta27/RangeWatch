# ADR 0003 — Soil predictors (implemented) and CHELSA as a climate source (plan)

- **Status:** Part A (SoilGrids soil predictors) **accepted and implemented**, opt-in per bioclim
  version. Part B (climate source and predictors chosen per species; CHELSA stacks, CMIP6
  futures and CHELSA-BIOCLIM+ variables; predictor-set comparison) **implemented**, decided on
  2026-11-18. The deployment default stays WorldClim, and CHELSA is opt-in per species. That
  keeps CLAUDE.md's "WorldClim v2.1 (primary), CHELSA (optional)" valid.
- **Date:** 2026-11-17, updated 2026-11-18
- **Touches:** CLAUDE.md tech stack ("Bioclim source: WorldClim v2.1 (primary), CHELSA
  (optional cross-check)"); "Global static layer … versioned"; reproducibility requirements.
  reply.md §2.2, §2.3, §3.1, §8 and RQ2 (nested models M0 → M1 → M2).

## Part A — SoilGrids 2.0 soil predictors (implemented)

**Why.** Climate-only models cannot represent substrate limits. The clearest case is
*Heterotheca subaxillaris*: its climate-only model has CBI −0.02, and it is tied to coastal
sands like *Acacia saligna* and *Carpobrotus edulis* (reply.md §3.1).

**Design.**

- Soil properties are **static predictor layers** of a bioclim version (`metadata.static_layers`,
  `static_layers/soil_<property>.tif`). They are not extra bands of the climate COG.
  - `BioclimStack` merges them with the climate bands for point sampling, block reads and
    overviews.
  - Training, background sampling, MESS, the five extrapolation diagnostics, projection,
    scenarios and high-resolution views all see one predictor set, with no change to those
    modules.
- **Soil is static under future climate.** A scenario stack holds climate only. Its projection
  reuses the version's soil layers. This is the standard assumption, and the bulletin states it.
- **New versions only.** `bioclim_store add-soil --from <v> --version <new>` copies the source
  version and adds the soil layers. It then re-runs the VIF screen over climate + soil
  (`--core` protects hypothesis-driven predictors). Existing versions and the models trained on
  them are untouched.
- **Data.**
  - ISRIC SoilGrids 2.0 aggregated means (5 km or 1 km), from
    `files.isric.org/soilgrids/latest/data_aggregated`.
  - Topsoil 0–30 cm as the thickness-weighted mean of the three upper depths.
  - ISRIC conversion factors to conventional units.
  - Reprojected from Homolosine with average resampling, and masked to the version's land cells.
- **Default properties:** pH, sand, clay, SOC, bulk density, and available water capacity
  (θ33 kPa − θ1500 kPa). Silt is excluded because the three texture fractions sum to 100 %.
- **Persistence and lineage.**
  - Soil values are not cached in `occurrences`. That table keeps `bio1..bio19` only, so there
    is no schema change.
  - Soil is re-sampled each run and captured in the hashed `training_data.parquet`.
  - Every model version's reproducibility record lists each static layer: source, citation,
    SHA-256, land coverage, and whether the model used it. The record is shown verbatim in the
    lineage table, the bulletin, the model card, ODMAP and the citation bundle.
- **Cross-check.** The CHELSA climate cross-check swaps climate bands only and keeps soil at
  its training values, so it still isolates the climate source.

**Known limitations.**

- SoilGrids is a model product with its own uncertainty, and its 90 % prediction intervals
  are not used yet.
- Land cells with no SoilGrids prediction (water, glaciers, some urban and bare-rock cells)
  lose their records from training.
- At 30″, soil comes from the version grid unless `--with-hires` builds 1 km layers.

## Part B — Climate source and predictors per species (implemented)

### Decisions (2026-11-18)

1. **Climate data is a per-species choice.** It is offered in the UI, not set once for the
   whole deployment.
2. **Which climate dimension limits a species is not known in advance.** It may be rainfall,
   temperature, humidity, wind or a combination. So predictors are chosen per species, and the
   data can rank candidate sets.
3. **Resolution: as fine as the data allow.** That is 30″ (about 1 km), the native resolution of
   both WorldClim and CHELSA, used for regional models.

### Data facts that shape the design

Checked against the CHELSA bucket listing on 2026-11-18:

- **CHELSA V2.1, 1981–2010:**
  - BIO1–19;
  - relative humidity `hurs_*`, VPD `vpd_*`, wind `sfcWind_*`, moisture index `cmi_*`, PET
    `pet_penman_*`, radiation `rsds_*` and cloud cover `clt_*` (each as max/mean/min/range);
  - aridity `ai`, `swb`;
  - growing season and frost (`gdd*`, `ngd*`, `gsl`, `gsp`, `gst`, `fcf`), `scd`, `npp`.
- **CHELSA CMIP6 futures:**
  - GFDL-ESM4, IPSL-CM6A-LR, MPI-ESM1-2-HR, MRI-ESM2-0, UKESM1-0-LL;
  - SSP1-2.6, SSP3-7.0 and SSP5-8.5;
  - 2011–2040, 2041–2070 and 2071–2100.
  - Each has BIO1–19 plus growing season, frost, snow and NPP, but **no humidity, VPD, wind,
    PET, moisture index or radiation**. This was verified for MPI-ESM1-2-HR / SSP5-8.5 /
    2041–2070. The builder treats any missing file as "no future data" rather than assuming
    the same for the others.
- **WorldClim:** CMIP6 futures are temperature and precipitation only.

Consequence: a model that uses humidity or wind can map the present distribution and be
validated, but it **cannot** be projected into the future from these sources. RangeWatch never
fills a missing future layer with present values. It reports the scenario as unavailable, with
the reason.

### What was built

- **Climate layers.** `metadata.climate_layers` holds climate variables outside BIO1–19, each with
  a file per scenario. They differ from static soil layers: a scenario view of the stack uses the
  layer's own future file, or raises `PresentOnlyPredictorError`. Versions record
  `source_family` and `baseline_period`.
- **`build-chelsa`.**
  - It builds CHELSA versions on a WorldClim version's grid and land mask, in WorldClim units.
  - `--variables` adds CHELSA-BIOCLIM+ variables (`default` = hurs_mean, vpd_max,
    sfcWind_mean, cmi_mean, gdd5, gsl, fcf).
  - `--with-hires` also builds the matched version's 30″ grid from the same CHELSA files.
- **`add-scenario`.**
  - On a CHELSA version it registers a CHELSA CMIP6 future: BIO1–19, plus every extra variable
    CHELSA publishes for that future.
  - The others stay present-only for that scenario and are logged.
  - Sources are never mixed: CHELSA futures only on CHELSA versions, WorldClim futures never on
    CHELSA versions.
- **Per-species choice.**
  - Stored as `species_registry.climate_version` and `predictor_config`, and set with
    `PUT /species/{key}/environment`. Changing either flags a retrain.
  - Predictor choice is automatic (the version's global VIF shortlist), or the user's
    variables. User-chosen variables are VIF-screened on **the species' own background** (its
    accessible environments) unless the user turns the screen off. Protected predictors are
    never dropped.
  - The lineage records requested, unavailable, dropped and used predictors, the present-only
    ones, and the climate family, baseline and resolution.
- **`GET /bioclim/versions`** lists the installed versions, with predictors grouped by
  temperature, precipitation, humidity, wind, radiation, water balance, growing season or soil,
  and whether future data exists for each.
- **Predictor-set comparison.**
  - Run with `POST /species/{key}/predictor-comparison` or the Model tab.
  - It refits the current version's exact training locations (from its snapshot) with candidate
    sets: current, temperature, precipitation, temperature + precipitation, + humidity/wind,
    + water balance, + growing season, + soil, and all. Sets can also be user-defined.
  - Each set is VIF-screened. Sets are ranked by spatial block CV CBI, then AUC. Sets within
    0.02 CBI of the best are reported as performing alike.
  - "Use this set" stores the set as the species' choice.
  - This is the per-species answer to "rain for some species, temperature for others". It is a
    relative comparison on the same data, and should be confirmed on independent field records.
- **Cross-check.** It picks a partner from the other family: a CHELSA-trained species is checked
  against WorldClim. Predictors the partner lacks (e.g. humidity) are kept at their training
  values and listed.
- **Data-quality test.** The "before the climate baseline" test takes its year from the version:
  1970 for WorldClim, 1981 for CHELSA. `SDM_QUALITY_BASELINE_START_YEAR` overrides it.

### Resolution

- 30″ (about 1 km) is the finest resolution either source offers. It is used for **regional**
  models (study region at 30″, limited by `SDM_HIRES_MAX_CELLS`), and for validation against
  field data at that scale.
- Global models stay at the version's base resolution. 2.5′ is recommended for production.
- A global 30″ model means about 0.9 billion cells per raster, times five extrapolation
  diagnostics, times every scenario. That is not viable as a nightly-updated service.
- Regional 30″ models currently have no scenario stacks at 30″, so their futures are projected
  only on the base grid.

### Still open

- CHELSA CMIP6 futures at 30″ for regional models (the same builder on the hi-res grid; heavy).
- Temporally matched predictors (annual CHELSA/TerraClimate series) for event-dated records.
- Making CHELSA the deployment default: an explicit CLAUDE.md change, if ever wanted.
