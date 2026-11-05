"""Record-level data-quality tests and quality tiers for occurrence records.

The tests follow CoordinateCleaner (Zizka et al. 2019) where it has a counterpart:
capitals, country/province centroids, biodiversity institutions, GBIF headquarters, equal
coordinates. They are complemented by GBIF's own interpretation issue flags, a
coordinate-precision test scaled to the model grid, a date test against the climate baseline,
two outlier tests (environmental and spatial), cultivation tests (locality / habitat text and
nearness to a botanical garden), an identification-support test for iNaturalist observations
and a duplicate-specimen test.

Two kinds of test:

* **error** tests mark records that should not enter training: the location is very likely
  wrong, or the record duplicates another one. Records they flag are excluded from training
  by default (`QualityConfig.exclude`).
* **warning** tests mark records worth a look. They are reported and lower the record's
  quality tier but are never excluded by default: for an invading species an isolated or
  climatically unusual record is often the invasion front, not an error.

Nothing here modifies the occurrence store (append-only). Flags are computed per training run
on the records considered, written to the model version's `quality_flags.parquet` snapshot
(SHA-256 in the lineage) and summarised in its reproducibility record, together with the
configuration and the reference data version the tests used.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.neighbors import BallTree

from src.features.quality_refs import QualityReferences

QUALITY_VERSION = "quality_v1"
EARTH_RADIUS_M = 6_371_008.8
METRES_PER_DEGREE = 111_320.0
# GBIF Secretariat, Copenhagen (CoordinateCleaner cc_gbif).
GBIF_HQ_LONLAT = (12.5788, 55.6761)
SPECIMEN_BASIS = {"PRESERVED_SPECIMEN", "MATERIAL_SAMPLE", "MATERIAL_CITATION"}

# GBIF interpretation issues grouped into tests. Geospatial errors are mostly removed already
# by the `hasGeospatialIssue = FALSE` query filter; the test catches what slips through (and
# records fetched through other routes).
GBIF_GEOSPATIAL_ERRORS = frozenset(
    {
        "ZERO_COORDINATE",
        "COORDINATE_OUT_OF_RANGE",
        "COORDINATE_INVALID",
        "COUNTRY_COORDINATE_MISMATCH",
        "COORDINATE_REPROJECTION_FAILED",
    }
)
# Coordinates GBIF corrected during interpretation (probably right now, but worth a look).
GBIF_COORDINATE_CORRECTED = frozenset(
    {
        "PRESUMED_SWAPPED_COORDINATE",
        "PRESUMED_NEGATED_LATITUDE",
        "PRESUMED_NEGATED_LONGITUDE",
        "COORDINATE_REPROJECTION_SUSPICIOUS",
    }
)
GBIF_TAXON_MATCH = frozenset({"TAXON_MATCH_FUZZY", "TAXON_MATCH_HIGHERRANK"})
GBIF_DATE_ISSUES = frozenset({"RECORDED_DATE_INVALID", "RECORDED_DATE_UNLIKELY"})

# Locality / habitat / remarks wording of cultivated or planted individuals (English plus the
# commonest herbarium languages). "Garden escape" and similar still match: they are flagged for
# review, not excluded. "Park" alone is left out (national / nature parks are wild sites).
CULTIVATION_PATTERN = re.compile(
    r"\b(?:cultivated|cultivad[ao]s?|cultiv[ée]e?s?|kultiviert|planted|plantation|ornamental|"
    r"gardens?|jard[ií]n(?:es)?|jardim|giardino|garten|nursery|arboretum|botanic\w*|hortus|"
    r"(?:city|urban|public|municipal) park)\b|\b(?:cult|hort)\.",
    re.IGNORECASE,
)
DETAIL_TEXT_FIELDS = ("locality", "habitat", "remarks")
GRSCICOLL_GARDEN_TYPE = "BOTANICAL_GARDEN"
# Record numbers meaning "no number" (sine numero), which cannot identify duplicates.
_NO_NUMBER = {"", "sn", "none", "na", "nan", "unknown", "0"}

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class QualityTest:
    id: str
    label: str
    severity: Severity
    description: str
    reference: str | None = None


TESTS: dict[str, QualityTest] = {
    t.id: t
    for t in (
        QualityTest(
            "equal_coordinates",
            "Equal coordinates",
            "error",
            "Longitude equals latitude, a common data-entry error.",
            "CoordinateCleaner cc_equ",
        ),
        QualityTest(
            "gbif_headquarters",
            "GBIF headquarters",
            "error",
            "Within the buffer of the GBIF Secretariat in Copenhagen, where records without "
            "a real location have been geocoded.",
            "CoordinateCleaner cc_gbif",
        ),
        QualityTest(
            "capital",
            "Country capital",
            "error",
            "At a country capital's coordinates (within the exact buffer): records geocoded "
            "to the capital instead of the collecting site.",
            "CoordinateCleaner cc_cap; Natural Earth populated places",
        ),
        QualityTest(
            "near_capital",
            "Near a country capital",
            "warning",
            "Within CoordinateCleaner's wider capital buffer but not at the capital point. "
            "Flagged for review only: the wide buffer covers whole metropolitan areas, where "
            "urban and coastal species are genuinely recorded (on Israel Nature and Parks "
            "Authority survey data it caught 197 precise Acacia saligna records a median "
            "7.5 km from Tel Aviv and Jerusalem).",
            "CoordinateCleaner cc_cap (10 km default buffer)",
        ),
        QualityTest(
            "country_centroid",
            "Country centroid",
            "error",
            "Within the buffer of a country centroid: records geocoded to the country only.",
            "CoordinateCleaner cc_cen; Natural Earth admin-0",
        ),
        QualityTest(
            "province_centroid",
            "Province centroid",
            "error",
            "Within the buffer of a province / state centroid: records geocoded to the "
            "province only.",
            "CoordinateCleaner cc_cen; Natural Earth admin-1",
        ),
        QualityTest(
            "institution",
            "Biodiversity institution",
            "error",
            "Within the buffer of a herbarium, museum, botanical garden or other collection "
            "holder: often the institution's address, or a cultivated plant.",
            "CoordinateCleaner cc_inst; GBIF GRSciColl",
        ),
        QualityTest(
            "gbif_geospatial_issue",
            "GBIF geospatial issue",
            "error",
            "GBIF flagged the coordinates as invalid, out of range, zero, failing "
            "reprojection, or outside the stated country.",
            "GBIF occurrence issues",
        ),
        QualityTest(
            "low_precision",
            "Low coordinate precision",
            "error",
            "The location is less precise than one model grid cell: the stated coordinate "
            "uncertainty, or (if none is given) the precision implied by the number of "
            "decimals, exceeds the cell size.",
            "CoordinateCleaner cd_round (record-level analogue)",
        ),
        QualityTest(
            "duplicate_specimen",
            "Duplicate specimen",
            "error",
            "Same collector, collector number and date as another specimen (herbarium "
            "duplicates sent to several institutions). One record of each set is kept; the "
            "others would inflate local density.",
            "CoordinateCleaner cc_dupl (collector / number / date variant)",
        ),
        QualityTest(
            "gbif_coordinate_corrected",
            "Coordinates corrected by GBIF",
            "warning",
            "GBIF presumed the coordinates swapped or negated and corrected them.",
            "GBIF occurrence issues",
        ),
        QualityTest(
            "gbif_taxon_match",
            "Uncertain taxon match",
            "warning",
            "GBIF matched the name fuzzily or only to a higher rank.",
            "GBIF occurrence issues",
        ),
        QualityTest(
            "gbif_date_issue",
            "Date issue",
            "warning",
            "GBIF flagged the recorded date as invalid or unlikely.",
            "GBIF occurrence issues",
        ),
        QualityTest(
            "before_climate_baseline",
            "Before the climate baseline",
            "warning",
            "Recorded before the climate baseline period starts, so the climate at the time "
            "may differ from the predictors.",
        ),
        QualityTest(
            "cultivation_text",
            "Cultivation wording",
            "warning",
            "Locality, habitat or remarks mention a garden, planting, nursery or cultivation "
            "('cult.', 'planted', 'botanic garden'...). Flagged for review only: "
            "'garden escape' records are wild.",
        ),
        QualityTest(
            "near_botanical_garden",
            "Near a botanical garden",
            "warning",
            "Within the botanical-garden buffer of a GRSciColl institution of type botanical "
            "garden: possibly a cultivated plant (or an escape from the collection).",
            "GBIF GRSciColl (type BOTANICAL_GARDEN)",
        ),
        QualityTest(
            "weak_identification",
            "Weak identification",
            "warning",
            "iNaturalist observation whose identification is contested (at least one "
            "disagreeing identification), or whose observer took a computer-vision suggestion "
            "confirmed by no more than one other person.",
            "iNaturalist identification counts",
        ),
        QualityTest(
            "environmental_outlier",
            "Environmental outlier",
            "warning",
            "Unusual climate compared with the species' other records (robust Mahalanobis "
            "distance beyond the chi-square cut-off). Flagged for review only: it can be an "
            "error or a genuine range edge / invasion front.",
            "robust covariance (MinCovDet)",
        ),
        QualityTest(
            "spatial_isolate",
            "Spatially isolated",
            "warning",
            "No other record of the species within the isolation distance. Flagged for review "
            "only: an isolated record may be a new introduction.",
        ),
    )
}
DEFAULT_EXCLUDE: tuple[str, ...] = tuple(t.id for t in TESTS.values() if t.severity == "error")


@dataclass(frozen=True)
class QualityConfig:
    # Flags whose records are excluded from training; every other flag is report-only.
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE
    # Capital: excluded within the exact buffer (the capital point itself), warning only
    # within CoordinateCleaner's 10 km default. Other buffers are CoordinateCleaner defaults.
    capital_buffer_m: float = 1_000.0
    capital_vicinity_m: float = 10_000.0
    centroid_buffer_m: float = 1_000.0
    institution_buffer_m: float = 100.0
    botanical_garden_buffer_m: float = 1_000.0
    # Agreements include the observer's own identification (research grade needs two).
    weak_id_max_agreements: int = 2
    gbif_hq_buffer_m: float = 1_000.0
    # First year of the climate baseline (WorldClim v2.1: 1970–2000).
    baseline_start_year: int = 1970
    outlier_probability: float = 0.999
    outlier_min_records: int = 30
    outlier_fit_max_records: int = 5_000
    isolation_km: float = 1_000.0
    tier_a_max_uncertainty_m: float = 1_000.0

    def __post_init__(self) -> None:
        unknown = set(self.exclude) - set(TESTS)
        if unknown:
            raise ValueError(f"Unknown quality tests in exclude: {sorted(unknown)}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "exclude": list(self.exclude),
            "capital_buffer_m": self.capital_buffer_m,
            "capital_vicinity_m": self.capital_vicinity_m,
            "centroid_buffer_m": self.centroid_buffer_m,
            "institution_buffer_m": self.institution_buffer_m,
            "botanical_garden_buffer_m": self.botanical_garden_buffer_m,
            "weak_id_max_agreements": self.weak_id_max_agreements,
            "gbif_hq_buffer_m": self.gbif_hq_buffer_m,
            "baseline_start_year": self.baseline_start_year,
            "outlier_probability": self.outlier_probability,
            "outlier_min_records": self.outlier_min_records,
            "outlier_fit_max_records": self.outlier_fit_max_records,
            "isolation_km": self.isolation_km,
            "tier_a_max_uncertainty_m": self.tier_a_max_uncertainty_m,
        }


@dataclass
class QualityResult:
    """Per-record flags/tiers aligned with the input frame, plus the run summary."""

    flags: list[list[str]]
    tier: np.ndarray
    excluded: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)

    def frame(self, occ: pd.DataFrame) -> pd.DataFrame:
        """The `quality_flags.parquet` snapshot: one row per record assessed."""
        out = occ[["source", "record_id", "longitude", "latitude"]].copy()
        out["quality_flags"] = [";".join(f) for f in self.flags]
        out["quality_tier"] = self.tier
        out["quality_excluded"] = self.excluded
        out = out.astype({"source": "string", "record_id": "Int64"})
        return out.sort_values(["source", "record_id"], kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------- helpers
def _decimals(value: float) -> int:
    """Decimal places as written (shortest round-trip repr, trailing zeros dropped)."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        return max(0, -int(text.lower().split("e")[1]))
    frac = text.split(".")[1] if "." in text else ""
    frac = frac.rstrip("0")
    return len(frac)


def implied_uncertainty_m(lon: float, lat: float) -> float:
    """Half the last-digit step of the more precise coordinate, in metres (a coordinate
    written with d decimals is known to ±0.5·10^-d degrees)."""
    d = max(_decimals(lon), _decimals(lat))
    return 0.5 * 10.0 ** (-d) * METRES_PER_DEGREE


def _radians(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.deg2rad(np.column_stack([lat, lon]))


def _within(
    lon: np.ndarray, lat: np.ndarray, ref_lon: np.ndarray, ref_lat: np.ndarray, buffer_m: float
) -> np.ndarray:
    """True where a record lies within `buffer_m` of any reference point."""
    if len(ref_lon) == 0 or len(lon) == 0:
        return np.zeros(len(lon), dtype=bool)
    tree = BallTree(_radians(ref_lon, ref_lat), metric="haversine")
    dist, _ = tree.query(_radians(lon, lat), k=1)
    return dist[:, 0] * EARTH_RADIUS_M <= buffer_m


def _issue_sets(values: Sequence[Any]) -> list[set[str] | None]:
    out: list[set[str] | None] = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(None)
        else:
            out.append({str(i).upper() for i in v})
    return out


def _record_details(occ: pd.DataFrame) -> list[dict[str, Any] | None]:
    if "details" not in occ:
        return [None] * len(occ)
    return [v if isinstance(v, dict) else None for v in occ["details"].tolist()]


def _norm_key(value: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(value).lower())


def _duplicate_specimens(
    occ: pd.DataFrame, rec_details: list[dict[str, Any] | None], basis: np.ndarray
) -> np.ndarray:
    """All but the first (by source, record id) of each set of specimens sharing collector,
    collector number and event date."""
    n = len(occ)
    dates = (
        occ["event_date"].astype("string").fillna("").to_numpy()
        if "event_date" in occ
        else np.full(n, "")
    )
    keys: list[tuple[str, str, str] | None] = []
    for i, d in enumerate(rec_details):
        d = d or {}
        number = _norm_key(d.get("record_number", ""))
        collector = _norm_key(d.get("recorded_by", ""))
        usable = basis[i] in SPECIMEN_BASIS and collector and number not in _NO_NUMBER
        keys.append((collector, number, str(dates[i])) if usable and dates[i] else None)
    sources = occ["source"].astype(str).tolist() if "source" in occ else [""] * n
    ids = pd.to_numeric(occ["record_id"], errors="coerce").tolist() if "record_id" in occ else []
    order = sorted(range(n), key=lambda i: (sources[i], ids[i] if ids else i))
    seen: set[tuple[str, str, str]] = set()
    out = np.zeros(n, dtype=bool)
    for i in order:
        k = keys[i]
        if k is None:
            continue
        if k in seen:
            out[i] = True
        else:
            seen.add(k)
    return out


# ------------------------------------------------------------------- tests
def assess_quality(
    occ: pd.DataFrame,
    predictors: Sequence[str],
    cell_deg: float,
    refs: QualityReferences | None = None,
    cfg: QualityConfig | None = None,
    seed: int = 0,
) -> QualityResult:
    """Run every test on `occ` (needs longitude/latitude; uses event_date, basis_of_record,
    coordinate_uncertainty_m, issues, details and the predictor columns when present).

    Tests whose inputs are missing (no reference table installed, too few records, issues
    not recorded) are reported as skipped with the reason, never silently passed.
    """
    cfg = cfg or QualityConfig()
    n = len(occ)
    lon = occ["longitude"].to_numpy(dtype="float64")
    lat = occ["latitude"].to_numpy(dtype="float64")
    flagged: dict[str, np.ndarray] = {}
    status: dict[str, tuple[str, str | None]] = {}
    details: dict[str, Any] = {}

    def run(test_id: str, mask: np.ndarray) -> None:
        flagged[test_id] = np.asarray(mask, dtype=bool)
        status[test_id] = ("run", None)

    def skip(test_id: str, reason: str) -> None:
        flagged[test_id] = np.zeros(n, dtype=bool)
        status[test_id] = ("skipped", reason)

    run("equal_coordinates", np.isclose(lon, lat, rtol=0, atol=1e-9))
    run(
        "gbif_headquarters",
        _within(
            lon,
            lat,
            np.array([GBIF_HQ_LONLAT[0]]),
            np.array([GBIF_HQ_LONLAT[1]]),
            cfg.gbif_hq_buffer_m,
        ),
    )

    ref_tests = (
        ("capital", "capitals", cfg.capital_buffer_m),
        ("near_capital", "capitals", cfg.capital_vicinity_m),
        ("country_centroid", "country_centroids", cfg.centroid_buffer_m),
        ("province_centroid", "province_centroids", cfg.centroid_buffer_m),
        ("institution", "institutions", cfg.institution_buffer_m),
    )
    for test_id, table, buffer_m in ref_tests:
        ref = refs.table(table) if refs is not None else None
        if ref is None or ref.empty:
            skip(
                test_id,
                f"reference table '{table}' not installed "
                "(python -m src.features.quality_refs build)",
            )
            continue
        run(
            test_id,
            _within(
                lon,
                lat,
                ref["longitude"].to_numpy(float),
                ref["latitude"].to_numpy(float),
                buffer_m,
            ),
        )
    # The vicinity ring excludes the capital point itself (already an error there).
    flagged["near_capital"] &= ~flagged["capital"]

    basis = (
        occ["basis_of_record"].fillna("").astype(str).str.upper().to_numpy()
        if "basis_of_record" in occ
        else np.full(n, "")
    )
    source = occ["source"].astype(str).to_numpy() if "source" in occ else np.full(n, "GBIF")
    rec_details = _record_details(occ)
    has_details = np.array([d is not None for d in rec_details], dtype=bool)
    details["n_details_unknown"] = int((~has_details).sum())
    no_details = (
        "record details (locality, collector, identifications) not recorded for these "
        "records yet (re-sync)"
    )

    # Cultivated / planted individuals and herbarium duplicates.
    if not has_details.any():
        skip("cultivation_text", no_details)
        skip("duplicate_specimen", no_details)
    else:
        run(
            "cultivation_text",
            np.array(
                [
                    d is not None
                    and any(
                        CULTIVATION_PATTERN.search(str(d.get(f) or "")) for f in DETAIL_TEXT_FIELDS
                    )
                    for d in rec_details
                ],
                dtype=bool,
            ),
        )
        run("duplicate_specimen", _duplicate_specimens(occ, rec_details, basis))

    inst = refs.table("institutions") if refs is not None else None
    gardens = (
        inst[inst["types"].fillna("").astype(str).str.upper().str.contains(GRSCICOLL_GARDEN_TYPE)]
        if inst is not None and not inst.empty and "types" in inst
        else None
    )
    if gardens is None or gardens.empty:
        skip(
            "near_botanical_garden",
            "no botanical gardens in reference table 'institutions' "
            "(python -m src.features.quality_refs build)",
        )
    else:
        near = _within(
            lon,
            lat,
            gardens["longitude"].to_numpy(float),
            gardens["latitude"].to_numpy(float),
            cfg.botanical_garden_buffer_m,
        )
        # Inside the (much smaller) institution buffer it is already an error.
        run("near_botanical_garden", near & ~flagged["institution"])

    # Identification support of iNaturalist observations.
    inat = source == "iNaturalist"
    inat_known = inat & np.array(
        [d is not None and "id_agreements" in d for d in rec_details], dtype=bool
    )
    if not inat.any():
        skip("weak_identification", "no iNaturalist observations")
    elif not inat_known.any():
        skip("weak_identification", "identification counts not recorded yet (re-sync)")
    else:
        weak = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(inat_known):
            d = rec_details[i] or {}
            disagreements = int(d.get("id_disagreements") or 0)
            agreements = int(d.get("id_agreements") or 0)
            vision = bool(d.get("observer_id_from_vision"))
            weak[i] = disagreements > 0 or (vision and agreements <= cfg.weak_id_max_agreements)
        run("weak_identification", weak)

    # GBIF issue flags (iNaturalist records carry none; NULL = ingested before they were kept)
    issue_sets = _issue_sets(occ["issues"].tolist()) if "issues" in occ else [None] * n
    gbif = (occ["source"] == "GBIF").to_numpy() if "source" in occ else np.ones(n, dtype=bool)
    unknown = np.array([s is None for s in issue_sets]) & gbif
    known = [s or set() for s in issue_sets]
    issue_counts = Counter(i for s in known for i in s)
    details["issue_counts"] = dict(issue_counts.most_common(15))
    details["n_issues_unknown"] = int(unknown.sum())
    for test_id, group in (
        ("gbif_geospatial_issue", GBIF_GEOSPATIAL_ERRORS),
        ("gbif_coordinate_corrected", GBIF_COORDINATE_CORRECTED),
        ("gbif_taxon_match", GBIF_TAXON_MATCH),
        ("gbif_date_issue", GBIF_DATE_ISSUES),
    ):
        if gbif.any() and unknown[gbif].all():
            skip(test_id, "GBIF issue flags not recorded for these records yet (re-sync)")
        else:
            run(test_id, np.array([bool(s & group) for s in known]))

    # Precision relative to the model grid.
    cell_m = cell_deg * METRES_PER_DEGREE
    unc = (
        pd.to_numeric(occ["coordinate_uncertainty_m"], errors="coerce").to_numpy(float)
        if "coordinate_uncertainty_m" in occ
        else np.full(n, np.nan)
    )
    implied = np.array([implied_uncertainty_m(x, y) for x, y in zip(lon, lat, strict=True)])
    effective = np.where(np.isnan(unc), implied, unc)
    run("low_precision", effective > cell_m)
    details["grid_cell_m"] = round(cell_m, 1)

    years = (
        pd.to_datetime(occ["event_date"], errors="coerce").dt.year.to_numpy(float)
        if "event_date" in occ
        else np.full(n, np.nan)
    )
    run("before_climate_baseline", years < cfg.baseline_start_year)
    details["n_undated"] = int(np.isnan(years).sum())

    _environmental_outliers(occ, predictors, cfg, seed, run, skip, details)

    if n < 2:
        skip("spatial_isolate", "fewer than 2 records")
    else:
        tree = BallTree(_radians(lon, lat), metric="haversine")
        dist, _ = tree.query(_radians(lon, lat), k=2)
        run("spatial_isolate", dist[:, 1] * EARTH_RADIUS_M / 1000.0 > cfg.isolation_km)

    # ------------------------------------------------ combine
    order = list(TESTS)
    flags = [[t for t in order if flagged[t][i]] for i in range(n)]
    excluded = np.array([any(f in cfg.exclude for f in fl) for fl in flags], dtype=bool)
    warned = np.array([any(f not in cfg.exclude for f in fl) for fl in flags], dtype=bool)
    tier = np.where(
        excluded,
        "excluded",
        np.where(
            warned | np.isnan(unc),
            "C",
            np.where(
                np.isin(basis, list(SPECIMEN_BASIS)) & (unc <= cfg.tier_a_max_uncertainty_m),
                "A",
                "B",
            ),
        ),
    )
    tests_out = []
    for t in TESTS.values():
        st, reason = status[t.id]
        tests_out.append(
            {
                "id": t.id,
                "label": t.label,
                "severity": t.severity,
                "description": t.description,
                "reference": t.reference,
                "excluded": t.id in cfg.exclude,
                "status": st,
                "reason": reason,
                "n_flagged": int(flagged[t.id].sum()),
            }
        )
    tiers = Counter(tier.tolist())
    summary = {
        "version": QUALITY_VERSION,
        "config": cfg.as_dict(),
        "reference": refs.describe() if refs is not None else None,
        "n_records": n,
        "n_excluded": int(excluded.sum()),
        "n_flagged_kept": int((warned & ~excluded).sum()),
        "tiers": {k: int(tiers.get(k, 0)) for k in ("A", "B", "C", "excluded")},
        "tier_rules": {
            "A": "specimen / material sample with stated uncertainty "
            f"≤ {cfg.tier_a_max_uncertainty_m:g} m and no flags",
            "B": "other records with stated uncertainty and no flags",
            "C": "warning flags, or no stated coordinate uncertainty",
            "excluded": "flagged by an excluded (error) test",
        },
        "tests": tests_out,
        **details,
    }
    return QualityResult(flags=flags, tier=tier, excluded=excluded, summary=summary)


def _environmental_outliers(
    occ: pd.DataFrame,
    predictors: Sequence[str],
    cfg: QualityConfig,
    seed: int,
    run: Any,
    skip: Any,
    details: dict[str, Any],
) -> None:
    preds = [p for p in predictors if p in occ]
    if not preds:
        skip("environmental_outlier", "no predictor values")
        return
    x = occ[preds].to_numpy(dtype="float64")
    ok = ~np.isnan(x).any(axis=1)
    if ok.sum() < max(cfg.outlier_min_records, len(preds) + 2):
        skip("environmental_outlier", f"fewer than {cfg.outlier_min_records} records")
        return
    xv = x[ok]
    sd = xv.std(axis=0)
    sd[sd == 0] = 1.0
    z = (xv - xv.mean(axis=0)) / sd
    rng = np.random.default_rng(seed)
    fit = (
        z
        if len(z) <= cfg.outlier_fit_max_records
        else z[rng.choice(len(z), cfg.outlier_fit_max_records, replace=False)]
    )
    try:
        mcd = MinCovDet(random_state=seed).fit(fit)
        d2 = mcd.mahalanobis(z)
    except (ValueError, np.linalg.LinAlgError) as exc:
        skip("environmental_outlier", f"robust covariance failed ({exc})")
        return
    cutoff = float(chi2.ppf(cfg.outlier_probability, df=len(preds)))
    mask = np.zeros(len(occ), dtype=bool)
    mask[ok] = d2 > cutoff
    run("environmental_outlier", mask)
    details["environmental_outlier_cutoff"] = {
        "predictors": preds,
        "chi2_df": len(preds),
        "probability": cfg.outlier_probability,
        "squared_distance_cutoff": round(cutoff, 3),
    }


# ------------------------------------------------------------ independence
def overlap_with_occurrences(
    ref_lon: Sequence[float],
    ref_lat: Sequence[float],
    ref_dates: Sequence[Any],
    occ_lon: Sequence[float],
    occ_lat: Sequence[float],
    occ_dates: Sequence[Any],
    distance_m: float = 100.0,
    max_days: int = 1,
) -> np.ndarray:
    """True for reference records that duplicate an occurrence already in the store: one
    within `distance_m`, and (when both are dated) recorded within `max_days` of it.

    An agency survey is often also published to GBIF; such records are not independent of
    the training data and are left out of the independent evaluation.
    """
    n = len(ref_lon)
    if n == 0 or len(occ_lon) == 0:
        return np.zeros(n, dtype=bool)
    tree = BallTree(
        _radians(np.asarray(occ_lon, float), np.asarray(occ_lat, float)), metric="haversine"
    )
    near = tree.query_radius(
        _radians(np.asarray(ref_lon, float), np.asarray(ref_lat, float)),
        r=distance_m / EARTH_RADIUS_M,
    )
    rd = pd.to_datetime(pd.Series(list(ref_dates)), errors="coerce").to_numpy()
    od = pd.to_datetime(pd.Series(list(occ_dates)), errors="coerce").to_numpy()
    out = np.zeros(n, dtype=bool)
    tolerance = np.timedelta64(max_days, "D")
    for i, idx in enumerate(near):
        if len(idx) == 0:
            continue
        if pd.isna(rd[i]):
            out[i] = True
            continue
        cand = od[idx]
        undated = pd.isna(cand)
        out[i] = bool(undated.any() or (np.abs(cand[~undated] - rd[i]) <= tolerance).any())
    return out
