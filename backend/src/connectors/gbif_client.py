"""GBIF connector. The only module allowed to call `pygbif` directly.

* First-time species: DOI-backed `occurrences.download` (requires GBIF credentials via env).
  Falls back to paged `occurrences.search` when credentials are not configured.
* Repeat species: incremental `occurrences.search` filtered on `lastInterpreted`.
"""

from __future__ import annotations

import csv
import io
import logging
import math
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.domain import OccurrenceRecord, RangeLabel

log = logging.getLogger(__name__)

# GBIF dataset key of the "iNaturalist Research-grade Observations" dataset. Records from it
# carry the iNat observation id as `catalogNumber`, which is used to de-duplicate iNat pulls.
INAT_GBIF_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"

_EXCLUDED_BASIS = {"FOSSIL_SPECIMEN", "LIVING_SPECIMEN"}
_EXCLUDED_ESTABLISHMENT = {"managed", "cultivated", "captive"}
_SEARCH_PAGE = 300
# The search API stops responding reliably past offset 10,000 (responses trickle and never
# complete), so paging is confined to this window and larger queries are sliced spatially.
SEARCH_WINDOW = 9_900
_MIN_SLICE_DEG = 0.01
# (connect, read) timeout for every GBIF HTTP call: the API occasionally stalls a
# connection indefinitely, and pygbif sets no timeout of its own.
HTTP_TIMEOUT = (10, 60)
MAX_ATTEMPTS = 4
# Rate limiting (429) is expected when paging hundreds of thousands of records: wait longer.
MAX_RATE_LIMITED_ATTEMPTS = 8
RATE_LIMIT_MAX_WAIT_S = 120.0
MIN_CALL_INTERVAL_S = 0.2

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class TaxonSuggestion:
    taxon_key: int
    scientific_name: str
    canonical_name: str | None
    rank: str | None
    kingdom: str | None
    phylum: str | None
    class_: str | None
    order: str | None
    family: str | None
    genus: str | None


@dataclass(frozen=True, slots=True)
class DownloadResult:
    download_key: str
    doi: str | None
    records: list[OccurrenceRecord]


def map_range_label(establishment_means: str | None, degree: str | None = None) -> RangeLabel:
    """Map GBIF establishmentMeans / degreeOfEstablishment vocab onto native/introduced/unknown.

    These fields are sparse for most taxa, which is why native range is never defined from
    them alone (see modeling/native_range.py and CLAUDE.md constraint 6).
    """
    em = (establishment_means or "").strip().lower()
    doe = (degree or "").strip().lower()
    if em.startswith("native") or doe == "native":
        return "native"
    if em.startswith("introduced") or em in {"invasive", "naturalised", "naturalized"}:
        return "introduced"
    if doe in {"invasive", "naturalised", "naturalized", "established", "colonising", "casual"}:
        return "introduced"
    return "unknown"


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    # GBIF eventDate may be an interval ("2019-05-01/2019-05-03") or a partial date.
    text = text.split("/")[0]
    for fmt, length in (("%Y-%m-%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        try:
            return datetime.strptime(text[:length], fmt).date()
        except ValueError:
            continue
    return None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_record(
    raw: Mapping[str, Any], taxon_key: int, max_uncertainty_m: float
) -> OccurrenceRecord | None:
    """Clean one GBIF record (search JSON or SIMPLE_CSV row). Returns None if unusable."""
    gbif_id = raw.get("gbifID") or raw.get("key")
    lat = _to_float(raw.get("decimalLatitude"))
    lon = _to_float(raw.get("decimalLongitude"))
    if gbif_id in (None, "") or lat is None or lon is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None
    if str(raw.get("occurrenceStatus") or "PRESENT").upper() == "ABSENT":
        return None
    basis = raw.get("basisOfRecord") or None
    if basis in _EXCLUDED_BASIS:
        return None
    em = raw.get("establishmentMeans")
    doe = raw.get("degreeOfEstablishment")
    if str(em or "").lower() in _EXCLUDED_ESTABLISHMENT or str(doe or "").lower() in (
        _EXCLUDED_ESTABLISHMENT
    ):
        return None
    uncertainty = _to_float(raw.get("coordinateUncertaintyInMeters"))
    if uncertainty is not None and uncertainty > max_uncertainty_m:
        return None

    crossref: int | None = None
    if raw.get("datasetKey") == INAT_GBIF_DATASET_KEY:
        try:
            crossref = int(str(raw.get("catalogNumber")))
        except (TypeError, ValueError):
            crossref = None

    return OccurrenceRecord(
        taxon_key=taxon_key,
        source="GBIF",
        record_id=int(gbif_id),
        longitude=lon,
        latitude=lat,
        event_date=_parse_date(raw.get("eventDate")),
        coordinate_uncertainty_m=uncertainty,
        basis_of_record=basis,
        country_code=raw.get("countryCode") or None,
        range_label=map_range_label(em, doe),
        inat_crossref_id=crossref,
    )


class GbifClient:
    """Thin, mockable wrapper around pygbif."""

    def __init__(
        self,
        user: str | None = None,
        pwd: str | None = None,
        email: str | None = None,
        max_uncertainty_m: float = 10_000.0,
        poll_seconds: int = 30,
        timeout_seconds: int = 3 * 60 * 60,
        search_max_records: int = 100_000,
        sleep: Callable[[float], None] = time.sleep,
        timeout: tuple[float, float] = HTTP_TIMEOUT,
    ) -> None:
        from pygbif import occurrences, species  # local import keeps tests import-light

        self._occ = occurrences
        self._species = species
        self._user, self._pwd, self._email = user, pwd, email
        self.max_uncertainty_m = max_uncertainty_m
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.search_max_records = search_max_records
        self._sleep = sleep
        self._timeout = timeout
        self._last_call = 0.0

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call a pygbif function politely: paced, with a request timeout, and with backoff on
        timeouts, connection errors and 5xx — and on 429 (rate limited), honouring GBIF's
        Retry-After and allowing more, longer waits, since large species need many calls."""
        import requests

        attempt = 0
        while True:
            attempt += 1
            self._pace()
            try:
                return fn(*args, timeout=self._timeout, **kwargs)
            except (requests.Timeout, requests.ConnectionError) as exc:
                err: Exception = exc
                limit, wait = MAX_ATTEMPTS, 2.0**attempt
            except requests.HTTPError as exc:
                resp = exc.response
                code = resp.status_code if resp is not None else 0
                if code == 429:
                    err = exc
                    limit = MAX_RATE_LIMITED_ATTEMPTS
                    wait = _retry_after(resp) or min(RATE_LIMIT_MAX_WAIT_S, 5.0 * 2**attempt)
                elif code >= 500:
                    err, limit, wait = exc, MAX_ATTEMPTS, 2.0**attempt
                else:
                    raise
            if attempt >= limit:
                raise err
            log.warning(
                "GBIF call failed (%s); retry %d/%d in %.0fs", err, attempt, limit - 1, wait
            )
            self._sleep(wait)

    def _pace(self) -> None:
        """Keep at least MIN_CALL_INTERVAL_S between calls (GBIF asks clients not to burst)."""
        gap = MIN_CALL_INTERVAL_S - (time.monotonic() - self._last_call)
        if gap > 0:
            self._sleep(gap)
        self._last_call = time.monotonic()

    @property
    def download_enabled(self) -> bool:
        return bool(self._user and self._pwd and self._email)

    # ------------------------------------------------------------------ species
    def suggest_species(self, query: str, limit: int = 10) -> list[TaxonSuggestion]:
        raw = self._call(self._species.name_suggest, q=query, limit=limit)
        items = raw.get("results", []) if isinstance(raw, dict) else list(raw or [])
        out: list[TaxonSuggestion] = []
        for item in items:
            key = item.get("key") or item.get("usageKey")
            if key is None:
                continue
            out.append(
                TaxonSuggestion(
                    taxon_key=int(key),
                    scientific_name=item.get("scientificName") or item.get("canonicalName", ""),
                    canonical_name=item.get("canonicalName"),
                    rank=item.get("rank"),
                    kingdom=item.get("kingdom"),
                    phylum=item.get("phylum"),
                    class_=item.get("class"),
                    order=item.get("order"),
                    family=item.get("family"),
                    genus=item.get("genus"),
                )
            )
        return out

    def get_taxon(self, taxon_key: int) -> TaxonSuggestion:
        item = self._call(self._species.name_usage, key=taxon_key)
        return TaxonSuggestion(
            taxon_key=int(item.get("key", taxon_key)),
            scientific_name=item.get("scientificName") or item.get("canonicalName", ""),
            canonical_name=item.get("canonicalName"),
            rank=item.get("rank"),
            kingdom=item.get("kingdom"),
            phylum=item.get("phylum"),
            class_=item.get("class"),
            order=item.get("order"),
            family=item.get("family"),
            genus=item.get("genus"),
        )

    # --------------------------------------------------------- full download
    def download_occurrences(self, taxon_key: int, workdir: Path) -> DownloadResult:
        """DOI-backed GBIF download (first-time species). Blocks until ready; run in a worker."""
        if not self.download_enabled:
            raise RuntimeError("GBIF download API requires SDM_GBIF_USER/PWD/EMAIL")
        queries = [
            f"taxonKey = {taxon_key}",
            "hasCoordinate = TRUE",
            "hasGeospatialIssue = FALSE",
            "occurrenceStatus = PRESENT",
        ]
        resp = self._occ.download(
            queries, format="SIMPLE_CSV", user=self._user, pwd=self._pwd, email=self._email
        )
        download_key = resp[0] if isinstance(resp, (tuple, list)) else str(resp)
        log.info("GBIF download %s requested for taxon %s", download_key, taxon_key)

        deadline = time.monotonic() + self.timeout_seconds
        meta: dict[str, Any] = {}
        while time.monotonic() < deadline:
            meta = self._call(self._occ.download_meta, download_key)
            status = str(meta.get("status", "")).upper()
            if status == "SUCCEEDED":
                break
            if status in {"FAILED", "KILLED", "CANCELLED"}:
                raise RuntimeError(f"GBIF download {download_key} ended with status {status}")
            self._sleep(self.poll_seconds)
        else:
            raise TimeoutError(f"GBIF download {download_key} did not finish in time")

        workdir.mkdir(parents=True, exist_ok=True)
        got = self._occ.download_get(download_key, path=str(workdir))
        zip_path = Path(got["path"] if isinstance(got, dict) else got)
        records = list(self.parse_simple_csv_zip(zip_path, taxon_key))
        return DownloadResult(download_key=download_key, doi=meta.get("doi"), records=records)

    def parse_simple_csv_zip(self, zip_path: Path, taxon_key: int) -> Iterator[OccurrenceRecord]:
        with zipfile.ZipFile(zip_path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
                reader = csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
                yield from self._normalize_many(reader, taxon_key)

    # ------------------------------------------------------- paged search
    def search_occurrences(
        self,
        taxon_key: int,
        since: datetime | None = None,
        until: datetime | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[OccurrenceRecord]:
        """Paged occurrence search; with `since`, only records (re)interpreted since then.

        GBIF's search API stalls beyond offset ~10,000, so larger result sets are split into
        spatial slices (lat/lon ranges) that each fit in that window. Records on slice
        boundaries are de-duplicated by gbifID. `on_progress(fetched, total)` runs per page.
        """
        params: dict[str, Any] = {
            "taxonKey": taxon_key,
            "hasCoordinate": True,
            "hasGeospatialIssue": False,
            "occurrenceStatus": "PRESENT",
        }
        if since is not None:
            end = until or datetime.now(UTC)
            # GBIF only accepts day-precision dates here (a time component returns 400).
            # Overlap with the previous fetch day is harmless: re-seen records are
            # de-duplicated on insert and never count towards the retrain delta.
            params["lastInterpreted"] = f"{_day(since)},{_day(end)}"

        total = self._count(params)
        if total == 0:
            return []
        if total > self.search_max_records:
            log.warning(
                "Taxon %s has %d records; search fallback capped at %d — configure GBIF "
                "credentials to use the DOI-backed download API instead",
                taxon_key,
                total,
                self.search_max_records,
            )
        slices = (
            [(params, total)]
            if total <= SEARCH_WINDOW
            else self._partition(params, (-90.0, 90.0), (-180.0, 180.0))
        )
        target = min(total, self.search_max_records)
        # Above the cap, take the same fraction from every spatial slice so the retained
        # records keep the species' geographic spread (stopping after the first slices would
        # silently drop whole regions).
        share = target / total

        out: dict[int, OccurrenceRecord] = {}
        fetched = 0
        for sp, n in slices:
            quota = min(SEARCH_WINDOW, n if share >= 1 else max(1, math.ceil(n * share)))
            offset = 0
            while offset < quota:
                limit = min(_SEARCH_PAGE, quota - offset)
                page = self._call(self._occ.search, limit=limit, offset=offset, **sp)
                results = page.get("results", [])
                for rec in self._normalize_many(results, taxon_key):
                    out[rec.record_id] = rec
                offset += len(results)
                fetched += len(results)
                if on_progress is not None:
                    on_progress(min(fetched, target), target)
                if page.get("endOfRecords", True) or not results:
                    break
        return list(out.values())

    def _count(self, params: dict[str, Any]) -> int:
        return int(self._call(self._occ.search, limit=0, **params).get("count", 0))

    def _partition(
        self,
        params: dict[str, Any],
        lat: tuple[float, float],
        lon: tuple[float, float],
    ) -> list[tuple[dict[str, Any], int]]:
        """Recursively bisect a lat/lon box until each slice holds ≤ SEARCH_WINDOW records.

        Returns (slice query, record count) pairs."""
        sp = {
            **params,
            "decimalLatitude": f"{lat[0]},{lat[1]}",
            "decimalLongitude": f"{lon[0]},{lon[1]}",
        }
        n = self._count(sp)
        if n == 0:
            return []
        if n <= SEARCH_WINDOW:
            return [(sp, n)]
        if lat[1] - lat[0] >= lon[1] - lon[0] and lat[1] - lat[0] > _MIN_SLICE_DEG:
            mid = (lat[0] + lat[1]) / 2
            return self._partition(params, (lat[0], mid), lon) + self._partition(
                params, (mid, lat[1]), lon
            )
        if lon[1] - lon[0] > _MIN_SLICE_DEG:
            mid = (lon[0] + lon[1]) / 2
            return self._partition(params, lat, (lon[0], mid)) + self._partition(
                params, lat, (mid, lon[1])
            )
        log.warning(
            "%d records share a %s° cell; only the first %d are retrievable via search",
            n,
            _MIN_SLICE_DEG,
            SEARCH_WINDOW,
        )
        return [(sp, n)]

    def _normalize_many(
        self, rows: Iterable[Mapping[str, Any]], taxon_key: int
    ) -> Iterator[OccurrenceRecord]:
        for row in rows:
            rec = normalize_record(row, taxon_key, self.max_uncertainty_m)
            if rec is not None:
                yield rec


def _retry_after(resp: Any) -> float | None:
    """Seconds from a Retry-After header (numeric form), capped; None if absent/unparseable."""
    try:
        value = float(resp.headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return None
    return min(max(value, 1.0), RATE_LIMIT_MAX_WAIT_S)


def _day(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%d")


def download_citation(
    doi: str | None,
    download_key: str | None,
    on: date | None = None,
    taxon_key: int | None = None,
) -> str:
    """Citation string per GBIF citation guidelines.

    Only downloads are DOI-citable; a search-API pull is cited by its query and flagged so.
    """
    when = (on or date.today()).strftime("%d %B %Y")
    if doi:
        return f"GBIF.org ({when}) GBIF Occurrence Download https://doi.org/{doi}"
    if download_key:
        return f"GBIF.org ({when}) GBIF Occurrence Download {download_key}"
    query = f"https://api.gbif.org/v1/occurrence/search?taxonKey={taxon_key}"
    return (
        f"GBIF.org ({when}) GBIF Occurrence Data, search API query {query} (not DOI-citable; "
        "configure GBIF credentials for a DOI-backed download)"
    )


def inat_citation(inat_taxon_id: int, on: date | None = None) -> str:
    when = (on or date.today()).strftime("%d %B %Y")
    return (
        f"iNaturalist contributors, iNaturalist ({when}). Research-grade observations of taxon "
        f"{inat_taxon_id}, https://api.inaturalist.org/v1/observations?taxon_id={inat_taxon_id}"
    )
