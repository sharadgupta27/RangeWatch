"""iNaturalist connector. The only module allowed to call `pyinaturalist` directly.

Used for (a) taxon autocomplete metadata (common names, thumbnails) and (b) supplementary,
fresher research-grade observations. Records already present in GBIF via the iNaturalist
research-grade dataset are de-duplicated by the pipeline using `inat_crossref_id`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from src.domain import OccurrenceRecord, RangeLabel

log = logging.getLogger(__name__)

_PER_PAGE = 200
# pyinaturalist already paces requests (~1/s); these cover the throttling iNaturalist still
# applies to long paging runs ("normal_throttling" 429s) and transient server errors.
MAX_ATTEMPTS = 4
MAX_RATE_LIMITED_ATTEMPTS = 6
RATE_LIMIT_MAX_WAIT_S = 120.0


@dataclass(frozen=True, slots=True)
class InatTaxon:
    inat_taxon_id: int
    name: str
    common_name: str | None
    rank: str | None
    thumbnail_url: str | None
    observations_count: int | None


def _parse_location(obs: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return (lon, lat) from the various shapes the API / pyinaturalist produce."""
    loc = obs.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        return float(loc[1]), float(loc[0])
    if isinstance(loc, str) and "," in loc:
        lat, lon = loc.split(",", 1)
        return float(lon), float(lat)
    geo = obs.get("geojson") or {}
    coords = geo.get("coordinates") if isinstance(geo, Mapping) else None
    if coords and len(coords) == 2:
        return float(coords[0]), float(coords[1])
    return None


def _parse_obs_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def normalize_observation(
    obs: Mapping[str, Any],
    taxon_key: int,
    max_uncertainty_m: float,
    label: RangeLabel = "unknown",
) -> OccurrenceRecord | None:
    if obs.get("obscured") or obs.get("captive"):
        return None
    xy = _parse_location(obs)
    if xy is None:
        return None
    lon, lat = xy
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None
    acc = obs.get("positional_accuracy")
    acc_f = float(acc) if acc not in (None, "") else None
    if acc_f is not None and acc_f > max_uncertainty_m:
        return None
    return OccurrenceRecord(
        taxon_key=taxon_key,
        source="iNaturalist",
        record_id=int(obs["id"]),
        longitude=lon,
        latitude=lat,
        event_date=_parse_obs_date(obs.get("observed_on")),
        coordinate_uncertainty_m=acc_f,
        basis_of_record="HUMAN_OBSERVATION",
        country_code=None,
        range_label=label,
    )


class InatClient:
    """Thin, mockable wrapper around pyinaturalist."""

    def __init__(
        self,
        max_uncertainty_m: float = 10_000.0,
        max_records: int = 20_000,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        import pyinaturalist

        self._api = pyinaturalist
        self.max_uncertainty_m = max_uncertainty_m
        self.max_records = max_records
        self._sleep = sleep

    def autocomplete(self, query: str, limit: int = 10) -> list[InatTaxon]:
        resp = self._api.get_taxa_autocomplete(q=query, per_page=limit)
        return [_to_taxon(t) for t in resp.get("results", [])]

    def find_taxon(self, scientific_name: str) -> InatTaxon | None:
        resp = self._api.get_taxa(q=scientific_name, rank="species", per_page=5)
        for t in resp.get("results", []):
            if str(t.get("name", "")).lower() == scientific_name.lower():
                return _to_taxon(t)
        return None

    def get_observations(
        self,
        inat_taxon_id: int,
        taxon_key: int,
        updated_since: datetime | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[OccurrenceRecord]:
        """Research-grade observations, labelled via the API's place-aware native/introduced
        filters (two cheap id-only passes). `on_progress(fetched, total)` runs per page of the
        main pass."""
        base: dict[str, Any] = {
            "taxon_id": inat_taxon_id,
            "quality_grade": "research",
            "geoprivacy": "open",
            "taxon_geoprivacy": "open",
            "captive": False,
        }
        if updated_since is not None:
            base["updated_since"] = updated_since
        introduced = set(self._ids({**base, "introduced": True}))
        native = set(self._ids({**base, "native": True}))

        out: list[OccurrenceRecord] = []
        for obs in self._paged(base, only_id=False, on_progress=on_progress):
            oid = int(obs["id"])
            label: RangeLabel = (
                "introduced" if oid in introduced else "native" if oid in native else "unknown"
            )
            rec = normalize_observation(obs, taxon_key, self.max_uncertainty_m, label)
            if rec is not None:
                out.append(rec)
        return out

    def _ids(self, params: dict[str, Any]) -> Iterable[int]:
        for obs in self._paged(params, only_id=True):
            yield int(obs["id"])

    def _paged(
        self,
        params: dict[str, Any],
        only_id: bool,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Iterable[Mapping[str, Any]]:
        """Id-ordered keyset pagination (the API caps page*per_page at 10k)."""
        id_above = 0
        fetched = 0
        while fetched < self.max_records:
            resp = self._get_observations(
                **params,
                id_above=id_above,
                order_by="id",
                order="asc",
                per_page=_PER_PAGE,
                only_id=only_id or None,
            )
            results = resp.get("results", [])
            if not results:
                return
            # `total_results` counts only ids above the cursor, so add what came before.
            remaining = int(resp.get("total_results") or len(results))
            yield from results
            total = min(fetched + remaining, self.max_records)
            fetched += len(results)
            if on_progress is not None:
                on_progress(fetched, max(total, fetched))
            id_above = int(results[-1]["id"])
            if len(results) < _PER_PAGE:
                return

    def _get_observations(self, **params: Any) -> dict[str, Any]:
        """`get_observations` with backoff on 429 (honouring Retry-After), 5xx, timeouts and
        connection errors. Only used on the worker's paging path — request-thread lookups
        (autocomplete) fail fast instead."""
        import requests

        get: Callable[..., dict[str, Any]] = self._api.get_observations
        attempt = 0
        while True:
            attempt += 1
            try:
                return get(**params)
            except (requests.Timeout, requests.ConnectionError) as exc:
                err: Exception = exc
                limit, wait = MAX_ATTEMPTS, 2.0**attempt
            except requests.HTTPError as exc:
                resp = exc.response
                code = resp.status_code if resp is not None else 0
                if code == 429:
                    err, limit = exc, MAX_RATE_LIMITED_ATTEMPTS
                    wait = _retry_after(resp) or min(RATE_LIMIT_MAX_WAIT_S, 10.0 * 2**attempt)
                elif code >= 500:
                    err, limit, wait = exc, MAX_ATTEMPTS, 2.0**attempt
                else:
                    raise
            if attempt >= limit:
                raise err
            log.warning(
                "iNaturalist call failed (%s); retry %d/%d in %.0fs",
                err,
                attempt,
                limit - 1,
                wait,
            )
            self._sleep(wait)


def _retry_after(resp: Any) -> float | None:
    """Seconds from a Retry-After header (numeric form), capped; None if absent/unparseable."""
    try:
        value = float(resp.headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return None
    return min(max(value, 1.0), RATE_LIMIT_MAX_WAIT_S)


def _to_taxon(t: Mapping[str, Any]) -> InatTaxon:
    photo = t.get("default_photo") or {}
    return InatTaxon(
        inat_taxon_id=int(t["id"]),
        name=str(t.get("name", "")),
        common_name=t.get("preferred_common_name"),
        rank=t.get("rank"),
        thumbnail_url=photo.get("square_url") or photo.get("url"),
        observations_count=t.get("observations_count"),
    )
