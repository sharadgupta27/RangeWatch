"""Connector tests — pygbif / pyinaturalist are mocked; no live network."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime
from types import SimpleNamespace

from src.connectors.gbif_client import (
    INAT_GBIF_DATASET_KEY,
    GbifClient,
    download_citation,
    map_range_label,
    normalize_record,
)
from src.connectors.inaturalist_client import InatClient, normalize_observation


def test_map_range_label_vocabulary():
    assert map_range_label("Native") == "native"
    assert map_range_label("NativeReintroduced") == "native"
    assert map_range_label("Introduced") == "introduced"
    assert map_range_label("IntroducedAssistedColonisation") == "introduced"
    assert map_range_label(None, "invasive") == "introduced"
    assert map_range_label(None, "native") == "native"
    assert map_range_label("Uncertain") == "unknown"
    assert map_range_label(None) == "unknown"


def test_normalize_record_cleaning():
    base = {
        "gbifID": "12",
        "decimalLatitude": "45.1",
        "decimalLongitude": "7.2",
        "eventDate": "2019-05-01/2019-05-03",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "countryCode": "IT",
        "establishmentMeans": "introduced",
    }
    rec = normalize_record(base, 1, 10_000)
    assert rec is not None
    assert (rec.record_id, rec.longitude, rec.latitude) == (12, 7.2, 45.1)
    assert rec.event_date == date(2019, 5, 1)
    assert rec.range_label == "introduced"

    assert (
        normalize_record({**base, "decimalLatitude": "0", "decimalLongitude": "0"}, 1, 1e4) is None
    )
    assert normalize_record({**base, "basisOfRecord": "FOSSIL_SPECIMEN"}, 1, 1e4) is None
    assert normalize_record({**base, "basisOfRecord": "LIVING_SPECIMEN"}, 1, 1e4) is None
    assert normalize_record({**base, "occurrenceStatus": "ABSENT"}, 1, 1e4) is None
    assert normalize_record({**base, "degreeOfEstablishment": "cultivated"}, 1, 1e4) is None
    assert normalize_record({**base, "coordinateUncertaintyInMeters": "50000"}, 1, 1e4) is None
    assert normalize_record({**base, "decimalLatitude": "95"}, 1, 1e4) is None


def test_normalize_record_inat_crossref():
    raw = {
        "key": 99,
        "decimalLatitude": 10,
        "decimalLongitude": 10,
        "datasetKey": INAT_GBIF_DATASET_KEY,
        "catalogNumber": "123456",
    }
    rec = normalize_record(raw, 1, 1e4)
    assert rec is not None and rec.inat_crossref_id == 123456


def _client_with(occ_mock) -> GbifClient:
    client = GbifClient.__new__(GbifClient)
    client._occ = occ_mock
    client._species = SimpleNamespace()
    client._user = client._pwd = client._email = None
    client.max_uncertainty_m = 10_000
    client.search_max_records = 100_000
    client.poll_seconds = 0
    client.timeout_seconds = 10
    client._sleep = lambda s: None
    client._timeout = (1, 1)
    client._last_call = 0.0
    return client


class FakeGbifSearch:
    """In-memory GBIF search honouring limit/offset, count queries and lat/lon ranges."""

    def __init__(self, n: int, seed: int = 0, stall_after: int | None = None):
        import numpy as np

        rng = np.random.default_rng(seed)
        self.rows = [
            {"key": i, "decimalLatitude": float(y), "decimalLongitude": float(x)}
            for i, (x, y) in enumerate(
                zip(rng.uniform(-170, 170, n), rng.uniform(-60, 70, n), strict=True)
            )
        ]
        self.calls: list[dict] = []
        self.stall_after = stall_after

    @staticmethod
    def _in(v: float, rng: str | None) -> bool:
        if rng is None:
            return True
        lo, hi = (float(x) for x in rng.split(","))
        return lo <= v <= hi

    def search(self, **kw):
        kw.pop("timeout")
        self.calls.append(kw)
        rows = [
            r
            for r in self.rows
            if self._in(r["decimalLatitude"], kw.get("decimalLatitude"))
            and self._in(r["decimalLongitude"], kw.get("decimalLongitude"))
        ]
        limit, offset = kw.get("limit", 300), kw.get("offset", 0)
        if self.stall_after is not None and offset + limit > self.stall_after:
            raise AssertionError("paged beyond the GBIF offset window")
        page = rows[offset : offset + limit]
        return {"results": page, "count": len(rows), "endOfRecords": offset + limit >= len(rows)}


def test_incremental_search_pages_and_uses_last_interpreted():
    fake = FakeGbifSearch(320)
    client = _client_with(fake)
    since = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    progress = []
    out = client.search_occurrences(
        42,
        since=since,
        until=datetime(2025, 2, 1, tzinfo=UTC),
        on_progress=lambda n, t: progress.append((n, t)),
    )
    assert len(out) == 320
    assert progress == [(300, 320), (320, 320)]
    first = fake.calls[0]
    assert first["limit"] == 0  # count query first
    assert first["lastInterpreted"] == "2025-01-02,2025-02-01"  # GBIF: day precision only
    assert first["taxonKey"] == 42 and first["hasGeospatialIssue"] is False
    assert [c["offset"] for c in fake.calls[1:]] == [0, 300]


def test_large_result_sets_are_sliced_spatially_within_the_offset_window(monkeypatch):
    from src.connectors import gbif_client

    monkeypatch.setattr(gbif_client, "SEARCH_WINDOW", 100)
    fake = FakeGbifSearch(1_000, stall_after=100)
    out = _client_with(fake).search_occurrences(1)
    assert sorted(r.record_id for r in out) == list(range(1_000))  # complete, no duplicates
    assert all("decimalLatitude" in c for c in fake.calls[2:])


def test_download_flow_polls_and_parses(tmp_path):
    header = "gbifID\tdecimalLatitude\tdecimalLongitude\teventDate\testablishmentMeans\n"
    rows = "1\t45\t7\t2020-01-01\tnative\n2\t\t\t2020\t\n3\t10\t20\t2021-03\tintroduced\n"
    zpath = tmp_path / "dl.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("0001.csv", header + rows)
    statuses = iter(["RUNNING", "SUCCEEDED"])
    occ = SimpleNamespace(
        download=lambda q, **kw: ("0001-XYZ", {}),
        download_meta=lambda key, **kw: {"status": next(statuses), "doi": "10.15468/dl.abc"},
        download_get=lambda key, path: {"path": str(zpath)},
    )
    client = _client_with(occ)
    client._user, client._pwd, client._email = "u", "p", "e@x"
    res = client.download_occurrences(7, tmp_path / "work")
    assert res.download_key == "0001-XYZ" and res.doi == "10.15468/dl.abc"
    assert [r.record_id for r in res.records] == [1, 3]
    assert [r.range_label for r in res.records] == ["native", "introduced"]


def test_download_citation_prefers_doi():
    c = download_citation("10.15468/dl.abc", "0001", on=date(2026, 3, 1))
    assert c == "GBIF.org (01 March 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.abc"


def test_inat_normalize_and_label():
    obs = {
        "id": 5,
        "location": [45.0, 7.0],
        "observed_on": datetime(2024, 5, 1),
        "positional_accuracy": 20,
    }
    rec = normalize_observation(obs, 1, 1e4, "introduced")
    assert rec is not None
    assert (rec.longitude, rec.latitude, rec.source) == (7.0, 45.0, "iNaturalist")
    assert rec.event_date == date(2024, 5, 1)
    assert normalize_observation({**obs, "obscured": True}, 1, 1e4) is None
    assert normalize_observation({**obs, "positional_accuracy": 50_000}, 1, 1e4) is None


def test_inat_get_observations_labels_by_id_sets():
    def get_observations(**kw):
        if kw["id_above"] > 0:
            return {"results": []}
        if kw.get("introduced"):
            return {"results": [{"id": 2}]}
        if kw.get("native"):
            return {"results": [{"id": 1}]}
        return {
            "results": [
                {"id": 1, "location": "10,10", "observed_on": "2020-01-01"},
                {"id": 2, "location": "11,11", "observed_on": "2021-01-01"},
                {"id": 3, "location": "12,12", "observed_on": "2022-01-01"},
            ]
        }

    client = InatClient.__new__(InatClient)
    client._api = SimpleNamespace(get_observations=get_observations)
    client.max_uncertainty_m = 1e4
    client.max_records = 1000
    recs = client.get_observations(777, 1)
    assert {r.record_id: r.range_label for r in recs} == {
        1: "native",
        2: "introduced",
        3: "unknown",
    }


def test_inat_paging_backs_off_when_throttled():
    import requests

    waits: list[float] = []
    calls = {"n": 0}

    def throttled(**kw):
        calls["n"] += 1
        if calls["n"] <= 3:
            resp = requests.Response()
            resp.status_code = 429
            raise requests.HTTPError(response=resp)
        return {"results": []}

    client = InatClient.__new__(InatClient)
    client._api = SimpleNamespace(get_observations=throttled)
    client.max_uncertainty_m = 1e4
    client.max_records = 1000
    client._sleep = waits.append
    assert client.get_observations(777, 1) == []
    assert waits == [20.0, 40.0, 80.0]  # exponential backoff without a Retry-After header


def test_inat_client_errors_are_not_retried():
    import pytest
    import requests

    def bad_request(**kw):
        resp = requests.Response()
        resp.status_code = 422
        raise requests.HTTPError(response=resp)

    client = InatClient.__new__(InatClient)
    client._api = SimpleNamespace(get_observations=bad_request)
    client.max_uncertainty_m = 1e4
    client.max_records = 1000
    client._sleep = lambda s: pytest.fail("should not retry a 4xx")
    with pytest.raises(requests.HTTPError):
        client.get_observations(777, 1)


def test_inat_progress_reports_a_stable_total():
    ids = list(range(1, 451))

    def get_observations(**kw):
        above = [i for i in ids if i > kw["id_above"]]  # keyset: total counts ids > cursor
        return {"results": [{"id": i} for i in above[:200]], "total_results": len(above)}

    client = InatClient.__new__(InatClient)
    client._api = SimpleNamespace(get_observations=get_observations)
    client.max_uncertainty_m = 1e4
    client.max_records = 1000
    seen: list[tuple[int, int]] = []
    list(client._paged({}, only_id=True, on_progress=lambda f, t: seen.append((f, t))))
    assert seen == [(200, 450), (400, 450), (450, 450)]


def test_parse_zip_handles_utf8(tmp_path):
    zpath = tmp_path / "u.zip"
    buf = io.StringIO("gbifID\tdecimalLatitude\tdecimalLongitude\tlocality\n1\t1\t1\tZürich\n")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("x.csv", buf.getvalue())
    client = _client_with(SimpleNamespace())
    assert len(list(client.parse_simple_csv_zip(zpath, 1))) == 1


def test_gbif_calls_time_out_and_retry_transient_failures():
    import requests

    attempts = []

    def flaky(**kw):
        attempts.append(kw["timeout"])
        if len(attempts) < 3:
            raise requests.Timeout("stalled")
        return {"results": [], "endOfRecords": True, "count": 0}

    client = _client_with(SimpleNamespace(search=flaky))
    assert client.search_occurrences(1) == []
    assert attempts == [(1, 1)] * 3  # every call carries the timeout; 2 retries, then success


def test_gbif_client_errors_are_not_retried():
    import pytest
    import requests

    calls = []

    def bad_request(**kw):
        calls.append(1)
        resp = requests.Response()
        resp.status_code = 400
        raise requests.HTTPError(response=resp)

    client = _client_with(SimpleNamespace(search=bad_request))
    with pytest.raises(requests.HTTPError):
        client.search_occurrences(1)
    assert len(calls) == 1


def test_search_api_citation_is_not_presented_as_a_download():
    c = download_citation(None, None, on=date(2026, 3, 1), taxon_key=42)
    assert "None" not in c and "Download" not in c
    assert "taxonKey=42" in c and "not DOI-citable" in c


def test_capped_search_samples_every_region_proportionally(monkeypatch):
    """Above the record cap the fallback must keep the geographic spread, not the first N."""
    from src.connectors import gbif_client

    monkeypatch.setattr(gbif_client, "SEARCH_WINDOW", 100)
    fake = FakeGbifSearch(1_000, seed=3)
    client = _client_with(fake)
    client.search_max_records = 300
    out = client.search_occurrences(1)
    assert 300 <= len(out) <= 340  # per-slice rounding up
    lats = [r.latitude for r in out]
    # Both hemispheres / bands represented roughly in proportion (source is uniform -60..70).
    south = sum(lat < 0 for lat in lats) / len(lats)
    assert 0.3 < south < 0.6


def test_rate_limiting_honours_retry_after_and_retries_longer():
    import requests

    waits: list[float] = []
    calls = {"n": 0}

    def limited(**kw):
        calls["n"] += 1
        if calls["n"] <= 5:  # more 429s than the generic retry budget (4 attempts)
            resp = requests.Response()
            resp.status_code = 429
            resp.headers["Retry-After"] = "7"
            raise requests.HTTPError(response=resp)
        return {"results": [], "endOfRecords": True, "count": 0}

    client = _client_with(SimpleNamespace(search=limited))
    client._sleep = waits.append
    assert client.search_occurrences(1) == []
    assert [w for w in waits if w > 1] == [7.0] * 5  # Retry-After respected on every 429
