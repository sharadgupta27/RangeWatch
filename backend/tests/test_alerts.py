"""Alert delivery (U10): webhook payloads, SSRF guard, failures recorded, never raised."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from src.domain import Alert, AlertSubscription
from src.monitoring.alerts import (
    WebhookTargetError,
    deliver,
    effort_classes,
    rss_feed,
    validate_webhook_url,
)
from tests.conftest import TAXON_KEY


def alert() -> Alert:
    return Alert(
        TAXON_KEY,
        "outside_envelope",
        2,
        [{"source": "GBIF", "record_id": 1, "longitude": 1, "latitude": 2}],
        "2 new records outside the envelope",
        effort_classes={"expansion": 1, "discovery": 1, "unknown": 0},
    )


def test_webhook_targets_must_be_public_https(settings):
    for url in (
        "http://hooks.example.org/x",
        "https://localhost/x",
        "https://10.0.0.5/x",
        "ftp://example.org/x",
        "https://user:pw@example.org/x",
    ):
        with pytest.raises(WebhookTargetError):
            validate_webhook_url(url, settings)
    allowed = settings.model_copy(update={"alert_webhook_allowed_hosts": ["localhost"]})
    assert validate_webhook_url("https://localhost/x", allowed) == "https://localhost/x"


def test_delivery_posts_json_and_records_failures(settings):
    s = settings.model_copy(update={"alert_webhook_allowed_hosts": ["hooks.test", "down.test"]})
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(500 if request.url.host == "down.test" else 204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    subs = [
        AlertSubscription("webhook", "https://hooks.test/a"),
        AlertSubscription("webhook", "https://down.test/b"),
        AlertSubscription("webhook", "https://hooks.test/c", kinds=["outside_known_range"]),
        AlertSubscription("email", "team@example.org"),  # SMTP not configured
    ]
    out = deliver(alert(), subs, "Testus invasivus", s, client=client)
    assert [d["status"] for d in out] == ["sent", "failed", "failed"]  # 3rd sub filtered out
    assert "SDM_SMTP_HOST" in out[2]["error"]
    assert seen[0]["species"] == "Testus invasivus" and seen[0]["effort_classes"]["expansion"] == 1


def test_effort_classes_without_a_layer_are_unknown_and_feed_escapes(settings):
    assert list(effort_classes(None, [1.0], [2.0], 10)) == ["unknown"]
    a = alert()
    a.summary = "<script>"
    xml = rss_feed([a], {TAXON_KEY: "A & B"}, settings)
    assert "&lt;script&gt;" in xml and "A &amp; B" in xml
    assert np.asarray(effort_classes(None, [], [], 10)).size == 0
