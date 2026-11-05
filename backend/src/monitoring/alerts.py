"""Early-detection alerts (U10) with the effort-aware classification of B10.

After every ingestion the pipeline checks the *new* records (not seen before, not alerted
before) against the current model version:

  * outside_envelope: suitability below the version's threshold (the same test the retrain
    trigger uses — reused, never duplicated, and never changing the trigger);
  * outside_known_range: farther than `alert_known_range_km` from every previously stored
    record and outside the confirmed native range.

Each record is classified with the target group's effort raster of the version (B10):
"expansion" when the cell already held ≥ `effort_sampled_min_records` target-group records
(observers were there before, so the species is probably new), "discovery" when it held
fewer (observers just reached it), "unknown" without an effort layer.

Alerts are stored (append-only) and delivered to subscriptions by webhook (JSON POST) or
e-mail (SMTP from settings), and published as an RSS feed. Webhook URLs must be https and
resolve to public addresses (no SSRF into the deployment's network) unless explicitly allowed.
"""

from __future__ import annotations

import ipaddress
import logging
import smtplib
import socket
from collections.abc import Sequence
from datetime import datetime
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.sax.saxutils import escape

import httpx
import numpy as np

from src.config import Settings
from src.domain import Alert, AlertSubscription
from src.features.raster_sampler import sample_raster_file

log = logging.getLogger(__name__)
MAX_RECORDS_PER_ALERT = 500
WEBHOOK_TIMEOUT_S = 10.0
EFFORT_CLASSES = ("expansion", "discovery", "unknown")


def effort_classes(
    effort_path: Path | None, lon: Sequence[float], lat: Sequence[float], min_records: int
) -> np.ndarray:
    """B10 class per point from a version's effort.tif (log10(records + 1), nodata where the
    target group has no records)."""
    n = len(lon)
    if effort_path is None or not effort_path.exists():
        return np.full(n, "unknown", dtype=object)
    v = sample_raster_file(effort_path, lon, lat).astype("float64")
    sampled = np.nan_to_num(v, nan=0.0) >= np.log10(min_records + 1.0)
    return np.where(sampled, "expansion", "discovery").astype(object)


def class_counts(classes: Sequence[str]) -> dict[str, int]:
    arr = np.asarray(classes, dtype=object)
    return {c: int((arr == c).sum()) for c in EFFORT_CLASSES}


def alert_summary(kind: str, n: int, name: str, counts: dict[str, int], detail: str) -> str:
    what = {
        "outside_envelope": "outside the current suitability envelope",
        "outside_known_range": "far from every known record",
    }[kind]
    return (
        f"{n} new {name} record{'s' if n != 1 else ''} {what} ({detail}). Effort check: "
        f"{counts['expansion']} in previously sampled cells (likely real expansion), "
        f"{counts['discovery']} in previously unsampled cells (likely discovery)"
        + (f", {counts['unknown']} without effort data" if counts["unknown"] else "")
        + "."
    )


# ---------------------------------------------------------------- delivery
class WebhookTargetError(ValueError):
    pass


def validate_webhook_url(url: str, settings: Settings) -> str:
    """https only, no credentials, and the host must resolve to public addresses — unless
    the host is explicitly allowed (SDM_ALERT_WEBHOOK_ALLOWED_HOSTS)."""
    parsed = urlparse(url)
    allowed_schemes = ("https", "http") if settings.alert_webhook_allow_http else ("https",)
    if parsed.scheme not in allowed_schemes or not parsed.hostname:
        raise WebhookTargetError(f"Webhook URL must use {' or '.join(allowed_schemes)}")
    if parsed.username or parsed.password:
        raise WebhookTargetError("Webhook URL must not embed credentials")
    host = parsed.hostname
    if settings.alert_webhook_allowed_hosts and host in settings.alert_webhook_allowed_hosts:
        return url
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise WebhookTargetError(f"Webhook host {host!r} does not resolve") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise WebhookTargetError(f"Webhook host {host!r} resolves to a non-public address")
    return url


def alert_payload(alert: Alert, species_name: str, settings: Settings) -> dict[str, Any]:
    return {
        "alert_id": str(alert.alert_id),
        "taxon_key": alert.taxon_key,
        "species": species_name,
        "kind": alert.kind,
        "created_ts": alert.created_ts.isoformat(),
        "model_version": alert.model_version,
        "n_records": alert.n_records,
        "effort_classes": alert.effort_classes,
        "summary": alert.summary,
        "records": alert.records,
        "url": f"{settings.api_public_url}/species/{alert.taxon_key}/alerts",
    }


def _send_email(to: str, subject: str, body: str, settings: Settings) -> None:
    if not settings.smtp_host:
        raise RuntimeError("e-mail delivery is not configured (SDM_SMTP_HOST)")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = settings.smtp_from, to, subject
    msg.set_content(body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=WEBHOOK_TIMEOUT_S) as smtp:
        if settings.smtp_starttls:
            smtp.starttls()
        if settings.smtp_user and settings.smtp_password:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)


def deliver(
    alert: Alert,
    subscriptions: Sequence[AlertSubscription],
    species_name: str,
    settings: Settings,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Send an alert to every matching subscription; failures are recorded, never raised."""
    out: list[dict[str, Any]] = []
    payload = alert_payload(alert, species_name, settings)
    for sub in subscriptions:
        if alert.kind not in sub.kinds:
            continue
        entry: dict[str, Any] = {
            "subscription_id": str(sub.subscription_id),
            "channel": sub.channel,
            "ts": datetime.now(alert.created_ts.tzinfo).isoformat(),
        }
        try:
            if sub.channel == "webhook":
                validate_webhook_url(sub.target, settings)
                http = client or httpx.Client(timeout=WEBHOOK_TIMEOUT_S, follow_redirects=False)
                try:
                    resp = http.post(sub.target, json=payload)
                    resp.raise_for_status()
                finally:
                    if client is None:
                        http.close()
            else:
                _send_email(
                    sub.target,
                    f"RangeWatch alert: {species_name} ({alert.kind.replace('_', ' ')})",
                    f"{alert.summary}\n\n{payload['url']}",
                    settings,
                )
            entry["status"] = "sent"
        except Exception as exc:  # delivery must never break ingestion
            log.warning("Alert delivery to %s failed: %s", sub.channel, exc)
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        out.append(entry)
    return out


def rss_feed(alerts: Sequence[Alert], names: dict[int, str], settings: Settings) -> str:
    """RSS 2.0 feed of alerts (newest first)."""
    items = []
    for a in alerts:
        name = names.get(a.taxon_key, str(a.taxon_key))
        link = f"{settings.api_public_url}/species/{a.taxon_key}/alerts"
        items.append(
            "<item>"
            f"<title>{escape(name)}: {escape(a.kind.replace('_', ' '))} "
            f"({a.n_records})</title>"
            f"<link>{escape(link)}</link>"
            f'<guid isPermaLink="false">{a.alert_id}</guid>'
            f"<pubDate>{format_datetime(a.created_ts)}</pubDate>"
            f"<description>{escape(a.summary)}</description>"
            "</item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel>'
        "<title>RangeWatch early-detection alerts</title>"
        f"<link>{escape(settings.api_public_url)}</link>"
        "<description>New records outside the suitability envelope or far from every known "
        "record, classified by prior sampling effort.</description>"
        + "".join(items)
        + "</channel></rss>"
    )
