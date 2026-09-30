"""Bulletin rendering test (HTML; PDF conversion needs WeasyPrint system libs → Docker)."""

from __future__ import annotations

import importlib.util

import pytest

from src.bulletin.generator import BulletinGenerator
from tests.conftest import TAXON_KEY
from tests.test_api import trained  # noqa: F401  (module-scoped trained-species fixture)


def test_bulletin_html_contains_all_sections(trained):  # noqa: F811
    settings, repo, artifacts = trained
    html = BulletinGenerator(repo, artifacts, settings).render_html(TAXON_KEY, 1)
    for heading in (
        "Executive summary",
        "Native range and suitability",
        "Global projection, candidate invasion zones and extrapolation",
        "Severity index",
        "Variable importance",
        "Model diagnostics",
        "Expansion outlook",
        "Methods &amp; reproducibility",
        "References",
    ):
        assert heading in html, heading
    assert "Transferability caveat" in html
    assert "MESS &lt; 0" in html  # uncertainty layer always explained
    assert "Severity index = " in html  # weights documented, not a black box
    assert "GBIF.org" in html  # citation
    assert html.count("data:image/png;base64,") == 5  # all figures embedded
    order = [
        html.index(f"<td>{c}</td>")
        for c in ("suitability", "climate analogy", "spread rate", "ecological impact prior")
    ]
    assert order == sorted(order)  # fixed component order regardless of jsonb key order


@pytest.mark.skipif(
    importlib.util.find_spec("weasyprint") is None,
    reason="WeasyPrint not installed (runs in the backend Docker image)",
)
def test_bulletin_pdf(trained):  # noqa: F811
    settings, repo, artifacts = trained
    out = BulletinGenerator(repo, artifacts, settings).generate(TAXON_KEY, 1)
    assert artifacts.local_path(out["pdf"]).stat().st_size > 10_000
