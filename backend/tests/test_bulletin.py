"""Bulletin rendering test (HTML; PDF conversion needs WeasyPrint system libs → Docker)."""

from __future__ import annotations

import numpy as np
import pytest

from src.bulletin.generator import BulletinGenerator
from src.modeling.evaluation import independent_evaluation
from tests.conftest import TAXON_KEY
from tests.test_api import trained  # noqa: F401  (module-scoped trained-species fixture)


def test_bulletin_html_contains_all_sections(trained):  # noqa: F811
    settings, repo, artifacts = trained
    html = BulletinGenerator(repo, artifacts, settings).render_html(TAXON_KEY, 1)
    for heading in (
        "Executive summary",
        "Native range and suitability",
        "Global projection, candidate invasion zones and extrapolation",
        "Extrapolation diagnostics",
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
    for method in ("exDet", "MOP", "Shape", "AOA", "Mesgaran", "Meyer &amp; Pebesma"):
        assert method in html, method  # every diagnostic listed with its reference
    assert "NT1 &lt; 0 or NT2 &gt; 1" in html  # flag rules documented
    assert html.count("data:image/png;base64,") == 6  # all figures embedded (+ consensus map)
    order = [
        html.index(f"<td>{c}</td>")
        for c in ("suitability", "climate analogy", "spread rate", "ecological impact prior")
    ]
    assert order == sorted(order)  # fixed component order regardless of jsonb key order


def _weasyprint_usable() -> bool:
    try:
        import weasyprint  # noqa: F401  (also needs the Pango system libraries)
    except (ImportError, OSError):
        return False
    return True


@pytest.mark.skipif(
    not _weasyprint_usable(),
    reason="WeasyPrint or its system libraries missing (runs in the backend Docker image)",
)
def test_bulletin_pdf(trained):  # noqa: F811
    settings, repo, artifacts = trained
    out = BulletinGenerator(repo, artifacts, settings).generate(TAXON_KEY, 1)
    assert artifacts.local_path(out["pdf"]).stat().st_size > 10_000


def test_bulletin_reports_data_quality_effort_and_independent_evaluation(trained):  # noqa: F811
    settings, repo, artifacts = trained
    html = BulletinGenerator(repo, artifacts, settings).render_html(TAXON_KEY, 1)
    assert "<strong>Data quality.</strong>" in html
    assert "Not run: country capital" in html  # no gazetteers in tests: said, not hidden
    assert "<strong>Sampling effort.</strong>" in html
    assert "Quality flags" in html  # snapshot hash in the reproducibility table
    mv = repo.get_model_version(TAXON_KEY, 1)
    entry = {
        "dataset_id": "upload-b",
        "name": "Agency survey",
        "kind": "upload",
        "model_version": 1,
        "evaluated_ts": "2026-10-20T00:00:00",
        "n_records": 20,
        "n_excluded_overlap": 0,
        # 17 of 20 presences above the threshold, all in analog cells
        "metrics": independent_evaluation(
            np.r_[np.full(17, 0.9), np.full(3, 0.1)],
            np.array([]),
            np.linspace(0, 1, 200),
            threshold=0.5,
            presence_extrapolated=np.zeros(20, dtype=bool),
        ),
    }
    repo.update_model_artifacts(
        TAXON_KEY, 1, {**mv.artifacts, "independent_evaluation": {"upload-b": entry}}
    )
    try:
        html = BulletinGenerator(repo, artifacts, settings).render_html(TAXON_KEY, 1)
        assert "Independent evaluation" in html and "Agency survey" in html
        assert "85%" in html and "85% / –" in html
    finally:
        repo.update_model_artifacts(TAXON_KEY, 1, mv.artifacts)


def test_bulletin_methods_describe_weights_bias_covariate_and_training_uploads(
    trained,  # noqa: F811
):
    settings, repo, artifacts = trained
    mv = repo._models[TAXON_KEY][1]
    original = mv.metrics
    repro = {
        **original["reproducibility"],
        "presence_weights": {"method": "quality_tier_density", "min": 0.4, "max": 2.1},
        "bias_covariates": {
            "method": "target_group_effort",
            "status": "used",
            "transform": "log1p(target-group records per effort pixel)",
            "condition": "max",
            "conditioning_value": 7.25,
        },
        "training_uploads": [
            {"name": "INPA plots", "n_added": 36, "n_overlap_excluded": 4},
        ],
    }
    mv.metrics = {**original, "reproducibility": repro}
    try:
        html = BulletinGenerator(repo, artifacts, settings).render_html(TAXON_KEY, 1)
        assert "Presences were weighted (quality tier density" in html
        assert "held at its max (7.25)" in html
        assert "INPA plots (36 presences, 4 duplicates" in html
    finally:
        mv.metrics = original
