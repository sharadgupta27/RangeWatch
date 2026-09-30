"""Severity index tests: transparent weighting, validation, confidence independence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.severity_index import (
    COMPONENTS,
    SeverityConfig,
    SeverityInputs,
    compute_severity,
    methods_footnote,
    spread_rate_cells_per_year,
)


def _inputs(**kw) -> SeverityInputs:
    base = dict(
        candidate_mean_suitability=0.6,
        candidate_fraction_of_nonnative_land=0.05,
        candidate_envelope_fraction=0.4,
        candidate_mess_ok_fraction=0.9,
        new_cells_per_year=5.0,
        model_type="B",
    )
    return SeverityInputs(**{**base, **kw})


def test_score_is_weighted_sum_of_components():
    res = compute_severity(_inputs(), SeverityConfig())
    c, w = res["components"], res["weights"]
    assert c["suitability"] == pytest.approx(0.6 * 0.05 / 0.10)
    assert c["climate_analogy"] == pytest.approx(0.4)
    assert c["spread_rate"] == pytest.approx(0.5)  # r = k → 0.5
    assert c["ecological_impact_prior"] == pytest.approx(0.5)
    assert res["score"] == pytest.approx(sum(w[k] * c[k] for k in COMPONENTS), abs=1e-3)
    assert sum(w.values()) == pytest.approx(1.0)
    assert res["score_0_100"] == pytest.approx(res["score"] * 100, abs=0.1)


def test_weights_are_normalised_and_configurable():
    cfg = SeverityConfig(
        weights={
            "suitability": 2,
            "climate_analogy": 0,
            "spread_rate": 0,
            "ecological_impact_prior": 0,
        }
    )
    res = compute_severity(_inputs(), cfg)
    assert res["weights"]["suitability"] == 1.0
    assert res["score"] == pytest.approx(res["components"]["suitability"], abs=1e-4)


@pytest.mark.parametrize(
    "kw",
    [
        {
            "weights": {
                "suitability": -1,
                "climate_analogy": 1,
                "spread_rate": 1,
                "ecological_impact_prior": 1,
            }
        },
        {
            "weights": {
                "suitability": 0,
                "climate_analogy": 0,
                "spread_rate": 0,
                "ecological_impact_prior": 0,
            }
        },
        {"weights": {"suitability": 1}},
        {"impact_prior": 1.5},
        {"suitability_saturation": 0},
    ],
)
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        SeverityConfig(**kw)


def test_mess_changes_confidence_not_score():
    """CLAUDE.md constraint 2: low MESS reduces displayed confidence, not severity."""
    hi = compute_severity(_inputs(candidate_mess_ok_fraction=0.95), SeverityConfig())
    lo = compute_severity(_inputs(candidate_mess_ok_fraction=0.10), SeverityConfig())
    assert hi["score"] == lo["score"]
    assert hi["confidence"]["level"] == "high"
    assert lo["confidence"]["level"] == "low"


def test_model_a_is_low_confidence():
    res = compute_severity(
        _inputs(model_type="A", candidate_mess_ok_fraction=1.0), SeverityConfig()
    )
    assert res["confidence"]["level"] == "low"


def test_footnote_documents_weights():
    res = compute_severity(_inputs(), SeverityConfig())
    note = methods_footnote(res)
    for c in COMPONENTS:
        assert f"S_{c}" in note


def test_spread_rate_counts_new_cells_outside_native():
    occ = pd.DataFrame(
        {
            "longitude": [0.5, 1.5, 2.5, 2.7, 3.5, 50.5],
            "latitude": [0.5, 0.5, 0.5, 0.6, 0.5, 0.5],
            "year": [2020, 2021, 2022, 2023, 2023, 2023],
        }
    )
    outside = np.array([True, True, True, True, True, False])
    # new cells: 2020:1, 2021:1, 2022:1 (2.7 shares 2022's cell), 2023:1 → 4 over 5 years
    r = spread_rate_cells_per_year(occ, outside, window_years=5, current_year=2024)
    assert r == pytest.approx(4 / 5)
    assert spread_rate_cells_per_year(occ, np.zeros(6, bool), 5, 2024) == 0.0
