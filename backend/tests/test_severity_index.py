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
        {"confidence_low_extrapolated": 0.05, "confidence_moderate_extrapolated": 0.1},
        {"confidence_low_extrapolated": 1.2},
        {"confidence_min_analog": -0.1},
        {"confidence_max_mess_extrapolated": 1.5},
        {"confidence_min_cbi": -2},
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


def test_consensus_drives_confidence_with_configurable_thresholds():
    """Confidence uses the five-diagnostic consensus when the model has it — never the score."""

    def conf(ok: float, majority: float, cfg: SeverityConfig | None = None) -> dict:
        res = compute_severity(
            _inputs(
                candidate_mess_ok_fraction=0.99,  # MESS alone would say "high"
                candidate_consensus_ok_fraction=ok,
                candidate_consensus_majority_fraction=majority,
            ),
            cfg or SeverityConfig(),
        )
        return res

    high, mod, low = conf(0.9, 0.05), conf(0.7, 0.15), conf(0.5, 0.4)
    assert high["score"] == mod["score"] == low["score"]
    assert [r["confidence"]["level"] for r in (high, mod, low)] == ["high", "moderate", "low"]
    assert high["confidence"]["basis"] == "consensus"
    assert high["confidence"]["consensus_majority_fraction"] == 0.05
    # few cells robustly extrapolated, but too few fully analogous → moderate
    assert conf(0.5, 0.05)["confidence"]["level"] == "moderate"
    # thresholds are user-configurable
    lenient = SeverityConfig(
        confidence_low_extrapolated=0.5,
        confidence_moderate_extrapolated=0.45,
        confidence_min_analog=0.3,
    )
    assert conf(0.5, 0.4, lenient)["confidence"]["level"] == "high"
    assert "25%" in methods_footnote(high) and "3 of the 5" in methods_footnote(high)


def test_mess_cap_holds_confidence_at_moderate_when_consensus_is_lenient():
    """Range-only extrapolation (MESS & exDet NT1 agree, distance methods do not) can miss the
    3-of-5 consensus; MESS alone still caps confidence."""
    inputs = _inputs(
        candidate_mess_ok_fraction=0.70,
        candidate_consensus_ok_fraction=0.67,
        candidate_consensus_majority_fraction=0.098,
    )
    res = compute_severity(inputs, SeverityConfig())
    assert res["confidence"]["level"] == "moderate"
    assert res["confidence"]["reasons"] == [
        "MESS cap: 30.0% of the candidate zone has MESS < 0 (> 25%)"
    ]
    relaxed = SeverityConfig(confidence_max_mess_extrapolated=0.5)
    assert compute_severity(inputs, relaxed)["confidence"]["level"] == "high"


def test_cbi_floor_holds_confidence_at_moderate_without_touching_the_score():
    good = compute_severity(
        _inputs(candidate_mess_ok_fraction=0.99, cbi_mean=0.9), SeverityConfig()
    )
    poor = compute_severity(
        _inputs(candidate_mess_ok_fraction=0.99, cbi_mean=-0.02), SeverityConfig()
    )
    assert good["score"] == poor["score"]
    assert good["confidence"]["level"] == "high" and good["confidence"]["reasons"] == []
    assert poor["confidence"]["level"] == "moderate"
    assert poor["confidence"]["reasons"][0].startswith("CBI floor: spatial-CV CBI -0.02 < 0.20")
    assert poor["confidence"]["cbi_mean"] == -0.02
    # a cap never raises confidence, and low stays low
    low = compute_severity(
        _inputs(candidate_mess_ok_fraction=0.1, cbi_mean=-0.02), SeverityConfig()
    )
    assert low["confidence"]["level"] == "low" and len(low["confidence"]["reasons"]) == 2
    # unknown CBI does not cap
    nan = compute_severity(
        _inputs(candidate_mess_ok_fraction=0.99, cbi_mean=float("nan")), SeverityConfig()
    )
    assert nan["confidence"]["level"] == "high" and nan["confidence"]["cbi_mean"] is None
    assert "CBI is below 0.20" in methods_footnote(poor)


def test_models_without_diagnostics_fall_back_to_mess_with_the_same_thresholds():
    res = compute_severity(_inputs(candidate_mess_ok_fraction=0.85), SeverityConfig())
    assert res["confidence"]["basis"] == "mess"
    assert res["confidence"]["level"] == "moderate"  # 15 % MESS-extrapolated > 10 %
    assert res["confidence"]["consensus_ok_fraction"] is None


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
