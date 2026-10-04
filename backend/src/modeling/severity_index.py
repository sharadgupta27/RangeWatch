"""Invasion severity index — a documented, configurable weighted composite (never a black box).

    Severity = w1·S_suitability + w2·S_climate_analogy + w3·S_spread_rate + w4·S_impact_prior

Component definitions (each in [0, 1]):

* S_suitability — mean predicted suitability over candidate invasion zones, scaled by how much
  of the non-native land area they cover: mean_suit × min(1, f / f_sat), with f the candidate
  fraction of non-native land and f_sat the saturation fraction (default 10 %).
* S_climate_analogy — fraction of candidate-zone area whose climate lies inside the species'
  *occupied* climatic envelope (every predictor between the 5th and 95th percentile of
  presence records).
* S_spread_rate — r / (r + k): r is the mean number of newly occupied 1°×1° cells per year
  outside the native range over the last N years; k is the half-saturation rate
  (default 5 cells/yr). Sensitive to sampling-effort growth — disclosed in the bulletin.
* S_impact_prior — user-supplied ecological impact prior (e.g. from GISD/EICAT); 0.5 = unknown.

Confidence is reported *separately* and never scales the score (CLAUDE.md constraint 2). It
is driven by the model design (Model A native-only = low) and by extrapolation over the
candidate zone, judged by the consensus of the five diagnostics (MESS, exDet, MOP, Shape, AOA):

* low      — Model A, or more than `confidence_low_extrapolated` (default 25 %) of the candidate
             area is flagged by at least 3 of the 5 diagnostics;
* moderate — more than `confidence_moderate_extrapolated` (default 10 %) is flagged by at least
             3 of 5, or less than `confidence_min_analog` (default 60 %) is flagged by none;
* high     — otherwise,

with two caps that hold confidence at moderate at best:

* MESS cap — more than `confidence_max_mess_extrapolated` (default 25 %) of the candidate area
             has MESS < 0. MESS and exDet NT1 are the same range test, so range-only
             extrapolation can fall short of the 3-of-5 consensus; MESS alone still caps it.
* CBI floor — the spatially cross-validated Continuous Boyce Index is below
             `confidence_min_cbi` (default 0.2): the model ranks presences little better than
             random, so its projection cannot be trusted whatever the extrapolation.

Models trained before the diagnostics existed fall back to MESS alone: the MESS-extrapolated
share takes the place of the ≥ 3-of-5 share, and the "flagged by none" test is skipped.
Every rule that lowers confidence is reported in `confidence.reasons`. Changing these thresholds
is a documented, reviewed decision (they are user-configurable).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

COMPONENTS: tuple[str, ...] = (
    "suitability",
    "climate_analogy",
    "spread_rate",
    "ecological_impact_prior",
)
DEFAULT_WEIGHTS: dict[str, float] = {
    "suitability": 0.35,
    "climate_analogy": 0.25,
    "spread_rate": 0.20,
    "ecological_impact_prior": 0.20,
}

COMPONENT_DESCRIPTIONS: dict[str, str] = {
    "suitability": (
        "Mean predicted suitability across candidate invasion zones, scaled by the share of "
        "non-native land they cover (saturating at f_sat)."
    ),
    "climate_analogy": (
        "Share of candidate-zone area whose climate falls inside the species' occupied "
        "climatic envelope (5th–95th percentile of presence records for every predictor)."
    ),
    "spread_rate": (
        "Mean number of newly occupied 1° grid cells per year outside the native range "
        "(last N years), mapped to "
        "0–1 as r/(r+k). Can be inflated by growing observer effort."
    ),
    "ecological_impact_prior": (
        "Expert/literature prior on ecological impact (e.g. GISD, EICAT); 0.5 means unknown."
    ),
}


@dataclass(frozen=True)
class SeverityConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    impact_prior: float = 0.5
    impact_prior_source: str = "default (unknown)"
    suitability_saturation: float = 0.10
    spread_half_saturation: float = 5.0
    spread_window_years: int = 10
    # Confidence thresholds (shares of the candidate zone) — see the module docstring.
    confidence_low_extrapolated: float = 0.25
    confidence_moderate_extrapolated: float = 0.10
    confidence_min_analog: float = 0.60
    confidence_max_mess_extrapolated: float = 0.25
    confidence_min_cbi: float = 0.20

    def __post_init__(self) -> None:
        missing = set(COMPONENTS) - set(self.weights)
        if missing:
            raise ValueError(f"Missing severity weights: {sorted(missing)}")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("Severity weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one severity weight must be positive")
        if not 0 <= self.impact_prior <= 1:
            raise ValueError("impact_prior must be within [0, 1]")
        if not 0 < self.suitability_saturation <= 1:
            raise ValueError("suitability_saturation must be within (0, 1]")
        if self.spread_half_saturation <= 0 or self.spread_window_years < 2:
            raise ValueError("Invalid spread-rate parameters")
        if not (
            0 <= self.confidence_moderate_extrapolated <= self.confidence_low_extrapolated <= 1
        ):
            raise ValueError(
                "Confidence thresholds must satisfy 0 ≤ moderate ≤ low ≤ 1 (extrapolated share)"
            )
        if not 0 <= self.confidence_min_analog <= 1:
            raise ValueError("confidence_min_analog must be within [0, 1]")
        if not 0 <= self.confidence_max_mess_extrapolated <= 1:
            raise ValueError("confidence_max_mess_extrapolated must be within [0, 1]")
        if not -1 <= self.confidence_min_cbi <= 1:
            raise ValueError("confidence_min_cbi must be within [-1, 1]")

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = sum(self.weights[c] for c in COMPONENTS)
        return {c: self.weights[c] / total for c in COMPONENTS}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SeverityConfig:
        if not raw:
            return cls()
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeverityInputs:
    """Raw quantities produced by projection + occurrence history (stored per model version)."""

    candidate_mean_suitability: float
    candidate_fraction_of_nonnative_land: float
    candidate_envelope_fraction: float
    candidate_mess_ok_fraction: float
    new_cells_per_year: float
    model_type: str
    # Diagnostic consensus over the candidate zone (None for models without the diagnostics)
    candidate_consensus_ok_fraction: float | None = None
    candidate_consensus_majority_fraction: float | None = None
    # Spatial-CV Continuous Boyce Index of the model (None if unavailable)
    cbi_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def spread_rate_cells_per_year(
    occ: pd.DataFrame, outside_native: np.ndarray, window_years: int, current_year: int
) -> float:
    """Mean newly occupied 1° cells per year (outside native range) over the last N years.

    `occ` needs columns longitude, latitude, year.
    """
    df = occ.loc[outside_native & occ["year"].notna(), ["longitude", "latitude", "year"]]
    if df.empty:
        return 0.0
    cells = (
        np.floor(df["longitude"]).astype(int).astype(str)
        + ":"
        + np.floor(df["latitude"]).astype(int).astype(str)
    )
    first_year = df.assign(cell=cells).groupby("cell")["year"].min()
    start = current_year - window_years + 1
    years = np.arange(start, current_year + 1)
    counts = np.array([(first_year == y).sum() for y in years], dtype="float64")
    return float(counts.mean())


_LEVELS = ("low", "moderate", "high")


def confidence_level(inputs: SeverityInputs, cfg: SeverityConfig) -> tuple[str, str, list[str]]:
    """Confidence level, its basis ("consensus" of the five diagnostics, or "mess") and every
    rule that lowered it (empty for high confidence)."""
    mess_extrapolated = 1.0 - inputs.candidate_mess_ok_fraction
    if inputs.candidate_consensus_majority_fraction is not None:
        basis = "consensus"
        extrapolated = inputs.candidate_consensus_majority_fraction
        analog = inputs.candidate_consensus_ok_fraction
        flagged = "flagged by ≥ 3 of 5 extrapolation diagnostics"
    else:
        basis, extrapolated, analog, flagged = "mess", mess_extrapolated, None, "with MESS < 0"
    level = "high"
    reasons: list[str] = []

    def lower(to: str, reason: str) -> None:
        nonlocal level
        if _LEVELS.index(to) < _LEVELS.index(level):
            level = to
        reasons.append(reason)

    if inputs.model_type == "A":
        lower("low", "Model A: trained on native-range records only")
    if extrapolated > cfg.confidence_low_extrapolated:
        lower(
            "low",
            f"{extrapolated:.1%} of the candidate zone {flagged} "
            f"(> {cfg.confidence_low_extrapolated:.0%})",
        )
    elif extrapolated > cfg.confidence_moderate_extrapolated:
        lower(
            "moderate",
            f"{extrapolated:.1%} of the candidate zone {flagged} "
            f"(> {cfg.confidence_moderate_extrapolated:.0%})",
        )
    if analog is not None and analog < cfg.confidence_min_analog:
        lower(
            "moderate",
            f"only {analog:.1%} of the candidate zone is flagged by no diagnostic "
            f"(< {cfg.confidence_min_analog:.0%})",
        )
    if basis == "consensus" and mess_extrapolated > cfg.confidence_max_mess_extrapolated:
        lower(
            "moderate",
            f"MESS cap: {mess_extrapolated:.1%} of the candidate zone has MESS < 0 "
            f"(> {cfg.confidence_max_mess_extrapolated:.0%})",
        )
    cbi = inputs.cbi_mean
    if cbi is not None and np.isfinite(cbi) and cbi < cfg.confidence_min_cbi:
        lower(
            "moderate",
            f"CBI floor: spatial-CV CBI {cbi:.2f} < {cfg.confidence_min_cbi:.2f} — the model "
            "ranks presences little better than random",
        )
    return level, basis, reasons


def _confidence(inputs: SeverityInputs, cfg: SeverityConfig) -> dict[str, Any]:
    level, basis, reasons = confidence_level(inputs, cfg)
    if basis == "consensus":
        note = (
            "Confidence reflects model design, extrapolation over the candidate zone as judged "
            "by five diagnostics (MESS, exDet, MOP, Shape, AOA; MESS alone can cap it at "
            "moderate) and model fit (CBI floor). It does not modify the severity score."
        )
    else:
        note = (
            "Confidence reflects model design, MESS extrapolation (this model predates the "
            "exDet/MOP/Shape/AOA diagnostics) and model fit (CBI floor). It does not modify "
            "the severity score."
        )
    return {
        "level": level,
        "basis": basis,
        "reasons": reasons,
        "cbi_mean": _round_or_none(inputs.cbi_mean),
        "mess_ok_fraction": round(inputs.candidate_mess_ok_fraction, 4),
        "consensus_ok_fraction": _round_or_none(inputs.candidate_consensus_ok_fraction),
        "consensus_majority_fraction": _round_or_none(inputs.candidate_consensus_majority_fraction),
        "model_type": inputs.model_type,
        "note": note,
    }


def _round_or_none(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else round(v, 4)


def compute_severity(inputs: SeverityInputs, cfg: SeverityConfig) -> dict[str, Any]:
    f = inputs.candidate_fraction_of_nonnative_land
    s_suit = float(
        np.clip(inputs.candidate_mean_suitability * min(1.0, f / cfg.suitability_saturation), 0, 1)
    )
    r = max(0.0, inputs.new_cells_per_year)
    components = {
        "suitability": s_suit,
        "climate_analogy": float(np.clip(inputs.candidate_envelope_fraction, 0, 1)),
        "spread_rate": float(r / (r + cfg.spread_half_saturation)),
        "ecological_impact_prior": float(cfg.impact_prior),
    }
    weights = cfg.normalized_weights
    contributions = {c: weights[c] * components[c] for c in COMPONENTS}
    score = float(sum(contributions.values()))
    return {
        "score": round(score, 4),
        "score_0_100": round(score * 100, 1),
        "category": _category(score),
        "components": {c: round(v, 4) for c, v in components.items()},
        "weights": {c: round(w, 4) for c, w in weights.items()},
        "contributions": {c: round(v, 4) for c, v in contributions.items()},
        "descriptions": COMPONENT_DESCRIPTIONS,
        "config": cfg.to_dict(),
        "inputs": inputs.to_dict(),
        "confidence": _confidence(inputs, cfg),
        "formula": "Severity = Σ w_i · S_i  (weights normalised to sum to 1)",
    }


def _category(score: float) -> str:
    if score >= 0.75:
        return "very high"
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "moderate"
    return "low"


def methods_footnote(result: dict[str, Any]) -> str:
    w = result["weights"]
    cfg = result["config"]
    return (
        "Severity index = "
        + " + ".join(f"{w[c]:.2f}×S_{c}" for c in COMPONENTS)
        + f". Suitability saturation f_sat = {cfg['suitability_saturation']:.2f}; "
        f"spread half-saturation k = {cfg['spread_half_saturation']:.1f} cells/yr over "
        f"{cfg['spread_window_years']} years; impact prior = {cfg['impact_prior']:.2f} "
        f"({cfg['impact_prior_source']}). "
        + " ".join(f"S_{c}: {COMPONENT_DESCRIPTIONS[c]}" for c in COMPONENTS)
        + _confidence_footnote(cfg)
    )


def _confidence_footnote(cfg: dict[str, Any]) -> str:
    low = cfg.get("confidence_low_extrapolated", SeverityConfig.confidence_low_extrapolated)
    mod = cfg.get(
        "confidence_moderate_extrapolated", SeverityConfig.confidence_moderate_extrapolated
    )
    analog = cfg.get("confidence_min_analog", SeverityConfig.confidence_min_analog)
    mess_cap = cfg.get(
        "confidence_max_mess_extrapolated", SeverityConfig.confidence_max_mess_extrapolated
    )
    min_cbi = cfg.get("confidence_min_cbi", SeverityConfig.confidence_min_cbi)
    return (
        " Confidence (reported separately, never applied to the score): low for Model A or "
        f"when more than {low:.0%} of the candidate zone is flagged as extrapolation by at "
        f"least 3 of the 5 diagnostics (MESS, exDet, MOP, Shape, AOA); moderate when more than "
        f"{mod:.0%} is, or less than {analog:.0%} is flagged by none; otherwise high. It is "
        f"capped at moderate when more than {mess_cap:.0%} of the candidate zone has MESS < 0, "
        f"or when the spatial-CV CBI is below {min_cbi:.2f}."
    )
