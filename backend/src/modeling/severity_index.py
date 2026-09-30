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

Confidence is reported *separately* and never scales the score (CLAUDE.md constraint 2): it
is driven by MESS (share of candidate area without extrapolation) and by the model design
(Model A native-only = lower confidence).
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


def confidence_level(model_type: str, mess_ok_fraction: float) -> str:
    if model_type == "A" or mess_ok_fraction < 0.5:
        return "low"
    if mess_ok_fraction < 0.8:
        return "moderate"
    return "high"


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
        "confidence": {
            "level": confidence_level(inputs.model_type, inputs.candidate_mess_ok_fraction),
            "mess_ok_fraction": round(inputs.candidate_mess_ok_fraction, 4),
            "model_type": inputs.model_type,
            "note": (
                "Confidence reflects extrapolation (MESS) and model design; it does not "
                "modify the severity score."
            ),
        },
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
    )
