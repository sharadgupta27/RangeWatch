"""Poisson point-process SDM with an explicit effort offset (B8) and a bias surface shared
across species (B9).

MaxEnt is equivalent to an inhomogeneous Poisson point process (Renner & Warton 2013; Renner
et al. 2015). Fitting the point process directly makes the sampling-bias assumption explicit:

    log λ(x) = α + f(x)·β + Σ_k c_k·z_k(x) + Σ_j g_j·b_j(x)

* `z_k` are **fixed-coefficient** bias terms — the offset. With ``ppm_bias="offset"`` this is
  log(target-group records + 1) with c = 1: the expected number of records is proportional to
  the observer effort (Renner et al. 2015). With ``"shared"`` the coefficients come from a
  joint fit over many species recorded by the same observers (`fit_shared_bias`, Fithian et
  al. 2015), so a rare species borrows the bias estimate of its target group.
* `b_j` are **free** bias covariates (``"estimated"``: the effort itself, as in B6; static
  layers such as travel time). They are estimated and held at their conditioning value for
  prediction by the trainer (`BiasConditionedModel`).

The integral is approximated by uniform quadrature points (the background) with equal weights
(Berman & Turner 1992); f(x) are linear and quadratic terms of standardised predictors with a
ridge penalty. Prediction drops the offset (constant effort) and reports
1 − exp(−λ/λ̄), λ̄ the mean intensity over the quadrature — a cloglog-scaled relative
intensity (0.63 at the region's average).

The offset assumption is *tested*: the same model is refitted with the effort coefficient
free, and the likelihood-ratio statistic against c = 1 is reported (`offset_test`).

elapid 1.0.4 fits MaxEnt without offsets, so this is numpy / scipy code (ADR 0002).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

_ETA_CLIP = 30.0


@dataclass
class PpmData:
    predictors: list[str]
    po_x: pd.DataFrame  # presences: predictors + bias columns
    quad_x: pd.DataFrame  # quadrature (uniform background): predictors + bias columns
    # Column → fixed coefficient on the raw column (the offset part).
    fixed: dict[str, float] = field(default_factory=dict)
    # Bias columns whose coefficients are estimated.
    free: list[str] = field(default_factory=list)


def _standardiser(frame: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    if not cols:
        return np.zeros(0), np.zeros(0)
    ref = frame[list(cols)].to_numpy(dtype="float64")
    mean, sd = ref.mean(axis=0), ref.std(axis=0)
    sd[sd == 0] = 1.0
    return mean, sd


def _quadratic(frame: pd.DataFrame, cols: Sequence[str], mean: np.ndarray, sd: np.ndarray):
    z = (frame[list(cols)].to_numpy(dtype="float64") - mean) / sd
    return np.column_stack([z, z**2])


def _linear(frame: pd.DataFrame, cols: Sequence[str], mean: np.ndarray, sd: np.ndarray):
    if not cols:
        return np.zeros((len(frame), 0))
    return (frame[list(cols)].to_numpy(dtype="float64") - mean) / sd


def _offset(frame: pd.DataFrame, fixed: dict[str, float]) -> np.ndarray:
    out = np.zeros(len(frame))
    for col, coef in fixed.items():
        out += coef * frame[col].to_numpy(dtype="float64")
    return out


class PointProcessModel:
    """Fitted point process. `predict(x)` takes the climate predictors plus any *free* bias
    columns (held constant by the caller); the offset is never applied for prediction."""

    def __init__(
        self,
        predictors: Sequence[str],
        mean: np.ndarray,
        sd: np.ndarray,
        beta: np.ndarray,
        alpha: float,
        free: Sequence[str],
        free_mean: np.ndarray,
        free_sd: np.ndarray,
        gamma: np.ndarray,
        log_norm: float,
    ) -> None:
        self.predictors = list(predictors)
        self.mean, self.sd, self.beta, self.alpha = mean, sd, beta, float(alpha)
        self.free = list(free)
        self.free_mean, self.free_sd, self.gamma = free_mean, free_sd, gamma
        self.log_norm = float(log_norm)

    def log_intensity(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=self.predictors)
        eta = self.alpha + _quadratic(frame, self.predictors, self.mean, self.sd) @ self.beta
        if self.free:
            missing = [c for c in self.free if c not in frame.columns]
            if missing:
                raise ValueError(f"free bias columns missing for prediction: {missing}")
            eta = eta + _linear(frame, self.free, self.free_mean, self.free_sd) @ self.gamma
        return eta

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        eta = np.clip(self.log_intensity(x) - self.log_norm, -_ETA_CLIP, _ETA_CLIP)
        return -np.expm1(-np.exp(eta))

    @property
    def feature_names(self) -> list[str]:
        return [*self.predictors, *(f"{p}²" for p in self.predictors)]


def fit_ppm(data: PpmData, l2: float = 1.0, max_iter: int = 500) -> dict[str, Any]:
    """Penalised maximum likelihood of the point process; returns model, report and the
    unpenalised negative log-likelihood (for the offset test)."""
    preds = data.predictors
    mean, sd = _standardiser(data.quad_x, preds)
    fmean, fsd = _standardiser(data.quad_x, data.free)
    f_po = _quadratic(data.po_x, preds, mean, sd)
    f_q = _quadratic(data.quad_x, preds, mean, sd)
    b_po = _linear(data.po_x, data.free, fmean, fsd)
    b_q = _linear(data.quad_x, data.free, fmean, fsd)
    o_po, o_q = _offset(data.po_x, data.fixed), _offset(data.quad_x, data.fixed)
    n_f, n_b = f_po.shape[1], b_po.shape[1]
    w_q = 1.0 / len(f_q)  # region area normalised to 1 (absorbed by α)

    def nll(theta: np.ndarray) -> tuple[float, np.ndarray, float]:
        alpha, beta, gamma = theta[0], theta[1 : 1 + n_f], theta[1 + n_f :]
        eta_po = np.clip(alpha + f_po @ beta + b_po @ gamma + o_po, -_ETA_CLIP, _ETA_CLIP)
        eta_q = np.clip(alpha + f_q @ beta + b_q @ gamma + o_q, -_ETA_CLIP, _ETA_CLIP)
        mu_q = w_q * np.exp(eta_q)
        raw = -(eta_po.sum() - mu_q.sum())
        pen = 0.5 * l2 * (beta @ beta + gamma @ gamma)
        g_alpha = -(len(eta_po) - mu_q.sum())
        g_beta = -(f_po.sum(axis=0) - f_q.T @ mu_q) + l2 * beta
        g_gamma = -(b_po.sum(axis=0) - b_q.T @ mu_q) + l2 * gamma
        return raw + pen, np.r_[g_alpha, g_beta, g_gamma], raw

    theta0 = np.r_[np.log(max(len(f_po), 1)) - float(np.mean(o_q)), np.zeros(n_f + n_b)]
    res = minimize(
        lambda t: nll(t)[:2], theta0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter}
    )
    alpha, beta, gamma = res.x[0], res.x[1 : 1 + n_f], res.x[1 + n_f :]
    eta0_q = alpha + f_q @ beta + b_q @ gamma  # offset dropped: intensity at constant effort
    log_norm = float(np.log(np.mean(np.exp(np.clip(eta0_q, -_ETA_CLIP, _ETA_CLIP)))))
    model = PointProcessModel(preds, mean, sd, beta, alpha, data.free, fmean, fsd, gamma, log_norm)
    report = {
        "converged": bool(res.success),
        "iterations": int(res.nit),
        "neg_log_likelihood": round(float(nll(res.x)[2]), 4),
        "n_presences": len(f_po),
        "n_quadrature": len(f_q),
        "coefficients": {
            n: round(float(c), 5) for n, c in zip(model.feature_names, beta, strict=True)
        },
        "free_bias_coefficients": {
            # per raw unit of the column (the fit is on the standardised scale)
            n: round(float(c / s), 5)
            for n, c, s in zip(data.free, gamma, fsd, strict=True)
        },
        "fixed_bias_coefficients": {k: round(float(v), 5) for k, v in data.fixed.items()},
    }
    return {"model": model, "report": report, "nll": float(nll(res.x)[2])}


def offset_test(data: PpmData, column: str, l2: float = 1.0) -> dict[str, Any] | None:
    """Likelihood-ratio test of the fixed coefficient of `column` against a free one: refit
    with the column moved from `fixed` to `free` (Renner et al. 2015: the offset assumption
    is testable). None when the column is not a fixed term."""
    if column not in data.fixed:
        return None
    assumed = data.fixed[column]
    fixed = {k: v for k, v in data.fixed.items() if k != column}
    restricted = fit_ppm(data, l2)
    free = fit_ppm(
        PpmData(data.predictors, data.po_x, data.quad_x, fixed, [*data.free, column]), l2
    )
    stat = max(0.0, 2.0 * (restricted["nll"] - free["nll"]))
    estimate = free["report"]["free_bias_coefficients"][column]
    p = float(chi2.sf(stat, df=1))
    return {
        "column": column,
        "assumed_coefficient": round(float(assumed), 5),
        "free_coefficient": estimate,
        "lr_statistic": round(stat, 3),
        "p_value": round(p, 5),
        "interpretation": (
            "the data are consistent with the assumed effort coefficient"
            if p >= 0.05
            else "the data reject the assumed effort coefficient (records scale with effort "
            f"as ≈ effort^{estimate:.2f}); consider ppm_bias='estimated'"
        ),
        "penalty_note": "statistic from the unpenalised log-likelihoods of two ridge fits",
    }


# ---------------------------------------------------------------------------
# B9: one bias surface for many species (Fithian et al. 2015)
# ---------------------------------------------------------------------------
@dataclass
class SpeciesBiasData:
    taxon_key: int
    predictors: list[str]
    po_x: pd.DataFrame  # predictors + bias columns
    quad_x: pd.DataFrame


def fit_shared_bias(
    species: Sequence[SpeciesBiasData],
    bias_columns: Sequence[str],
    l2: float = 1.0,
    max_iter: int = 1000,
) -> dict[str, Any]:
    """Joint point process over species: each species has its own intercept and climate
    response, all share the coefficients of `bias_columns` (per raw unit, reusable as fixed
    offsets for a species with few records)."""
    cols = list(bias_columns)
    if len(species) < 2 or not cols:
        raise ValueError("a shared bias needs ≥ 2 species and ≥ 1 bias column")
    pooled = pd.concat([s.quad_x[cols] for s in species], ignore_index=True)
    bmean, bsd = _standardiser(pooled, cols)
    blocks = []
    for s in species:
        mean, sd = _standardiser(s.quad_x, s.predictors)
        blocks.append(
            (
                _quadratic(s.po_x, s.predictors, mean, sd),
                _quadratic(s.quad_x, s.predictors, mean, sd),
                _linear(s.po_x, cols, bmean, bsd),
                _linear(s.quad_x, cols, bmean, bsd),
                1.0 / len(s.quad_x),
            )
        )
    sizes = [1 + b[0].shape[1] for b in blocks]
    n_g = len(cols)
    starts = np.cumsum([0, *sizes])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        gamma = theta[starts[-1] :]
        total, grad = 0.0, np.zeros_like(theta)
        g_gamma = np.zeros(n_g)
        for i, (f_po, f_q, b_po, b_q, w_q) in enumerate(blocks):
            seg = theta[starts[i] : starts[i + 1]]
            alpha, beta = seg[0], seg[1:]
            eta_po = np.clip(alpha + f_po @ beta + b_po @ gamma, -_ETA_CLIP, _ETA_CLIP)
            eta_q = np.clip(alpha + f_q @ beta + b_q @ gamma, -_ETA_CLIP, _ETA_CLIP)
            mu_q = w_q * np.exp(eta_q)
            total += -(eta_po.sum() - mu_q.sum()) + 0.5 * l2 * (beta @ beta)
            grad[starts[i]] = -(len(eta_po) - mu_q.sum())
            grad[starts[i] + 1 : starts[i + 1]] = -(f_po.sum(axis=0) - f_q.T @ mu_q) + l2 * beta
            g_gamma += -(b_po.sum(axis=0) - b_q.T @ mu_q)
        total += 0.5 * l2 * (gamma @ gamma)
        grad[starts[-1] :] = g_gamma + l2 * gamma
        return total, grad

    theta0 = np.zeros(starts[-1] + n_g)
    for i, (f_po, *_rest) in enumerate(blocks):
        theta0[starts[i]] = np.log(max(len(f_po), 1))
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    gamma = res.x[starts[-1] :]
    return {
        "coefficients": {
            c: round(float(g / s), 6) for c, g, s in zip(cols, gamma, bsd, strict=True)
        },
        "standardised_coefficients": {
            c: round(float(g), 6) for c, g in zip(cols, gamma, strict=True)
        },
        "converged": bool(res.success),
        "iterations": int(res.nit),
        "neg_log_likelihood": round(float(res.fun), 4),
        "l2_penalty": l2,
        "species": [
            {"taxon_key": s.taxon_key, "n_presences": len(s.po_x), "n_quadrature": len(s.quad_x)}
            for s in species
        ],
        "method": "joint Poisson point process: species-specific intercepts and climate "
        "responses (linear + quadratic), shared observer-bias coefficients",
        "reference": "Fithian et al. 2015",
    }
