"""Integrated species distribution model (§4.4; Fithian et al. 2015, Isaac et al. 2020).

One shared environmental response β, two observation models fitted jointly by penalised
maximum likelihood (numpy / scipy only — no R worker, no new library):

  * presence-only records (GBIF, iNaturalist): inhomogeneous Poisson point process with
    log-intensity  a_po + f(x)·β + g·bias(x), the integral approximated by the background
    points as quadrature (Berman & Turner 1992; equivalent to MaxEnt, Renner & Warton 2013);
  * presence–absence survey records (training uploads with absences): Bernoulli with a
    complementary log-log link, P(present) = 1 − exp(−exp(a_pa + f(x)·β)).

f(x) are linear and quadratic terms of the standardised climate predictors. Because the
survey data anchor β, the coefficient g of the observer-bias covariate (target-group effort)
is *estimated*, not assumed, and prediction drops it: the published surface is the probability
of detecting the species in one survey unit of the presence–absence design, net of the
presence-only sampling bias. With no bias covariate the presence-only bias is an intercept.

Ridge penalty `l2` on β and g (standardised scale). Status: experimental, off by default; see
docs/adr/0002-integrated-sdm.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ETA_CLIP = 30.0


@dataclass
class IntegratedData:
    predictors: list[str]
    po_x: pd.DataFrame  # presence-only presences (predictors + bias columns)
    quad_x: pd.DataFrame  # background / quadrature points (predictors + bias columns)
    pa_x: pd.DataFrame  # presence–absence survey records (predictors)
    pa_y: np.ndarray  # 1 present, 0 absent
    bias_columns: list[str]


class IntegratedModel:
    """Fitted integrated model; `predict(x)` takes the climate predictors only."""

    def __init__(
        self,
        predictors: Sequence[str],
        mean: np.ndarray,
        sd: np.ndarray,
        beta: np.ndarray,
        a_pa: float,
        a_po: float,
        bias_columns: Sequence[str],
        bias_mean: np.ndarray,
        bias_sd: np.ndarray,
        gamma: np.ndarray,
    ) -> None:
        self.predictors = list(predictors)
        self.mean, self.sd, self.beta = mean, sd, beta
        self.a_pa, self.a_po = float(a_pa), float(a_po)
        self.bias_columns = list(bias_columns)
        self.bias_mean, self.bias_sd, self.gamma = bias_mean, bias_sd, gamma

    def features(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=self.predictors)
        z = (frame[self.predictors].to_numpy(dtype="float64") - self.mean) / self.sd
        return np.column_stack([z, z**2])

    def linear_predictor(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.features(x) @ self.beta

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        eta = np.clip(self.a_pa + self.linear_predictor(x), -_ETA_CLIP, _ETA_CLIP)
        return -np.expm1(-np.exp(eta))

    @property
    def feature_names(self) -> list[str]:
        return [*self.predictors, *(f"{p}²" for p in self.predictors)]


def _bias_matrix(frame: pd.DataFrame, cols: Sequence[str], mean: np.ndarray, sd: np.ndarray):
    if not cols:
        return np.zeros((len(frame), 0))
    return (frame[list(cols)].to_numpy(dtype="float64") - mean) / sd


def fit_integrated(data: IntegratedData, l2: float = 1.0, max_iter: int = 500) -> dict[str, Any]:
    """Joint penalised maximum likelihood; returns the model plus a fit report."""
    preds = data.predictors
    ref = data.quad_x[preds].to_numpy(dtype="float64")
    mean, sd = ref.mean(axis=0), ref.std(axis=0)
    sd[sd == 0] = 1.0
    bias_cols = list(data.bias_columns)
    if bias_cols:
        bref = data.quad_x[bias_cols].to_numpy(dtype="float64")
        bmean, bsd = bref.mean(axis=0), bref.std(axis=0)
        bsd[bsd == 0] = 1.0
    else:
        bmean = bsd = np.zeros(0)

    probe = IntegratedModel(
        preds, mean, sd, np.zeros(2 * len(preds)), 0, 0, bias_cols, bmean, bsd, np.zeros(0)
    )
    f_po, f_q, f_pa = (
        probe.features(data.po_x),
        probe.features(data.quad_x),
        probe.features(data.pa_x),
    )
    b_po = _bias_matrix(data.po_x, bias_cols, bmean, bsd)
    b_q = _bias_matrix(data.quad_x, bias_cols, bmean, bsd)
    y = np.asarray(data.pa_y, dtype="float64")
    n_f, n_b = f_po.shape[1], b_po.shape[1]
    w_q = 1.0 / len(f_q)  # region area normalised to 1: absorbed by the intercept a_po

    def unpack(theta: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        return theta[0], theta[1], theta[2 : 2 + n_f], theta[2 + n_f :]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a_po, a_pa, beta, gamma = unpack(theta)
        eta_po = np.clip(a_po + f_po @ beta + b_po @ gamma, -_ETA_CLIP, _ETA_CLIP)
        eta_q = np.clip(a_po + f_q @ beta + b_q @ gamma, -_ETA_CLIP, _ETA_CLIP)
        mu_q = w_q * np.exp(eta_q)
        ll_po = eta_po.sum() - mu_q.sum()
        eta_pa = np.clip(a_pa + f_pa @ beta, -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta_pa)
        log_p = np.log(-np.expm1(-mu) + 1e-300)
        ll_pa = float((y * log_p - (1 - y) * mu).sum())
        pen = 0.5 * l2 * (beta @ beta + gamma @ gamma)
        # gradients (of the negative penalised log-likelihood)
        d_po = np.ones(len(eta_po))
        d_q = -mu_q
        d_pa = np.where(y == 1, mu * np.exp(-mu) / (-np.expm1(-mu) + 1e-300), -mu)
        g_apo = -(d_po.sum() + d_q.sum())
        g_apa = -d_pa.sum()
        g_beta = -(f_po.T @ d_po + f_q.T @ d_q + f_pa.T @ d_pa) + l2 * beta
        g_gamma = -(b_po.T @ d_po + b_q.T @ d_q) + l2 * gamma
        return -(ll_po + ll_pa) + pen, np.r_[g_apo, g_apa, g_beta, g_gamma]

    prevalence = float(np.clip(y.mean() if len(y) else 0.5, 0.01, 0.99))
    theta0 = np.r_[np.log(max(len(f_po), 1)), np.log(-np.log(1 - prevalence)), np.zeros(n_f + n_b)]
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    a_po, a_pa, beta, gamma = unpack(res.x)
    model = IntegratedModel(preds, mean, sd, beta, a_pa, a_po, bias_cols, bmean, bsd, gamma)
    report = {
        "method": "integrated SDM: Poisson point process (presence-only) + cloglog Bernoulli "
        "(presence–absence) sharing one environmental response",
        "reference": "Fithian et al. 2015; Isaac et al. 2020; Renner et al. 2015",
        "features": "linear + quadratic terms of standardised predictors",
        "l2_penalty": l2,
        "converged": bool(res.success),
        "iterations": int(res.nit),
        "neg_log_likelihood": round(float(res.fun), 4),
        "n_presence_only": len(f_po),
        "n_quadrature": len(f_q),
        "n_pa_present": int((y == 1).sum()),
        "n_pa_absent": int((y == 0).sum()),
        "intercept_presence_absence": round(float(a_pa), 4),
        "intercept_presence_only": round(float(a_po), 4),
        "coefficients": {
            name: round(float(c), 5) for name, c in zip(model.feature_names, beta, strict=True)
        },
        "bias_coefficients": {
            name: round(float(c), 5) for name, c in zip(bias_cols, gamma, strict=True)
        },
        "prediction": "P(species detected in one presence–absence survey unit); the "
        "presence-only bias terms are dropped",
    }
    return {"model": model, "report": report}
