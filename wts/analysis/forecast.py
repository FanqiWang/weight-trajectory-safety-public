"""Forecasting: can the early part of a trajectory predict a later threshold crossing?

- Power law y ~ a * t^p, fitted by linear regression in log-log space. "The Geometry of
  Alignment Collapse" predicts alignment loss growing as t^4, so the fitted exponent is itself a test.
- Quartic law y ~ a * t^4 with p fixed (closed form a = sum(y t^4) / sum(t^8)).
"""

from __future__ import annotations

import numpy as np


def fit_power_law(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Returns (a, p, R^2) using points with t > 0 and y > 0; NaN if fewer than 3 points."""
    ok = (t > 0) & (y > 0) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    lt, ly = np.log(t[ok]), np.log(y[ok])
    p, loga = np.polyfit(lt, ly, 1)
    resid = ly - (p * lt + loga)
    r2 = 1 - resid.var() / ly.var() if ly.var() > 0 else float("nan")
    return float(np.exp(loga)), float(p), float(r2)


def fit_quartic(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit with the exponent fixed at 4; returns (a, R^2)."""
    ok = np.isfinite(y) & (t >= 0)
    t, y = t[ok].astype(float), y[ok]
    denom = np.sum(t ** 8)
    if denom == 0 or len(y) < 2:
        return float("nan"), float("nan")
    a = float(np.sum(y * t ** 4) / denom)
    resid = y - a * t ** 4
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return a, float(r2)


def first_crossing(steps: np.ndarray, y: np.ndarray, threshold: float) -> float:
    hit = np.where(np.isfinite(y) & (y >= threshold))[0]
    return float(steps[hit[0]]) if len(hit) else float("nan")


def forecast_crossing(steps: np.ndarray, y: np.ndarray, threshold: float, upto_step: float) -> dict:
    """Fit a power law on steps <= upto_step and predict the first step where y reaches threshold."""
    early = steps <= upto_step
    a, p, r2 = fit_power_law(steps[early].astype(float), y[early])
    if np.isfinite(a) and a > 0 and p > 0:
        predicted = float((threshold / a) ** (1.0 / p))
    else:
        predicted = float("nan")
    return {"fit_a": a, "fit_p": p, "fit_r2": r2, "predicted_crossing": predicted,
            "actual_crossing": first_crossing(steps, y, threshold)}
