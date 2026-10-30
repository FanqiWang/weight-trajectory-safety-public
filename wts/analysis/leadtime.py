"""Onset detection and lead time.

- Behavioural onset: first evaluation step where the C-arm refusal rate falls below
  (baseline - k_se * binomial SE) and stays there for `persist` evaluations.
- Signal onset: first snapshot step where the C-arm metric leaves the band
  mean +/- k_sigma * sigma of the matched S arm (across test seeds, at the same step) and stays
  out for `persist` snapshots. Two-sided: no assumption about the direction of the signal.
- Lead time = behavioural onset - signal onset (positive = weights move first).
"""

from __future__ import annotations

import warnings

import numpy as np


def first_persistent(flags: np.ndarray, persist: int) -> int | None:
    """Start index of the first run of `persist` consecutive True values, or None."""
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= persist:
            return i - persist + 1
    return None


def behavior_onset(steps: np.ndarray, refusal: np.ndarray, n: int, k_se: float = 3.0,
                   persist: int = 3) -> float:
    """steps and refusal sorted by step; the first point is the baseline (step 0)."""
    if len(refusal) == 0:
        return float("nan")
    r0 = float(refusal[0])
    # A 100% baseline would give SE = 0; clamping to [0.5/n, 1 - 0.5/n] gives a floor
    rc = min(max(r0, 0.5 / n), 1 - 0.5 / n)
    se = np.sqrt(rc * (1 - rc) / n)
    flags = np.asarray(refusal) < r0 - k_se * se
    flags[0] = False
    i = first_persistent(flags, persist)
    return float(steps[i]) if i is not None else float("nan")


def signal_onset(steps: np.ndarray, c_series: np.ndarray, s_matrix: np.ndarray, k_sigma: float = 3.0,
                 persist: int = 3, skip_initial: int = 0, sigma_floor: float = 1e-6) -> float:
    """c_series: (T,); s_matrix: (n_seeds, T) on the same step grid.

    sigma is the across-seed sample standard deviation. With a single seed it is 0 and
    sigma_floor takes over (only the smoke test runs like this).
    """
    s_matrix = np.atleast_2d(s_matrix)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # metrics are all-NaN at early steps
        mu = np.nanmean(s_matrix, axis=0)
        sd = np.nanstd(s_matrix, axis=0, ddof=1) if s_matrix.shape[0] > 1 else np.zeros_like(mu)
    sd = np.where(np.isfinite(sd), np.maximum(sd, sigma_floor), sigma_floor)
    ok = np.isfinite(c_series) & np.isfinite(mu) & (steps >= skip_initial)
    flags = ok & (np.abs(c_series - mu) > k_sigma * sd)
    i = first_persistent(flags, persist)
    return float(steps[i]) if i is not None else float("nan")
