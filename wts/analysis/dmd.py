"""Snapshot-method dynamic mode decomposition (DMD) computed from Gram matrices only.

With X = [x_1, ..., x_m] and Y = [x_2, ..., x_{m+1}] (columns are step updates delta), we seek a
linear operator A with Y ~ A X. Exact DMD takes a truncated SVD X = U S V^T and the reduced
operator A~ = U^T Y V S^-1, whose eigenvalues are the non-zero eigenvalues of A. Since
U = X V S^-1,
    A~ = S^-1 V^T (X^T Y) V S^-1,   with X^T X = V S^2 V^T,
so only the small matrices K_XX = X^T X and K_XY = X^T Y are needed, and both follow from the
trajectory Gram matrix. DMD runs on million-dimensional vectors through m x m inner products.

|lambda| > 1: growing mode; |lambda| < 1: decaying; ~1: persistent drift. Abrupt changes in
max |lambda| can mark turning points in the dynamics.
"""

from __future__ import annotations

import numpy as np

from .gram import window_gram


def dmd_eigs_from_gram(Kxx: np.ndarray, Kxy: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    evals, V = np.linalg.eigh(Kxx)
    if evals.max() <= 0:
        return np.array([])
    keep = evals > rtol * evals.max()
    sig = np.sqrt(evals[keep])
    Vs = V[:, keep] / sig  # V S^-1
    Atil = Vs.T @ Kxy @ Vs
    return np.linalg.eigvals(Atil)


def dense_dmd_eigs(X: np.ndarray, Y: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Textbook exact DMD on dense data; reference implementation for tests."""
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    keep = s > rtol ** 0.5 * s.max()  # matches the eigh threshold: s^2 > rtol * s_max^2
    U, s, Vh = U[:, keep], s[keep], Vh[keep]
    Atil = U.T @ Y @ Vh.T / s
    return np.linalg.eigvals(Atil)


def sliding_dmd(G: np.ndarray, window: int, rtol: float = 1e-10) -> dict[str, np.ndarray]:
    """Sliding-window DMD. Anchor a uses delta_{a-w}, ..., delta_{a-1} (w updates):
        X = [delta_{a-w}, ..., delta_{a-2}],  Y = [delta_{a-w+1}, ..., delta_{a-1}]
    Records:
      dmd_maxabs[a]      spectral radius max |lambda| (computable at a)
      dmd_pred_cos[a+1]  cosine between the one-step prediction Y X^+ delta_{a-1} and the true delta_a
    """
    T = G.shape[0]
    maxabs = np.full(T, np.nan)
    pred = np.full(T, np.nan)
    anchors = np.arange(window, T - 1)
    if len(anchors) == 0:
        return {"dmd_maxabs": maxabs, "dmd_pred_cos": pred}
    # w + 1 updates per anchor: delta_{a-w}, ..., delta_{a-1}, delta_a
    p = np.arange(window + 1)
    I = anchors[:, None] - window + p[None, :]
    K = window_gram(G, I, I + 1)  # (A, w+1, w+1)
    m = window - 1
    for k, a in enumerate(anchors):
        Kxx = K[k, :m, :m]
        Kxy = K[k, :m, 1:m + 1]
        if np.trace(Kxx) <= 0:
            continue
        lam = dmd_eigs_from_gram(Kxx, Kxy, rtol)
        if lam.size:
            maxabs[a] = np.abs(lam).max()
        # One-step prediction: c = (X^T X)^+ X^T delta_{a-1}, prediction = Y c
        evals = np.linalg.eigvalsh(Kxx)
        c = np.linalg.pinv(Kxx, rcond=rtol, hermitian=True) @ K[k, :m, window - 1]
        Kyy = K[k, 1:m + 1, 1:m + 1]
        num = c @ K[k, 1:m + 1, window]  # <Y c, delta_a>
        den2 = (c @ Kyy @ c) * K[k, window, window]
        if den2 > 0 and evals.max() > 0:
            pred[a + 1] = num / np.sqrt(den2)
    return {"dmd_maxabs": maxabs, "dmd_pred_cos": pred}
