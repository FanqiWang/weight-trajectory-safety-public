"""Trajectory metrics. Everything is computed from inner products, so it is invariant to how dW is factored.

Inputs, for one group (a single module, or several modules merged):
    G   (T, T)   trajectory Gram matrix, G[s,t] = <dW_s, dW_t>
    X   (T, J)   inner products with J reference endpoints, X[t,j] = <dW_t, E_j>
    Rg  (J, J)   Gram matrix of the reference endpoints
    F   (T,)     energy inside the free-direction subspace (optional)
    esa_agree / esa_valid (T,)   ESA counts per anchor

Merging modules means summing their G, X, Rg, F and counts, which equals concatenating their
dW into one vector, so aggregation by module type or layer is exact.

Causal timestamps: each value is stored at the first step at which it can be computed. For
example VCS(a) needs dW_{a+tau}, so it is stored at a + tau. Lead times therefore never use
information from the future.
"""

from __future__ import annotations

import numpy as np

from .dmd import sliding_dmd
from .gram import window_gram

EPS = 1e-30


def nnls_two(b: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Two-variable non-negative least squares: for each row b_t, minimize g^T M g - 2 b_t^T g over g >= 0.

    Equivalent to min ||dW_t - g_C mu_C - g_N mu_N||^2 up to the constant ||dW_t||^2, with
    b_t = [<dW_t, mu_C>, <dW_t, mu_N>] and M the Gram matrix of mu_C, mu_N. The optimum is one
    of four candidates (interior, either edge, origin); all are checked and the best feasible kept.
    b: (T, 2), M: (2, 2). Returns g: (T, 2).
    """
    b = np.asarray(b, dtype=float)
    T = b.shape[0]
    cands, vals = [], []
    # origin
    cands.append(np.zeros((T, 2)))
    vals.append(np.zeros(T))
    # one prototype only (1-D problem: g = max(0, b / M_kk))
    for k in range(2):
        g = np.zeros((T, 2))
        if M[k, k] > EPS:
            g[:, k] = np.maximum(0.0, b[:, k] / M[k, k])
        cands.append(g)
        vals.append(-(g[:, k] ** 2) * M[k, k])  # f = M_kk g^2 - 2 b g equals -b^2 / M_kk at g = b / M_kk
    # interior solution M g = b
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) > EPS * max(1.0, abs(M[0, 0] * M[1, 1])):
        g = np.linalg.solve(M, b.T).T
        ok = (g >= 0).all(axis=1)
        f = np.einsum("ti,ij,tj->t", g, M, g) - 2 * np.einsum("ti,ti->t", b, g)
        cands.append(np.where(ok[:, None], g, 0.0))
        vals.append(np.where(ok, f, np.inf))
    vals = np.stack(vals)  # (n_cand, T)
    best = np.argmin(vals, axis=0)
    return np.stack(cands)[best, np.arange(T)]


def prototype_weights(ref_names: list[str], ref_arms: list[str], reference_arms: dict) -> dict[str, np.ndarray]:
    """Prototypes mu_C, mu_S, mu_B are means of reference endpoints, represented as weight vectors."""
    J = len(ref_names)
    w = {}
    for fam in ("C", "S", "B"):
        arm = reference_arms.get(fam)
        idx = [j for j in range(J) if ref_arms[j] == arm]
        v = np.zeros(J)
        if idx:
            v[idx] = 1.0 / len(idx)
        w[fam] = v
    if not w["C"].any() or not w["S"].any():
        raise ValueError(f"missing reference runs: need reference-seed runs of "
                         f"{reference_arms['C']} and {reference_arms['S']}")
    # mu_N = (mu_S + mu_B) / 2; falls back to mu_S without B reference runs
    w["N"] = (w["S"] + w["B"]) / 2 if w["B"].any() else w["S"].copy()
    w["d"] = w["C"] - w["S"]  # harmful direction d = mu_C - mu_S
    return w


def compute_metrics(G: np.ndarray, X: np.ndarray, Rg: np.ndarray, weights: dict[str, np.ndarray],
                    acfg: dict, F: np.ndarray | None = None, esa_agree: np.ndarray | None = None,
                    esa_valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """{metric name: length-T array}, aligned by causal timestamp, NaN where not computable."""
    T = G.shape[0]
    n, tau, w_dmd = acfg["n_past"], acfg["tau"], acfg["dmd_window"]
    out: dict[str, np.ndarray] = {}
    sq = np.clip(np.diag(G), 0, None)
    norm = np.sqrt(sq)
    out["norm"] = norm

    step_norm = np.full(T, np.nan)
    idx = np.arange(1, T)
    step_norm[1:] = np.sqrt(np.clip(G[idx, idx] - 2 * G[idx, idx - 1] + G[idx - 1, idx - 1], 0, None))
    out["step_norm"] = step_norm

    # Harmful direction: cosine (main metric) and raw projection (control for the distance confound)
    wd = weights["d"]
    d_norm = np.sqrt(max(wd @ Rg @ wd, 0.0))
    proj = X @ wd
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dir_cos"] = np.where(norm > 0, proj / (norm * d_norm + EPS), np.nan)
    out["dir_proj"] = proj / (d_norm + EPS)

    # TRACE-style harmfulness score s_H
    wC, wN = weights["C"], weights["N"]
    b = np.stack([X @ wC, X @ wN], axis=1)
    M = np.array([[wC @ Rg @ wC, wC @ Rg @ wN], [wN @ Rg @ wC, wN @ Rg @ wN]])
    gam = nnls_two(b, M)
    s_h = gam[:, 0] / (gam[:, 0] + gam[:, 1] + 1e-12)
    out["s_H"] = np.where(norm > 0, s_h, np.nan)

    if F is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["free_frac"] = np.where(sq > 0, F / (sq + EPS), np.nan)

    out.update(predictability_metrics(G, n, tau))
    if esa_agree is not None:
        esa = np.full(T, np.nan)
        for a in range(T):
            if a + tau < T and esa_valid[a] > 0:
                esa[a + tau] = esa_agree[a] / esa_valid[a]
        out["esa"] = esa
    out.update(sliding_dmd(G, w_dmd))
    return out


def predictability_metrics(G: np.ndarray, n: int, tau: int, rcond: float = 1e-10) -> dict[str, np.ndarray]:
    """VCS and FRR structured-predictability probes, computed from G.

    At anchor a, past updates H = [delta_{a-n}, ..., delta_{a-1}] and future displacement y = dW_{a+tau} - dW_a:
      VCS(a) = cos(delta_{a-1}, y)                          does the latest step point where the trajectory goes?
      FRR(a) = sqrt(1 - y^T H (H^T H)^+ H^T y / y^T y)      fraction of y not explained by the span of past updates
    High VCS / low FRR = predictable. Values are stored at a + tau.
    """
    T = G.shape[0]
    vcs = np.full(T, np.nan)
    frr = np.full(T, np.nan)
    anchors = np.arange(n, T - tau)
    if len(anchors) == 0:
        return {"vcs": vcs, "frr": frr}
    # n + 1 displacements per anchor: delta_{a-n..a-1} (the p-th is W[a-n+p+1] - W[a-n+p]) and y
    p = np.arange(n)
    I = np.concatenate([anchors[:, None] - n + p[None, :], anchors[:, None]], axis=1)
    J = np.concatenate([anchors[:, None] - n + p[None, :] + 1, anchors[:, None] + tau], axis=1)
    K = window_gram(G, I, J)  # (A, n+1, n+1)
    for k, a in enumerate(anchors):
        HH, Hy, yy = K[k, :n, :n], K[k, :n, n], K[k, n, n]
        dd = K[k, n - 1, n - 1]
        if yy > 0 and dd > 0:
            vcs[a + tau] = K[k, n - 1, n] / np.sqrt(dd * yy)
            explained = Hy @ np.linalg.pinv(HH, rcond=rcond, hermitian=True) @ Hy
            frr[a + tau] = np.sqrt(np.clip(1.0 - explained / yy, 0.0, 1.0))
    return {"vcs": vcs, "frr": frr}


def esa_counts(entries: np.ndarray, n_past: int, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise sign agreement (ESA) counts per anchor a, returned as (agree, valid) of length T.

    entries: (T, n_coords), dW_t at the sampled coordinates. For each coordinate, compare
    sign(delta_{a-1}) with sign(dW_{a+tau} - dW_a); coordinates where either sign is 0 are skipped.
    Counts of several modules are summed when merging.
    """
    T = entries.shape[0]
    agree = np.zeros(T)
    valid = np.zeros(T)
    for a in range(max(1, n_past), T - tau):
        s1 = np.sign(entries[a] - entries[a - 1])
        s2 = np.sign(entries[a + tau] - entries[a])
        ok = (s1 != 0) & (s2 != 0)
        agree[a] = np.sum((s1 == s2) & ok)
        valid[a] = np.sum(ok)
    return agree, valid
