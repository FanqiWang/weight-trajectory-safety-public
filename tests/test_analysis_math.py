"""Every Gram-derived quantity matches a brute-force dense computation and is invariant to the factorization."""

import numpy as np
import torch
from conftest import random_trajectory

from wts.analysis.dmd import dense_dmd_eigs, dmd_eigs_from_gram, sliding_dmd
from wts.analysis.gram import delta_gram, factored_inner
from wts.analysis.metrics import compute_metrics, esa_counts, nnls_two, predictability_metrics
from wts.snapshots import factored_entries

ACFG = {"n_past": 3, "tau": 2, "dmd_window": 5}


def dense(L, R):
    return (L @ R).reshape(L.shape[0], -1).numpy()  # (T, d_out * d_in)


def refs(seed=10, J=4):
    """J reference endpoints; the first two act as C, the last two as S."""
    Ls, Rs = [], []
    for j in range(J):
        L, R = random_trajectory(T=2, seed=seed + j)
        Ls.append(L[-1])
        Rs.append(R[-1])
    return torch.stack(Ls), torch.stack(Rs)


WEIGHTS = {
    "C": np.array([0.5, 0.5, 0, 0]),
    "S": np.array([0, 0, 0.5, 0.5]),
    "B": np.zeros(4),
}
WEIGHTS["N"] = WEIGHTS["S"]
WEIGHTS["d"] = WEIGHTS["C"] - WEIGHTS["S"]


# ---------------------------------------------------------------- Gram correctness


def test_factored_gram_matches_dense(traj):
    L, R = traj
    W = dense(L, R)
    G = factored_inner(L, R, L, R).numpy()
    np.testing.assert_allclose(G, W @ W.T, rtol=1e-6, atol=1e-10)


def test_factored_gram_cross_and_chunking(traj):
    L, R = traj
    Lr, Rr = refs()
    X_full = factored_inner(L, R, Lr, Rr).numpy()
    X_chunk = factored_inner(L, R, Lr, Rr, max_chunk_elems=7).numpy()  # force chunking
    np.testing.assert_allclose(X_full, dense(L, R) @ dense(Lr, Rr).T, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(X_chunk, X_full, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------- factorization invariance


def test_metrics_invariant_to_factorization(traj):
    """Apply a different random invertible Q at every step (B -> B Q, A -> Q^-1 A): dW and every metric must not change."""
    L, R = traj
    Lr, Rr = refs()
    g = torch.Generator().manual_seed(1)
    r = L.shape[2]
    Q = torch.randn(L.shape[0], r, r, generator=g, dtype=torch.float64) + 3 * torch.eye(r, dtype=torch.float64)
    L2, R2 = L @ Q, torch.linalg.solve(Q, R)
    Qr = torch.randn(Lr.shape[0], r, r, generator=g, dtype=torch.float64) + 3 * torch.eye(r, dtype=torch.float64)
    Lr2, Rr2 = Lr @ Qr, torch.linalg.solve(Qr, Rr)

    def all_metrics(L, R, Lr, Rr):
        G = factored_inner(L, R, L, R).numpy()
        X = factored_inner(L, R, Lr, Rr).numpy()
        Rg = factored_inner(Lr, Rr, Lr, Rr).numpy()
        rows, cols = np.array([0, 3, 6, 2]), np.array([1, 4, 0, 2])
        agree, valid = esa_counts(factored_entries(L, R, rows, cols).numpy(), ACFG["n_past"], ACFG["tau"])
        return compute_metrics(G, X, Rg, WEIGHTS, ACFG, esa_agree=agree, esa_valid=valid)

    m1, m2 = all_metrics(L, R, Lr, Rr), all_metrics(L2, R2, Lr2, Rr2)
    assert set(m1) == set(m2)
    for k in m1:
        np.testing.assert_allclose(m1[k], m2[k], rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=k)


# ---------------------------------------------------------------- step-update inner products, VCS, FRR


def test_delta_gram_matches_dense(traj):
    L, R = traj
    W = dense(L, R)
    D = np.diff(W, axis=0)
    np.testing.assert_allclose(delta_gram(factored_inner(L, R, L, R).numpy()), D @ D.T, rtol=1e-6, atol=1e-10)


def test_vcs_frr_match_dense():
    L, R = random_trajectory(T=20, seed=3)
    W = dense(L, R)
    G = W @ W.T
    n, tau = 4, 3
    got = predictability_metrics(G, n, tau)
    for a in range(n, W.shape[0] - tau):
        H = np.stack([W[k + 1] - W[k] for k in range(a - n, a)], axis=1)
        y = W[a + tau] - W[a]
        d = W[a] - W[a - 1]
        vcs = d @ y / np.linalg.norm(d) / np.linalg.norm(y)
        coef, *_ = np.linalg.lstsq(H, y, rcond=None)
        frr = np.linalg.norm(y - H @ coef) / np.linalg.norm(y)
        assert np.isclose(got["vcs"][a + tau], vcs, rtol=1e-6), a
        assert np.isclose(got["frr"][a + tau], frr, rtol=1e-5, atol=1e-7), a


# ---------------------------------------------------------------- DMD


def _sorted(lam):
    lam = np.asarray(lam)
    return lam[np.lexsort((np.round(lam.imag, 8), np.round(lam.real, 8)))]


def test_dmd_from_gram_matches_dense(rng):
    X = rng.normal(size=(50, 6))
    Y = X @ rng.normal(size=(6, 6)) * 0.5 + 0.01 * rng.normal(size=(50, 6))
    got = dmd_eigs_from_gram(X.T @ X, X.T @ Y)
    want = dense_dmd_eigs(X, Y)
    np.testing.assert_allclose(_sorted(got), _sorted(want), rtol=1e-6, atol=1e-8)


def test_sliding_dmd_matches_dense():
    L, R = random_trajectory(T=16, seed=5)
    W = dense(L, R)
    G = W @ W.T
    w = 5
    out = sliding_dmd(G, w)
    for a in range(w, W.shape[0] - 1):
        deltas = [W[k + 1] - W[k] for k in range(a - w, a + 1)]  # delta_{a-w} ... delta_a
        X = np.stack(deltas[:w - 1], axis=1)
        Y = np.stack(deltas[1:w], axis=1)
        lam = dense_dmd_eigs(X, Y)
        assert np.isclose(out["dmd_maxabs"][a], np.abs(lam).max(), rtol=1e-6), a
        pred = Y @ np.linalg.lstsq(X, deltas[w - 1], rcond=None)[0]
        cos = pred @ deltas[w] / np.linalg.norm(pred) / np.linalg.norm(deltas[w])
        assert np.isclose(out["dmd_pred_cos"][a + 1], cos, rtol=1e-5, atol=1e-7), a


# ---------------------------------------------------------------- two-variable NNLS behind s_H


def test_nnls_two_matches_grid(rng):
    for trial in range(30):
        V = rng.normal(size=(2, 8))
        if trial % 3 == 0:
            V[1] = V[0] * 0.7 + 0.01 * rng.normal(size=8)  # nearly collinear prototypes exercise the boundary cases
        w = rng.normal(size=(5, 8))
        M = V @ V.T
        b = w @ V.T
        gam = nnls_two(b, M)
        assert (gam >= 0).all()
        grid = np.linspace(0, 6, 601)
        g0, g1 = np.meshgrid(grid, grid, indexing="ij")
        for t in range(5):
            f_grid = (M[0, 0] * g0**2 + 2 * M[0, 1] * g0 * g1 + M[1, 1] * g1**2 - 2 * b[t, 0] * g0 - 2 * b[t, 1] * g1)
            f_ours = gam[t] @ M @ gam[t] - 2 * b[t] @ gam[t]
            assert f_ours <= f_grid.min() + 1e-6, (trial, t)
            assert f_ours >= f_grid.min() - 0.05 * (abs(f_grid.min()) + 1), (trial, t)


# ---------------------------------------------------------------- ESA sampled entries


def test_factored_entries_match_dense(traj, rng):
    L, R = traj
    full = (L @ R).numpy()
    rows = rng.integers(0, L.shape[1], size=20)
    cols = rng.integers(0, R.shape[2], size=20)
    got = factored_entries(L, R, rows, cols).numpy()
    np.testing.assert_allclose(got, full[:, rows, cols], rtol=1e-12, atol=1e-12)


def test_esa_counts_match_definition(rng):
    E = rng.normal(size=(15, 30))
    E[:2] = 0.0
    agree, valid = esa_counts(E, n_past=3, tau=2)
    for a in range(3, 13):
        s1, s2 = np.sign(E[a] - E[a - 1]), np.sign(E[a + 2] - E[a])
        ok = (s1 != 0) & (s2 != 0)
        assert agree[a] == np.sum((s1 == s2) & ok) and valid[a] == ok.sum()
