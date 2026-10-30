"""Factored Gram matrices: the foundation of the analysis.

All quantities are defined on the effective update dW = L R (L = (alpha/r) * B, R = A), never
on A and B directly: the factorization is not unique (L -> L Q, R -> Q^-1 R), and only
quantities defined on dW are invariant to it.

Key identity: the Frobenius inner product of two low-rank matrices needs no dense matrix,
    <L1 R1, L2 R2>_F = tr(R1^T L1^T L2 R2) = tr((L1^T L2)(R2 R1^T)) = sum_{k,l} (L1^T L2)[k,l] (R1 R2^T)[k,l]
which costs O(r^2 (d_out + d_in)) instead of O(d_out d_in).

Given the trajectory Gram matrix G[s,t] = <dW_s, dW_t>, the inner product between any linear
combinations of snapshots follows from G. For step updates delta_s = dW_{s+1} - dW_s:
    <delta_s, delta_t> = G[s+1,t+1] - G[s+1,t] - G[s,t+1] + G[s,t]
so VCS, FRR and DMD all need only G.
"""

from __future__ import annotations

import numpy as np
import torch


def factored_inner(L1: torch.Tensor, R1: torch.Tensor, L2: torch.Tensor, R2: torch.Tensor,
                   max_chunk_elems: int = 50_000_000) -> torch.Tensor:
    """Batched <L1[s] R1[s], L2[t] R2[t]>_F, returned as a (T1, T2) matrix.

    L1: (T1, d_out, k1)  R1: (T1, k1, d_in)
    L2: (T2, d_out, k2)  R2: (T2, k2, d_in)

    The (T, d_out, k) stacks are flattened to (d_out, T*k) so that one matrix product yields
    every L1^T L2 block, and likewise for R. The two block matrices are multiplied elementwise
    and summed within each (k1, k2) block. Rows of T1 are processed in chunks to bound memory.
    """
    T1, d_out, k1 = L1.shape
    T2, _, k2 = L2.shape
    L2f = L2.permute(1, 0, 2).reshape(d_out, T2 * k2)  # (d_out, T2*k2)
    R2f = R2.reshape(T2 * k2, -1)  # (T2*k2, d_in)
    rows = max(1, max_chunk_elems // max(1, k1 * T2 * k2))
    out = torch.empty((T1, T2), dtype=L1.dtype, device=L1.device)
    for s0 in range(0, T1, rows):
        s1 = min(T1, s0 + rows)
        n = s1 - s0
        L1f = L1[s0:s1].permute(1, 0, 2).reshape(d_out, n * k1)
        R1f = R1[s0:s1].reshape(n * k1, -1)
        PL = L1f.T @ L2f  # (n*k1, T2*k2): [(s,k),(t,l)] = (L1_s^T L2_t)[k,l]
        PR = R1f @ R2f.T  # (n*k1, T2*k2): [(s,k),(t,l)] = (R1_s R2_t^T)[k,l]
        out[s0:s1] = (PL * PR).reshape(n, k1, T2, k2).sum(dim=(1, 3))
    return out


def dense_from_factors(L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """(T, d_out, d_in) dense dW. Tests and toy models only."""
    return L @ R


def disp_inner(G: np.ndarray, I1: np.ndarray, J1: np.ndarray, I2: np.ndarray, J2: np.ndarray) -> np.ndarray:
    """Inner products between displacements, <W[J1] - W[I1], W[J2] - W[I2]>, computed from G.

    All index arrays share (or broadcast to) one shape. Example: delta_s is the displacement I = s, J = s + 1.
    """
    return G[J1, J2] - G[J1, I2] - G[I1, J2] + G[I1, I2]


def window_gram(G: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Pairwise inner products of a set of displacements (the p-th is W[J[..., p]] - W[I[..., p]]).

    I, J: (..., m) -> (..., m, m). Computes the small Gram matrices of many anchors at once.
    """
    return disp_inner(G, I[..., :, None], J[..., :, None], I[..., None, :], J[..., None, :])


def delta_gram(G: np.ndarray) -> np.ndarray:
    """Gram matrix of step updates delta_s = dW_{s+1} - dW_s, shape (T-1, T-1)."""
    return G[1:, 1:] - G[1:, :-1] - G[:-1, 1:] + G[:-1, :-1]
