"""Stage 1 of the analysis: compress each run's snapshots into inner-product statistics,
cached at analysis/<setting>/cache/<run>.npz.

All metrics are computed from these small files; the snapshots (tens of GB) are read once.
Per run and module:
    G  (T, T)    trajectory Gram matrix
    X  (T, J)    inner products with J reference endpoints
    F  (T,)      energy in the free-direction subspace (if enabled)
    esa_agree / esa_valid (T,)   ESA counts
refs.npz holds the reference run names, their arms and each module's endpoint Gram matrix Rg (J, J).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from ..config import RunSpec, plan_runs, run_dir
from ..model import load_linear_weights, resolve_device
from ..snapshots import factored_entries, list_steps, load_module_factors, read_meta, sample_coords
from .gram import factored_inner
from .metrics import esa_counts

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def module_info(name: str) -> tuple[int, str]:
    """"model.layers.3.self_attn.q_proj" -> (3, "q_proj")"""
    m = LAYER_RE.search(name)
    return (int(m.group(1)) if m else -1), name.split(".")[-1]


def analysis_dir(cfg: dict, setting: str) -> Path:
    return Path(cfg["output_dir"]) / "analysis" / setting


def finished_runs(cfg: dict, setting: str) -> list[RunSpec]:
    return [s for s in plan_runs(cfg, setting) if (run_dir(cfg, s) / "DONE").exists()]


def analysis_steps(rdir: Path, stride: int) -> list[int]:
    steps = list_steps(rdir)
    sel = steps[::stride]
    if sel[-1] != steps[-1]:
        sel.append(steps[-1])
    return sel


def reference_specs(cfg: dict, specs: list[RunSpec]) -> list[RunSpec]:
    wanted = set(cfg["analysis"]["reference_arms"].values())
    return [s for s in specs if s.seed_group == "reference" and s.arm.name in wanted]


@torch.no_grad()
def free_subspaces(cfg: dict, modules: list[str], out_path: Path, log=print) -> dict | None:
    """Free direction: aligned model minus unaligned base; top-k left/right singular vectors per module."""
    base = cfg["analysis"]["free_direction_base"]
    if not base:
        return None
    if out_path.exists():
        return torch.load(out_path, weights_only=False)
    log(f"  computing free direction: {cfg['model']} - {base}")
    w_aligned = load_linear_weights(cfg, cfg["model"], modules)
    w_base = load_linear_weights(cfg, base, modules)
    k = cfg["analysis"]["free_direction_rank"]
    out = {}
    for m in modules:
        D = (w_aligned[m] - w_base[m]).double()
        k_eff = min(k, min(D.shape))
        # Randomized SVD: only the top singular vectors are needed
        U, S, V = torch.svd_lowrank(D, q=min(k_eff + 8, min(D.shape)), niter=6)
        out[m] = {"U": U[:, :k_eff].contiguous(), "V": V[:, :k_eff].contiguous(), "S": S[:k_eff]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return out


@torch.no_grad()
def extract_setting(cfg: dict, setting: str, log=print) -> dict:
    acfg = cfg["analysis"]
    adir = analysis_dir(cfg, setting)
    (adir / "cache").mkdir(parents=True, exist_ok=True)
    specs = finished_runs(cfg, setting)
    if not specs:
        raise RuntimeError(f"setting {setting} has no finished runs")
    meta = read_meta(run_dir(cfg, specs[0]))
    modules, scaling, shapes = meta["modules"], meta["scaling"], meta["shapes"]
    device = resolve_device(acfg["device"])

    refs = reference_specs(cfg, specs)
    if not refs:
        raise RuntimeError("no reference runs (C100/S100 on reference seeds); cannot build the harmful direction")
    ref_names = [s.name for s in refs]
    ref_arms = [s.arm.name for s in refs]

    # Reference endpoints: dW at the last step of every reference run
    endpoints = {}
    Rg = np.zeros((len(modules), len(refs), len(refs)))
    for mi, m in enumerate(modules):
        Ls, Rs = [], []
        for s in refs:
            rdir = run_dir(cfg, s)
            L, R = load_module_factors(rdir, m, [list_steps(rdir)[-1]], scaling)
            Ls.append(L[0])
            Rs.append(R[0])
        Lref, Rref = torch.stack(Ls).to(device), torch.stack(Rs).to(device)
        endpoints[m] = (Lref, Rref)
        Rg[mi] = factored_inner(Lref, Rref, Lref, Rref).cpu().numpy()
    np.savez(adir / "refs.npz", ref_names=np.array(ref_names), ref_arms=np.array(ref_arms),
             modules=np.array(modules), Rg=Rg)

    free = free_subspaces(cfg, modules, adir / "free_subspace.pt", log)
    coords = {m: sample_coords(shapes[m]["B"][0], shapes[m]["A"][1], acfg["esa_coords_per_module"], m)
              for m in modules}

    for s in specs:
        rdir = run_dir(cfg, s)
        out_path = adir / "cache" / f"{s.name}.npz"
        if out_path.exists() and out_path.stat().st_mtime > (rdir / "DONE").stat().st_mtime:
            with np.load(out_path) as cached:
                same_refs = list(cached["ref_names"]) == ref_names
            if same_refs:  # recompute X when the set of reference runs changed
                continue
        steps = analysis_steps(rdir, acfg["step_stride"])
        T, M = len(steps), len(modules)
        G = np.zeros((M, T, T))
        X = np.zeros((M, T, len(refs)))
        F = np.zeros((M, T)) if free else None
        agree = np.zeros((M, T))
        valid = np.zeros((M, T))
        for mi, m in enumerate(modules):
            L, R = load_module_factors(rdir, m, steps, scaling)
            L, R = L.to(device), R.to(device)
            G[mi] = factored_inner(L, R, L, R).cpu().numpy()
            X[mi] = factored_inner(L, R, *endpoints[m]).cpu().numpy()
            if free:
                U, V = free[m]["U"].to(device), free[m]["V"].to(device)
                # ||U^T dW V||^2: energy of dW in span{u_i v_j^T}; U^T L and R V are only k x r
                core = (U.T @ L) @ (R @ V)
                F[mi] = (core ** 2).sum(dim=(1, 2)).cpu().numpy()
            entries = factored_entries(L, R, *coords[m]).cpu().numpy()
            agree[mi], valid[mi] = esa_counts(entries, acfg["n_past"], acfg["tau"])
        arrays = dict(steps=np.array(steps), modules=np.array(modules), ref_names=np.array(ref_names), G=G, X=X,
                      esa_agree=agree, esa_valid=valid)
        if F is not None:
            arrays["F"] = F
        np.savez_compressed(out_path, **arrays)
        log(f"  extracted {s.name}: {T} steps x {M} modules")
    return {"modules": modules, "ref_names": ref_names, "ref_arms": ref_arms, "Rg": Rg}
