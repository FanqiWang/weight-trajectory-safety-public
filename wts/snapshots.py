"""Reading and writing per-step snapshots; extracting dW factors and sampled entries.

On disk:
    runs/<run>/snapshots/step_000123.safetensors   keys "<module>.A" and "<module>.B", fp32
    runs/<run>/meta.json                           scaling, modules and shapes, config, seed, git commit

The analysis always uses the factored form dW = L R with L = (alpha/r) * B (d_out x r) and
R = A (r x d_in). Folding the scaling into L means no later code needs to know about it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

STEP_RE = re.compile(r"step_(\d+)\.safetensors$")


def snapshot_path(run_dir: str | Path, step: int) -> Path:
    return Path(run_dir) / "snapshots" / f"step_{step:06d}.safetensors"


def save_snapshot(run_dir: str | Path, step: int, state: dict[str, torch.Tensor]) -> Path:
    path = snapshot_path(run_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tensors = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in state.items()}
    save_file(tensors, str(tmp), metadata={"step": str(step)})
    os.replace(tmp, path)  # atomic: a killed process never leaves a half-written snapshot
    return path


def list_steps(run_dir: str | Path) -> list[int]:
    d = Path(run_dir) / "snapshots"
    if not d.exists():
        return []
    return sorted(int(m.group(1)) for f in d.iterdir() if (m := STEP_RE.search(f.name)))


def load_snapshot(run_dir: str | Path, step: int) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(str(snapshot_path(run_dir, step)), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def write_meta(run_dir: str | Path, meta: dict) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def read_meta(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "meta.json").read_text())


def load_module_factors(run_dir: str | Path, module: str, steps: list[int], scaling: float,
                        dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Factors of one module over several steps: L (T, d_out, r), R (T, r, d_in), dW_t = L_t @ R_t."""
    Ls, Rs = [], []
    for s in steps:
        with safe_open(str(snapshot_path(run_dir, s)), framework="pt") as f:
            Ls.append(f.get_tensor(f"{module}.B").to(dtype) * scaling)
            Rs.append(f.get_tensor(f"{module}.A").to(dtype))
    return torch.stack(Ls), torch.stack(Rs)


def sample_coords(d_out: int, d_in: int, n: int, module: str) -> tuple[np.ndarray, np.ndarray]:
    """A fixed sample of n (i, j) coordinates for one module (used by ESA).
    The seed is derived from the module name, so every run uses the same coordinates."""
    from .data import stable_hash

    rng = np.random.default_rng(stable_hash("coords:" + module) % (2**32))
    n = min(n, d_out * d_in)
    flat = rng.choice(d_out * d_in, size=n, replace=False)
    return flat // d_in, flat % d_in


def factored_entries(L: torch.Tensor, R: torch.Tensor, rows: np.ndarray, cols: np.ndarray) -> torch.Tensor:
    """dW_t[i, j] = sum_k L_t[i, k] R_t[k, j] at the sampled coordinates only. Returns (T, n)."""
    rows_t = torch.as_tensor(rows, device=L.device)
    cols_t = torch.as_tensor(cols, device=R.device)
    Lsel = L[:, rows_t, :]  # (T, n, r)
    Rsel = R[:, :, cols_t]  # (T, r, n)
    return torch.einsum("tnk,tkn->tn", Lsel, Rsel)
