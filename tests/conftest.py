import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def random_trajectory(T=12, d_out=7, d_in=5, r=3, seed=0):
    """A random factored trajectory: L (T, d_out, r), R (T, r, d_in), float64.

    Smooth drift plus noise, closer to a real trajectory than pure noise and giving FRR and DMD
    non-trivial values.
    """
    g = torch.Generator().manual_seed(seed)
    L0 = torch.randn(d_out, r, generator=g, dtype=torch.float64)
    R0 = torch.randn(r, d_in, generator=g, dtype=torch.float64)
    dL = torch.randn(d_out, r, generator=g, dtype=torch.float64) * 0.3
    dR = torch.randn(r, d_in, generator=g, dtype=torch.float64) * 0.3
    t = torch.arange(T, dtype=torch.float64)[:, None, None]
    L = (L0 + t * dL) * (t + 1) / T + 0.05 * torch.randn(T, d_out, r, generator=g, dtype=torch.float64)
    R = R0 + t * dR + 0.05 * torch.randn(T, r, d_in, generator=g, dtype=torch.float64)
    L[0] = 0.0  # as with LoRA, dW = 0 at step 0
    return L, R


@pytest.fixture
def traj():
    return random_trajectory()


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def smoke_config(tmp_path, **overrides):
    from wts.config import deep_merge, load_config

    cfg = load_config(ROOT / "configs" / "smoke.yaml", {"output_dir": str(tmp_path / "out")})
    return deep_merge(cfg, overrides) if overrides else cfg
