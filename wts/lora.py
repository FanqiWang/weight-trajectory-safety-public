"""A minimal LoRA implementation with fp32 adapter parameters.

The A and B matrices are the object of study, so they are read and written at every step.
A small hand-written module keeps every tensor's location, shape and precision explicit and
lets evaluation copy any step's A and B into the model without rebuilding it.
`export_peft_adapter` writes any snapshot in peft format; tests/test_data_train.py checks that
peft produces the same outputs.

Notation:
    frozen base layer   y = W x                     W: (d_out, d_in)
    LoRA update         dW = (alpha / r) * B A      A: (r, d_in), B: (d_out, r)
    forward             y = W x + (alpha / r) * B (A x)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear with a parallel low-rank branch."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        device = base.weight.device
        # Adapters are fp32: per-step updates (~1e-5) would be largely rounded away in bf16,
        # corrupting the step-to-step differences the analysis relies on.
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=device))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        h = self.dropout(x).to(self.lora_A.dtype)
        # A first, then B: O(r * (d_in + d_out)) and the dense dW is never built
        delta = (h @ self.lora_A.T) @ self.lora_B.T
        return result + (delta * self.scaling).to(result.dtype)

    def delta_weight(self) -> torch.Tensor:
        """Dense dW; only for tests and toy models."""
        return self.scaling * (self.lora_B @ self.lora_A)


def _get_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(model: nn.Module, target: list[str], r: int, alpha: float, dropout: float = 0.0) -> "LoRAHandle":
    """Replace every nn.Linear inside a decoder layer whose name ends with one of `target`
    (e.g. ["q_proj", "v_proj"]) with a LoRALinear. Freezes all base parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or "layers." not in name:
            continue
        if name.split(".")[-1] in target:
            names.append(name)
    if not names:
        raise ValueError(f"no linear layers match {target}; check the module names with print(model)")
    modules = {}
    for name in names:
        parent, attr = _get_parent(model, name)
        wrapped = LoRALinear(getattr(parent, attr), r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        modules[name] = wrapped
    return LoRAHandle(model, modules)


class LoRAHandle:
    """Single entry point to all LoRA modules of a model: reset, read/write state, parameters.

    Modules are keyed by the original linear layer's path, e.g. "model.layers.3.self_attn.q_proj";
    snapshot keys are "<module>.A" and "<module>.B".
    """

    def __init__(self, model: nn.Module, modules: dict[str, LoRALinear]):
        self.model = model
        # Sorted by name so parameter order and initialization order are machine-independent
        self.modules = dict(sorted(modules.items()))

    @property
    def names(self) -> list[str]:
        return list(self.modules)

    @property
    def scaling(self) -> float:
        return next(iter(self.modules.values())).scaling

    def parameters(self) -> list[nn.Parameter]:
        params = []
        for m in self.modules.values():
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
            params += [m.lora_A, m.lora_B]
        return params

    def num_params(self) -> int:
        return sum(m.lora_A.numel() + m.lora_B.numel() for m in self.modules.values())

    def shapes(self) -> dict[str, dict[str, list[int]]]:
        return {n: {"A": list(m.lora_A.shape), "B": list(m.lora_B.shape)} for n, m in self.modules.items()}

    @torch.no_grad()
    def reset(self, seed: int) -> None:
        """Same initialization as peft: A ~ U(-1/sqrt(d_in), 1/sqrt(d_in)), B = 0, so dW = 0 at step 0.

        Uses a private CPU generator, so the result depends only on `seed`. C and S arms with
        the same seed start from identical A matrices.
        """
        gen = torch.Generator().manual_seed(int(seed))
        for m in self.modules.values():
            fan_in = m.lora_A.shape[1]
            # kaiming_uniform with a = sqrt(5) reduces to a bound of 1 / sqrt(fan_in)
            bound = 1.0 / math.sqrt(fan_in)
            a = torch.empty(m.lora_A.shape, dtype=torch.float32).uniform_(-bound, bound, generator=gen)
            m.lora_A.copy_(a)
            m.lora_B.zero_()

    @torch.no_grad()
    def zero(self) -> None:
        """Set dW = 0 (back to the base model)."""
        for m in self.modules.values():
            m.lora_B.zero_()

    @torch.no_grad()
    def state(self) -> dict[str, torch.Tensor]:
        out = {}
        for n, m in self.modules.items():
            out[f"{n}.A"] = m.lora_A.detach().to("cpu", torch.float32).clone()
            out[f"{n}.B"] = m.lora_B.detach().to("cpu", torch.float32).clone()
        return out

    @torch.no_grad()
    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        for n, m in self.modules.items():
            m.lora_A.copy_(state[f"{n}.A"].to(m.lora_A.device, torch.float32))
            m.lora_B.copy_(state[f"{n}.B"].to(m.lora_B.device, torch.float32))

    def remove(self) -> None:
        """Restore the original nn.Linear modules (used when switching settings)."""
        for name, m in self.modules.items():
            parent, attr = _get_parent(self.model, name)
            setattr(parent, attr, m.base)
        self.modules = {}


def export_peft_adapter(state: dict[str, torch.Tensor], out_dir: str | Path, base_model: str,
                        r: int, alpha: float, target: list[str]) -> Path:
    """Write one snapshot as a peft adapter directory (for PeftModel or vLLM LoRA hot-swapping)."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key, value in state.items():
        name, which = key.rsplit(".", 1)
        tensors[f"base_model.model.{name}.lora_{which}.weight"] = value.contiguous()
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "target_modules": list(target),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return out_dir
