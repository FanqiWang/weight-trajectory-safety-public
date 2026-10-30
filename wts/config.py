"""Load a YAML config, fill in defaults, and expand it into settings, arms and runs.

- setting: where LoRA is attached (e.g. `qv` = q/v projections, `all_linear` = every linear layer).
- arm: a training-data recipe, e.g. C10 = 10% harmful-compliance + 90% benign examples.
- run: setting x arm x seed, one full training trajectory, named like "qv_C10_seed0".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype": "bfloat16",  # base-model precision; LoRA parameters are always fp32
    "device": "auto",
    # Only used when model == "tiny": the toy Qwen2 model for smoke tests
    "tiny": {
        "init_seed": 1234,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 128,
        "max_position": 256,
        "align_steps": 300,  # full-parameter pre-alignment steps (0 = keep random init)
        "align_lr": 3.0e-3,
    },
    "settings": None,  # list; see _normalize_settings
    "train": {
        "lr": 2.0e-4,
        "schedule": "constant",  # constant | linear | cosine
        "warmup_steps": 0,
        "batch_size": 16,  # effective batch size per optimizer step
        "micro_batch_size": 4,  # examples per forward pass; must divide batch_size
        "max_len": 512,
        "max_steps": 400,
        "n_examples": 1000,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "grad_clip": None,  # None = no clipping (clipping changes trajectory geometry)
        "gradient_checkpointing": False,
        "deterministic": False,
    },
    "data": {
        "harmful_source": "beavertails",  # beavertails | pku_saferlhf | synthetic
        "beavertails_split": "30k_train",
        "benign_sources": ["gsm8k", "alpaca"],  # gsm8k | alpaca | ultrachat | synthetic
        "test_source": "advbench",  # advbench | synthetic
        "n_test": 200,
        "split_harmful_by_seed_group": True,  # reference and test seeds use disjoint harmful prompts
        "max_prompt_chars": 1500,
        "max_response_chars": 3000,
        "cache_dir": "data_cache",
        "system_prompt": None,  # None = the chat template's default system prompt
        "chat_template_kwargs": {},  # e.g. a fixed date_string for Llama 3
        "synthetic": {"n_harmful": 400, "n_benign": 400, "n_test": 40, "n_gsm8k": 20},
        "arms": [],
    },
    "seeds": {"test": [0, 1, 2], "reference": [100, 101, 102, 103]},
    # reference_every: snapshot interval for reference runs (only their endpoints are analysed);
    # None = same as test runs
    "save": {"every": 1, "optimizer_every": 50, "reference_every": None},
    "eval": {
        "every": 10,
        "max_new_tokens": 64,
        "batch_size": 32,
        "refusal_judge": "keywords",
        "judge_prefix_chars": 200,
        "n_gsm8k": 0,  # > 0 also measures GSM8K accuracy
        "gsm8k_every": 50,
        "gsm8k_max_new_tokens": 256,
        "reference_runs": "endpoints",  # endpoints | all | none
    },
    "analysis": {
        "n_past": 5,
        "tau": 5,
        "dmd_window": 8,
        "esa_coords_per_module": 4096,
        "step_stride": 1,
        "skip_initial_steps": 5,  # B = 0 at init makes the first few steps special
        "free_direction_base": None,  # e.g. Qwen/Qwen2.5-1.5B; None disables the free-direction metric
        "free_direction_rank": 8,
        "reference_arms": {"C": "C100", "S": "S100", "B": "B"},
        "onset": {"k_sigma": 3.0, "k_se": 3.0, "persist": 3, "sigma_floor": 1.0e-6},
        "forecast_fraction": 0.3,
        "forecast_threshold": 0.5,
        "leadtime_metrics": ["s_H", "dir_cos", "norm", "free_frac", "vcs", "frr", "esa", "dmd_maxabs"],
        "device": "auto",
    },
    "output_dir": "outputs/default",
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_config_path"] = str(path)
    _normalize_settings(cfg)
    _validate(cfg)
    return cfg


def _normalize_settings(cfg: dict) -> None:
    """Accept either a top-level `lora:` block (single setting) or a `settings:` list."""
    if not cfg.get("settings"):
        if "lora" not in cfg:
            raise ValueError("config has neither `settings` nor `lora`")
        cfg["settings"] = [{"name": cfg.get("setting_name", "default"), "lora": cfg["lora"]}]
    for s in cfg["settings"]:
        lora = {"r": 8, "alpha": 16, "dropout": 0.0}
        lora.update(s.get("lora", {}))
        s["lora"] = lora
        s.setdefault("save_every", cfg["save"]["every"])


def _validate(cfg: dict) -> None:
    t = cfg["train"]
    if t["batch_size"] % t["micro_batch_size"] != 0:
        raise ValueError("train.batch_size must be a multiple of train.micro_batch_size")
    test, ref = set(cfg["seeds"]["test"]), set(cfg["seeds"]["reference"])
    if test & ref:
        raise ValueError(f"test and reference seeds overlap: {sorted(test & ref)} (information leak)")
    names = [s["name"] for s in cfg["settings"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate setting names")
    arms = parse_arms(cfg)
    if len({a.name for a in arms}) != len(arms):
        raise ValueError("duplicate arm names")


@dataclass(frozen=True)
class Arm:
    name: str
    p: float  # fraction of harmful examples
    target: str  # comply | refuse (meaningless when p == 0)
    role: str  # test | reference | both

    @property
    def family(self) -> str:
        if self.p == 0:
            return "B"
        return "C" if self.target == "comply" else "S"


def parse_arms(cfg: dict) -> list[Arm]:
    arms = []
    for a in cfg["data"]["arms"]:
        p = float(a.get("p", 0.0))
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"arm {a['name']}: p must be in [0, 1]")
        target = a.get("target", "none" if p == 0 else None)
        if p > 0 and target not in ("comply", "refuse"):
            raise ValueError(f"arm {a['name']}: target must be comply or refuse")
        role = a.get("role")
        if role is None:
            role = "reference" if a.get("reference_only") else "test"
        if role not in ("test", "reference", "both"):
            raise ValueError(f"arm {a['name']}: role must be test, reference or both")
        arms.append(Arm(name=a["name"], p=p, target=target, role=role))
    return arms


@dataclass(frozen=True)
class RunSpec:
    setting: str
    arm: Arm
    seed: int
    seed_group: str  # test | reference

    @property
    def name(self) -> str:
        return f"{self.setting}_{self.arm.name}_seed{self.seed}"


def save_every_for(cfg: dict, setting: dict, seed_group: str) -> int:
    if seed_group == "reference" and cfg["save"].get("reference_every"):
        return int(cfg["save"]["reference_every"])
    return int(setting["save_every"])


def get_setting(cfg: dict, name: str) -> dict:
    for s in cfg["settings"]:
        if s["name"] == name:
            return s
    raise KeyError(name)


def plan_runs(cfg: dict, setting: str) -> list[RunSpec]:
    """All runs of one setting."""
    runs = []
    for arm in parse_arms(cfg):
        groups = {"test": ["test"], "reference": ["reference"], "both": ["test", "reference"]}[arm.role]
        for g in groups:
            for seed in cfg["seeds"][g]:
                runs.append(RunSpec(setting=setting, arm=arm, seed=int(seed), seed_group=g))
    return runs


def runs_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / "runs"


def run_dir(cfg: dict, spec: RunSpec) -> Path:
    return runs_dir(cfg) / spec.name
