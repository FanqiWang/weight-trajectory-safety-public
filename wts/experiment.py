"""Experiment orchestration: for every setting x arm x seed, build data -> train -> evaluate.

Re-running the same command resumes: finished runs (with a DONE marker) skip training,
interrupted runs resume from their optimizer checkpoint, and evaluated steps are skipped.
Command-line entry point: scripts/run_experiment.py.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from .config import plan_runs, run_dir, save_every_for
from .data import build_arm_examples, build_pools, manifest
from .evaluate import evaluate_current, evaluate_run, steps_to_evaluate
from .lora import inject_lora
from .model import load_model, load_tokenizer
from .snapshots import write_meta
from .train import train_run


def git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def n_saved_steps(max_steps: int, save_every: int) -> int:
    saved = set(range(0, max_steps + 1, save_every)) | {max_steps}
    return len(saved)


def storage_estimate(n_params: int, cfg: dict, setting: dict, specs: list) -> dict:
    per_step = n_params * 4  # fp32
    optim = n_params * 4 * 4  # LoRA parameters + two Adam moments + headroom
    per_run = [per_step * n_saved_steps(cfg["train"]["max_steps"], save_every_for(cfg, setting, s.seed_group)) + optim
               for s in specs]
    return {"bytes_per_step": per_step, "bytes_per_run": max(per_run, default=0), "bytes_total": sum(per_run)}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _filter_runs(specs, only_arms, only_seeds):
    if only_arms:
        specs = [s for s in specs if s.arm.name in only_arms]
    if only_seeds:
        specs = [s for s in specs if s.seed in only_seeds]
    return specs


def run_experiment(cfg: dict, yes: bool = False, only_settings: list[str] | None = None,
                   only_arms: list[str] | None = None, only_seeds: list[int] | None = None,
                   skip_eval: bool = False, dry_run: bool = False, log=print) -> None:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    settings = [s for s in cfg["settings"] if not only_settings or s["name"] in only_settings]
    log(f"Loading model and tokenizer: {cfg['model']}")
    tok = load_tokenizer(cfg)
    model = load_model(cfg)
    log(f"  device {next(model.parameters()).device}, base dtype {next(model.parameters()).dtype}")

    # Print the full plan and storage estimate before starting
    plans = []
    for setting in settings:
        lora = setting["lora"]
        handle = inject_lora(model, lora["target"], lora["r"], lora["alpha"], lora["dropout"])
        n_params = handle.num_params()
        handle.remove()
        specs = _filter_runs(plan_runs(cfg, setting["name"]), only_arms, only_seeds)
        est = storage_estimate(n_params, cfg, setting, specs)
        plans.append((setting, specs))
        log(f"\nSetting {setting['name']}: LoRA on {lora['target']}, r={lora['r']} alpha={lora['alpha']}, "
            f"{n_params:,} parameters")
        log(f"  {len(specs)} runs; {human(est['bytes_per_step'])} per snapshot, up to "
            f"{human(est['bytes_per_run'])} per run (save_every={setting['save_every']}, reference runs "
            f"{save_every_for(cfg, setting, 'reference')}), about {human(est['bytes_total'])} in total")
        for s in specs:
            log(f"    {s.name:28s} p={s.arm.p:<5g} target={s.arm.target:7s} ({s.seed_group})")
    if dry_run:
        return
    if not yes:
        answer = input("\nStart? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            log("Cancelled")
            return

    log("\nBuilding data pools (datasets are downloaded on first use)...")
    pools = build_pools(cfg)
    log(f"  harmful pool {len(pools.harmful)} prompts (about half each for reference/test), "
        f"benign {', '.join(f'{k}={len(v)}' for k, v in pools.benign.items())}, test {len(pools.test_prompts)}")

    baseline_cache: dict = {}
    for setting, specs in plans:
        lora_cfg = setting["lora"]
        handle = inject_lora(model, lora_cfg["target"], lora_cfg["r"], lora_cfg["alpha"], lora_cfg["dropout"])
        for spec in specs:
            rdir = run_dir(cfg, spec)
            log(f"\n=== {spec.name} ===")
            save_every = save_every_for(cfg, setting, spec.seed_group)
            if not (rdir / "DONE").exists():
                examples = build_arm_examples(pools, spec.arm.p, spec.arm.target, spec.seed, spec.seed_group,
                                              cfg["train"]["n_examples"])
                rdir.mkdir(parents=True, exist_ok=True)
                (rdir / "train_manifest.json").write_text(json.dumps(manifest(examples)))
                write_meta(rdir, {
                    "run": spec.name, "setting": spec.setting, "arm": spec.arm.__dict__, "seed": spec.seed,
                    "seed_group": spec.seed_group, "scaling": handle.scaling, "r": lora_cfg["r"],
                    "alpha": lora_cfg["alpha"], "target": lora_cfg["target"], "modules": handle.names,
                    "shapes": handle.shapes(), "save_every": save_every, "config": _jsonable(cfg),
                    "git_commit": git_commit(), "torch": torch.__version__, "python": sys.version.split()[0],
                    "platform": platform.platform(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                t0 = time.time()
                train_run(model, tok, handle, examples, rdir, cfg, spec.seed, save_every, log=log)
                log(f"  training took {time.time() - t0:.0f} s")
            else:
                log("  already trained (DONE), skipping")
            if not skip_eval:
                steps = steps_to_evaluate(cfg, rdir, spec.seed_group)
                evaluate_run(model, tok, handle, rdir, cfg, pools, steps, baseline_cache, log=log)
        handle.remove()


def run_baseline(cfg: dict, log=print) -> dict:
    """Refusal rate of the base model without any adapter (should be >= 90%)."""
    tok = load_tokenizer(cfg)
    model = load_model(cfg)
    pools = build_pools(cfg)
    t0 = time.time()
    stats, records = evaluate_current(model, tok, cfg, pools, with_gsm8k=cfg["eval"]["n_gsm8k"] > 0)
    stats["seconds"] = round(time.time() - t0, 1)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(stats, indent=2))
    with (out / "baseline_generations.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(json.dumps(stats, indent=2))
    if stats["refusal_rate"] < 0.9:
        log("warning: baseline refusal rate < 90%; consider another model, or check the chat template "
            "and the refusal judge")
    return stats


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))
