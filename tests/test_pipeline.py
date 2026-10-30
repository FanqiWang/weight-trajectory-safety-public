"""End-to-end pipeline and exact resume, on the toy model with synthetic data (CPU)."""

import subprocess
import sys

import torch
from conftest import ROOT, smoke_config

from wts.config import plan_runs, run_dir
from wts.data import build_arm_examples, build_pools
from wts.lora import inject_lora
from wts.model import load_model, load_tokenizer
from wts.snapshots import list_steps, load_snapshot
from wts.train import train_run


def test_full_pipeline_on_smoke_config(tmp_path):
    out = tmp_path / "out"
    for script in ("run_experiment.py", "analyze.py"):
        args = [sys.executable, str(ROOT / "scripts" / script), str(ROOT / "configs" / "smoke.yaml"),
                "--output-dir", str(out)]
        if script == "run_experiment.py":
            args.append("--yes")
        res = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]

    cfg = smoke_config(tmp_path)
    max_steps = cfg["train"]["max_steps"]
    for setting in ("qv", "all_linear"):
        for spec in plan_runs(cfg, setting):
            rdir = run_dir(cfg, spec)
            assert (rdir / "DONE").exists()
            assert len(list_steps(rdir)) == max_steps + 1  # snapshots = steps + 1
            assert (rdir / "eval.jsonl").exists()
        adir = out / "analysis" / setting
        figs = sorted(p.name for p in (adir / "figures").glob("fig*.png"))
        assert len(figs) == 7, figs
        for name in ("summary.md", "leadtime.csv", "leadtime_summary.csv", "forecast.csv"):
            assert (adir / name).stat().st_size > 0, name


def _train(cfg, run_path, stop_after=None):
    tok = load_tokenizer(cfg)
    model = load_model(cfg, device=torch.device("cpu"))
    lora = cfg["settings"][0]["lora"]
    handle = inject_lora(model, lora["target"], lora["r"], lora["alpha"])
    pools = build_pools(cfg)
    examples = build_arm_examples(pools, 0.1, "comply", 0, "test", cfg["train"]["n_examples"])
    return train_run(model, tok, handle, examples, run_path, cfg, seed=0, save_every=1, stop_after_step=stop_after)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    cfg = smoke_config(tmp_path, train={"max_steps": 12, "deterministic": True}, save={"optimizer_every": 4})
    full, split = tmp_path / "full", tmp_path / "split"
    assert _train(cfg, full) == "done"
    assert _train(cfg, split, stop_after=7) == "stopped"  # simulate a crash after step 7; the latest checkpoint is step 4
    assert not (split / "DONE").exists()
    assert _train(cfg, split) == "done"  # resume from step 4, rewrite steps 5-7 and continue
    assert list_steps(full) == list_steps(split) == list(range(13))
    for step in range(13):
        a, b = load_snapshot(full, step), load_snapshot(split, step)
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=1e-7, msg=f"step {step} {k}")
    n_log = len((split / "train_log.jsonl").read_text().splitlines())
    assert n_log == 12  # stale log lines after step 4 were truncated on resume
