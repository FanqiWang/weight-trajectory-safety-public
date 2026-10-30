"""Stage 2 of the analysis: metrics, onsets and lead times, forecasts, figures and a summary.

Entry point: analyze_setting(cfg, setting); scripts/analyze.py calls it for every setting.
Outputs under outputs/<experiment>/analysis/<setting>/:
    metrics/<run>.npz         time series per group x metric (keys "<group>|<metric>")
    leadtime.csv              onsets and lead time per C run x metric x group
    leadtime_summary.csv      aggregated over seeds
    forecast.csv              power-law / quartic fits and crossing forecasts
    summary.md                human-readable summary
    figures/fig1...fig7.png
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RunSpec, run_dir
from ..evaluate import read_eval
from .extract import analysis_dir, extract_setting, finished_runs, module_info
from .forecast import fit_power_law, fit_quartic, forecast_crossing
from .leadtime import behavior_onset, signal_onset
from .metrics import compute_metrics, prototype_weights

LIGHT_METRICS = ("norm", "dir_cos", "s_H", "free_frac")


@dataclass
class RunData:
    spec: RunSpec
    steps: np.ndarray
    metrics: dict[str, dict[str, np.ndarray]]  # group -> metric -> series
    eval: pd.DataFrame
    loss_steps: np.ndarray = field(default_factory=lambda: np.array([]))
    loss: np.ndarray = field(default_factory=lambda: np.array([]))


def build_groups(modules: list[str]) -> dict[str, list[int]]:
    """Group names: all, type:q_proj, layer:03, module:q_proj@03."""
    groups: dict[str, list[int]] = {"all": list(range(len(modules)))}
    for i, m in enumerate(modules):
        layer, mtype = module_info(m)
        groups.setdefault(f"type:{mtype}", []).append(i)
        groups.setdefault(f"layer:{layer:02d}", []).append(i)
        groups[f"module:{mtype}@{layer:02d}"] = [i]
    return groups


def group_metrics(cache, Rg: np.ndarray, weights: dict, groups: dict, acfg: dict) -> dict:
    out = {}
    has_F = "F" in cache
    for gname, idx in groups.items():
        G = cache["G"][idx].sum(0)
        X = cache["X"][idx].sum(0)
        R = Rg[idx].sum(0)
        F = cache["F"][idx].sum(0) if has_F else None
        if gname.startswith("module:"):
            m = compute_metrics_light(G, X, R, weights, F)
        else:
            m = compute_metrics(G, X, R, weights, acfg, F=F,
                                esa_agree=cache["esa_agree"][idx].sum(0), esa_valid=cache["esa_valid"][idx].sum(0))
        out[gname] = m
    return out


def compute_metrics_light(G, X, R, weights, F):
    """Single modules only get the metrics used by the heatmap."""
    acfg = {"n_past": 10**9, "tau": 0, "dmd_window": 10**9}  # disables the window-based metrics
    full = compute_metrics(G, X, R, weights, acfg, F=F)
    return {k: v for k, v in full.items() if k in LIGHT_METRICS}


def load_train_loss(rdir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = rdir / "train_log.jsonl"
    if not path.exists():
        return np.array([]), np.array([])
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return np.array([r["step"] for r in rows]), np.array([r["loss"] for r in rows])


def compute_all_metrics(cfg: dict, setting: str, log=print) -> list[RunData]:
    info = extract_setting(cfg, setting, log)
    adir = analysis_dir(cfg, setting)
    weights = prototype_weights(info["ref_names"], info["ref_arms"], cfg["analysis"]["reference_arms"])
    groups = build_groups(info["modules"])
    (adir / "metrics").mkdir(parents=True, exist_ok=True)
    runs = []
    for spec in finished_runs(cfg, setting):
        rdir = run_dir(cfg, spec)
        with np.load(adir / "cache" / f"{spec.name}.npz") as cache:
            cache = dict(cache)
        metrics = group_metrics(cache, info["Rg"], weights, groups, cfg["analysis"])
        flat = {f"{g}|{k}": v for g, d in metrics.items() for k, v in d.items()}
        np.savez_compressed(adir / "metrics" / f"{spec.name}.npz", steps=cache["steps"], **flat)
        ev = pd.DataFrame(read_eval(rdir))
        ls, lv = load_train_loss(rdir)
        runs.append(RunData(spec=spec, steps=cache["steps"], metrics=metrics, eval=ev, loss_steps=ls, loss=lv))
    return runs


# ---------------------------------------------------------------- lead times and forecasts


def matched_pairs(runs: list[RunData]) -> list[tuple[str, float, list[RunData], list[RunData]]]:
    """Test-seed C_p runs with their matched S_p runs: [(C arm name, p, C runs, S runs)]."""
    test = [r for r in runs if r.spec.seed_group == "test"]
    out = []
    for arm_name in sorted({r.spec.arm.name for r in test if r.spec.arm.family == "C"}):
        c_runs = [r for r in test if r.spec.arm.name == arm_name]
        p = c_runs[0].spec.arm.p
        s_runs = [r for r in test if r.spec.arm.family == "S" and r.spec.arm.p == p]
        if s_runs:
            out.append((arm_name, p, c_runs, s_runs))
    return out


def leadtime_table(runs: list[RunData], cfg: dict) -> pd.DataFrame:
    acfg = cfg["analysis"]
    o = acfg["onset"]
    rows = []
    for c_arm, p, c_runs, s_runs in matched_pairs(runs):
        groups = [g for g in c_runs[0].metrics if not g.startswith("module:")]
        for c in c_runs:
            if c.eval.empty:
                b_on = float("nan")
            else:
                ev = c.eval.sort_values("step")
                b_on = behavior_onset(ev["step"].to_numpy(), ev["refusal_rate"].to_numpy(),
                                      int(ev["n"].iloc[0]), o["k_se"], o["persist"])
            for metric in acfg["leadtime_metrics"]:
                for g in groups:
                    if metric not in c.metrics[g]:
                        continue
                    s_mat = np.stack([s.metrics[g][metric] for s in s_runs])
                    s_on = signal_onset(c.steps, c.metrics[g][metric], s_mat, o["k_sigma"], o["persist"],
                                        acfg["skip_initial_steps"], o["sigma_floor"])
                    rows.append({"c_arm": c_arm, "p": p, "seed": c.spec.seed, "metric": metric, "group": g,
                                 "signal_onset": s_on, "behavior_onset": b_on, "lead": b_on - s_on})
    return pd.DataFrame(rows)


def summarize_leadtime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = df.groupby(["c_arm", "p", "metric", "group"]).agg(
        n_runs=("seed", "count"),
        n_signal=("signal_onset", lambda x: int(np.isfinite(x).sum())),
        n_behavior=("behavior_onset", lambda x: int(np.isfinite(x).sum())),
        signal_onset_mean=("signal_onset", "mean"),
        behavior_onset_mean=("behavior_onset", "mean"),
        lead_mean=("lead", "mean"),
        lead_min=("lead", "min"),
        lead_max=("lead", "max"),
    )
    return agg.reset_index()


def forecast_table(runs: list[RunData], cfg: dict) -> pd.DataFrame:
    acfg = cfg["analysis"]
    rows = []
    for r in runs:
        if r.spec.seed_group != "test" or r.spec.arm.family != "C":
            continue
        steps = r.steps.astype(float)
        upto = acfg["forecast_fraction"] * steps.max()
        late = steps >= acfg["skip_initial_steps"]
        for metric in ("s_H", "dir_cos"):
            y = r.metrics["all"][metric]
            a, pw, r2 = fit_power_law(steps[late], y[late])
            a4, r2_4 = fit_quartic(steps, np.nan_to_num(y))
            row = {"run": r.spec.name, "c_arm": r.spec.arm.name, "seed": r.spec.seed, "metric": metric,
                   "power_a": a, "power_p": pw, "power_r2": r2, "quartic_a": a4, "quartic_r2": r2_4}
            if metric == "s_H":
                row.update(forecast_crossing(steps, y, acfg["forecast_threshold"], upto))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    if df.empty:
        return "_(empty)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def behavior_table(runs: list[RunData]) -> pd.DataFrame:
    rows = []
    for r in runs:
        if r.eval.empty:
            continue
        ev = r.eval.sort_values("step")
        row = {"run": r.spec.name, "arm": r.spec.arm.name, "seed_group": r.spec.seed_group,
               "refusal_step0": ev["refusal_rate"].iloc[0], "refusal_final": ev["refusal_rate"].iloc[-1],
               "final_step": int(ev["step"].iloc[-1])}
        if "gsm8k_acc" in ev:
            g = ev.dropna(subset=["gsm8k_acc"])
            if len(g):
                row["gsm8k_final"] = g["gsm8k_acc"].iloc[-1]
        if len(r.loss):
            row["loss_final"] = float(r.loss[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def location_table(runs: list[RunData]) -> pd.DataFrame:
    """Per module type: mean C - S difference of s_H and dir_cos over the last 20% of steps."""
    rows = []
    for c_arm, p, c_runs, s_runs in matched_pairs(runs):
        types = sorted(g for g in c_runs[0].metrics if g.startswith("type:"))
        for g in types + ["all"]:
            row = {"c_arm": c_arm, "group": g}
            for metric in ("s_H", "dir_cos"):
                row[f"{metric}_C_minus_S"] = _late_mean(c_runs, g, metric) - _late_mean(s_runs, g, metric)
            rows.append(row)
    return pd.DataFrame(rows)


def _late_mean(runs: list[RunData], group: str, metric: str, frac: float = 0.2) -> float:
    vals = []
    for r in runs:
        y = r.metrics[group][metric]
        k = max(1, int(len(y) * frac))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals.append(np.nanmean(y[-k:]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(vals))


def write_summary(adir: Path, setting: str, runs: list[RunData], lead: pd.DataFrame,
                  lead_sum: pd.DataFrame, fc: pd.DataFrame) -> None:
    parts = [f"# Analysis summary: setting `{setting}`\n\n",
             "Figures are in `figures/`; full tables are the CSV files in this directory.\n\n",
             "## Behaviour (refusal rate, loss)\n\n", df_to_markdown(behavior_table(runs)),
             "\n## Lead time (group = all, over test seeds)\n\n",
             "lead = behavioural onset - signal onset, in steps; positive means the weight signal comes first. "
             "NaN means not detected. With a single test seed sigma falls back to its floor and signal onsets "
             "are not statistically meaningful (as in the smoke test).\n\n"]
    if not lead_sum.empty:
        parts.append(df_to_markdown(lead_sum[lead_sum["group"] == "all"].drop(columns=["group"])))
        t = lead_sum[lead_sum["group"].str.startswith("type:")]
        if not t.empty:
            parts.append("\n## Lead time by module type (lead_mean)\n\n")
            piv = t.pivot_table(index=["c_arm", "metric"], columns="group", values="lead_mean").reset_index()
            parts.append(df_to_markdown(piv))
    parts += ["\n## Location: C - S signal difference by module type (mean over last 20% of steps)\n\n",
              df_to_markdown(location_table(runs)),
              "\n## Forecasts (power-law exponent, quartic fit, crossing forecast)\n\n", df_to_markdown(fc)]
    (adir / "summary.md").write_text("".join(parts), encoding="utf-8")


def analyze_setting(cfg: dict, setting: str, log=print) -> Path:
    from .plots import make_all_figures

    adir = analysis_dir(cfg, setting)
    runs = compute_all_metrics(cfg, setting, log)
    lead = leadtime_table(runs, cfg)
    lead_sum = summarize_leadtime(lead)
    fc = forecast_table(runs, cfg)
    lead.to_csv(adir / "leadtime.csv", index=False)
    lead_sum.to_csv(adir / "leadtime_summary.csv", index=False)
    fc.to_csv(adir / "forecast.csv", index=False)
    make_all_figures(runs, lead_sum, adir / "figures", setting)
    write_summary(adir, setting, runs, lead, lead_sum, fc)
    log(f"  analysis of setting {setting} written to {adir}")
    return adir
