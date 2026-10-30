"""The seven figures. Every figure shows the C, S and B arms against training step.

Colours: C (harmful compliance) red, S (refusal) blue, B (benign) grey. Lines are means over
test seeds; shaded bands are min-max ranges.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"C": "#c0392b", "S": "#2471a3", "B": "#7f8c8d"}


def _test_runs(runs):
    return [r for r in runs if r.spec.seed_group == "test"]


def _p_levels(runs) -> list[float]:
    return sorted({r.spec.arm.p for r in _test_runs(runs) if r.spec.arm.family == "C"})


def _arm_runs(runs, family: str, p: float | None):
    out = []
    for r in _test_runs(runs):
        if r.spec.arm.family != family:
            continue
        if family != "B" and r.spec.arm.p != p:
            continue
        out.append(r)
    return out


def _band(ax, x, ys: list[np.ndarray], color, label, style="-"):
    if not ys:
        return
    Y = np.vstack(ys)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # some steps are all-NaN (metric not yet computable)
        mean = np.nanmean(Y, axis=0)
        lo, hi = np.nanmin(Y, axis=0), np.nanmax(Y, axis=0)
    ax.plot(x, mean, style, color=color, label=label, lw=1.6)
    if len(ys) > 1:
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)


def _metric_band(ax, runs, family, p, group, metric, label=None, style="-"):
    rs = [r for r in _arm_runs(runs, family, p) if metric in r.metrics.get(group, {})]
    if rs:
        _band(ax, rs[0].steps, [r.metrics[group][metric] for r in rs], COLORS[family],
              label or f"{family}{'' if family == 'B' else f' p={p:g}'}", style)


def _eval_band(ax, runs, family, p, col, style="-", label=None):
    rs = [r for r in _arm_runs(runs, family, p) if not r.eval.empty and col in r.eval]
    if not rs:
        return
    ev = [r.eval.sort_values("step").dropna(subset=[col]) for r in rs]
    x = ev[0]["step"].to_numpy()
    ys = [e[col].to_numpy() for e in ev if len(e) == len(x)]
    _band(ax, x, ys, COLORS[family], label or f"{family} {col}", style)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig1_refusal(runs, out: Path, setting: str):
    ps = _p_levels(runs) or [None]
    fig, axes = plt.subplots(2, len(ps), figsize=(4.5 * len(ps), 6.5), squeeze=False)
    for j, p in enumerate(ps):
        ax = axes[0, j]
        for fam in ("C", "S", "B"):
            _eval_band(ax, runs, fam, p, "refusal_rate", label=f"{fam} refusal")
            _eval_band(ax, runs, fam, p, "gsm8k_acc", style=":", label=f"{fam} GSM8K")
        ax.set_title(f"[{setting}] p = {p:g}" if p is not None else setting)
        ax.set_ylabel("refusal rate (solid) / GSM8K acc (dotted)")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=7)
        ax = axes[1, j]
        for fam in ("C", "S", "B"):
            rs = [r for r in _arm_runs(runs, fam, p) if len(r.loss)]
            if rs:
                n = min(len(r.loss) for r in rs)
                _band(ax, rs[0].loss_steps[:n], [r.loss[:n] for r in rs], COLORS[fam], f"{fam} loss")
        ax.set_ylabel("train loss (response tokens)")
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    _save(fig, out / "fig1_refusal.png")


def fig2_signal(runs, out: Path, setting: str):
    ps = _p_levels(runs) or [None]
    fig, axes = plt.subplots(2, len(ps), figsize=(4.8 * len(ps), 6.5), squeeze=False)
    for j, p in enumerate(ps):
        for i, metric in enumerate(("s_H", "dir_cos")):
            ax = axes[i, j]
            for fam in ("C", "S", "B"):
                _metric_band(ax, runs, fam, p, "all", metric)
            ax.set_ylabel(metric)
            ax.set_title(f"[{setting}] p = {p:g}: {metric} vs refusal" if p is not None else metric)
            ax2 = ax.twinx()
            for fam in ("C", "S"):
                _eval_band(ax2, runs, fam, p, "refusal_rate", style="--")
            ax2.set_ylim(-0.03, 1.03)
            ax2.set_ylabel("refusal rate (dashed)")
            ax.legend(fontsize=7, loc="upper left")
            if i == 1:
                ax.set_xlabel("step")
    _save(fig, out / "fig2_signal.png")


def fig3_norm(runs, out: Path, setting: str):
    ps = _p_levels(runs) or [None]
    fig, axes = plt.subplots(2, len(ps), figsize=(4.5 * len(ps), 6.5), squeeze=False)
    for j, p in enumerate(ps):
        for i, (metric, label) in enumerate((("norm", "||ΔW||_F"), ("dir_proj", "raw projection on d"))):
            ax = axes[i, j]
            for fam in ("C", "S", "B"):
                _metric_band(ax, runs, fam, p, "all", metric)
            ax.set_ylabel(label)
            ax.set_title(f"[{setting}] p = {p:g}" if p is not None else setting)
            ax.legend(fontsize=7)
        axes[1, j].set_xlabel("step")
    _save(fig, out / "fig3_norm.png")


def fig4_heatmap(runs, out: Path, setting: str):
    ps = _p_levels(runs)
    metrics = ("s_H", "dir_cos")
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 3.6), squeeze=False)
    if not ps:
        _save(fig, out / "fig4_heatmap.png")
        return
    p = ps[-1]
    c_runs, s_runs = _arm_runs(runs, "C", p), _arm_runs(runs, "S", p)
    keys = [g for g in c_runs[0].metrics if g.startswith("module:")]
    types = sorted({k.split(":")[1].split("@")[0] for k in keys}, key=_type_order)
    layers = sorted({int(k.split("@")[1]) for k in keys})
    for ax, metric in zip(axes[0], metrics):
        H = np.full((len(types), len(layers)), np.nan)
        for a, t in enumerate(types):
            for b, layer in enumerate(layers):
                g = f"module:{t}@{layer:02d}"
                H[a, b] = _late(c_runs, g, metric) - _late(s_runs, g, metric)
        vmax = np.nanmax(np.abs(H)) if np.isfinite(H).any() else 1.0
        im = ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(len(types)), types)
        ax.set_xticks(range(len(layers)), [str(x) for x in layers], fontsize=6)
        ax.set_xlabel("layer")
        ax.set_title(f"[{setting}] {metric}: C − S (p = {p:g}, last 20% of steps)")
        fig.colorbar(im, ax=ax)
    _save(fig, out / "fig4_heatmap.png")


TYPE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _type_order(t: str) -> int:
    return TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER)


def _late(runs, group, metric, frac=0.2):
    vals = []
    for r in runs:
        y = r.metrics[group].get(metric)
        if y is None:
            continue
        k = max(1, int(len(y) * frac))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals.append(np.nanmean(y[-k:]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(vals)) if vals else np.nan


def _type_groups(runs) -> list[str]:
    any_run = _test_runs(runs)[0]
    return sorted((g for g in any_run.metrics if g.startswith("type:")), key=lambda g: _type_order(g[5:]))


def fig5_predictability(runs, out: Path, setting: str):
    ps = _p_levels(runs)
    p = ps[-1] if ps else None
    groups = _type_groups(runs)
    metrics = ("vcs", "frr", "esa")
    fig, axes = plt.subplots(len(metrics), len(groups), figsize=(3.6 * len(groups), 2.6 * len(metrics)),
                             squeeze=False)
    for i, metric in enumerate(metrics):
        for j, g in enumerate(groups):
            ax = axes[i, j]
            for fam in ("C", "S", "B"):
                _metric_band(ax, runs, fam, p, g, metric)
            if i == 0:
                ax.set_title(f"{g[5:]}  (p = {p:g})" if p is not None else g[5:])
            if j == 0:
                ax.set_ylabel(metric)
            if i == len(metrics) - 1:
                ax.set_xlabel("step")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"[{setting}] structured-predictability probes by module type", y=1.0)
    _save(fig, out / "fig5_predictability.png")


def fig6_dmd(runs, out: Path, setting: str):
    ps = _p_levels(runs)
    p = ps[-1] if ps else None
    groups = ["all"] + _type_groups(runs)
    fig, axes = plt.subplots(2, len(groups), figsize=(3.6 * len(groups), 5.2), squeeze=False)
    for j, g in enumerate(groups):
        for i, metric in enumerate(("dmd_maxabs", "dmd_pred_cos")):
            ax = axes[i, j]
            for fam in ("C", "S", "B"):
                _metric_band(ax, runs, fam, p, g, metric)
            if i == 0:
                ax.set_title(g.replace("type:", ""))
                ax.axhline(1.0, color="k", lw=0.5, ls=":")
            if j == 0:
                ax.set_ylabel({"dmd_maxabs": "max |λ|", "dmd_pred_cos": "1-step pred cos"}[metric])
            if i == 1:
                ax.set_xlabel("step")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"[{setting}] sliding-window DMD (p = {p:g})" if p is not None else setting, y=1.0)
    _save(fig, out / "fig6_dmd.png")


def fig7_leadtime(lead_sum: pd.DataFrame, out: Path, setting: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if lead_sum is None or lead_sum.empty:
        ax.text(0.5, 0.5, "no matched C/S test runs", ha="center")
        _save(fig, out / "fig7_leadtime.png")
        return
    d = lead_sum[lead_sum["group"] == "all"].copy()
    arms = sorted(d["c_arm"].unique())
    metrics = list(dict.fromkeys(d["metric"]))
    offsets = np.linspace(-0.25, 0.25, max(1, len(arms)))
    for k, arm in enumerate(arms):
        sub = d[d["c_arm"] == arm].set_index("metric").reindex(metrics)
        y = np.arange(len(metrics)) + offsets[k]
        mean = sub["lead_mean"].to_numpy(dtype=float)
        lo = mean - sub["lead_min"].to_numpy(dtype=float)
        hi = sub["lead_max"].to_numpy(dtype=float) - mean
        ax.errorbar(mean, y, xerr=[np.nan_to_num(lo), np.nan_to_num(hi)], fmt="o", capsize=3, label=arm)
        for yi, m in zip(y, mean):
            if not np.isfinite(m):
                ax.text(0, yi, " n/d", va="center", fontsize=7, color="gray")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(metrics)), metrics)
    ax.set_xlabel("lead time (steps) = behavior onset − signal onset;  > 0 means weights are earlier")
    ax.set_title(f"[{setting}] lead time by metric (group = all; mean, min–max over seeds)")
    ax.legend(fontsize=8)
    _save(fig, out / "fig7_leadtime.png")


def make_all_figures(runs, lead_sum, out: Path, setting: str) -> list[Path]:
    out = Path(out)
    fig1_refusal(runs, out, setting)
    fig2_signal(runs, out, setting)
    fig3_norm(runs, out, setting)
    fig4_heatmap(runs, out, setting)
    fig5_predictability(runs, out, setting)
    fig6_dmd(runs, out, setting)
    fig7_leadtime(lead_sum, out, setting)
    return sorted(out.glob("fig*.png"))
