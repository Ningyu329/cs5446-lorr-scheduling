"""Figure generation from ``results/raw/*.csv``.

Produces the report's main figures as standalone SVG (plus PNG previews):

  fig1_ablation      numTaskFinished by configuration, per instance, with CI
  fig2_effect        relative change vs S0 as a horizontal bar chart
  fig3_weights       lambda / gamma sweeps (where the sign flips)
  fig4_density       throughput vs fleet density
  fig5_delay         throughput vs delay severity
  fig6_trajectory    S4 adaptive lambda(t), gamma(t) over one episode
  fig7_pareto        travel distance vs tasks completed (the mechanism)
  fig8_mechanism     wait ticks vs detour per completed task

Usage:
    python3 analysis/figures.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.metrics import aggregate, ci95, compare, load_rows, mean, paired_diffs  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT, "results")
FIGS = os.path.join(RESULTS, "figures")

# Tencent-blue-led palette, colour-blind safe ordering
PALETTE = ["#0052D9", "#00A870", "#ED7B2F", "#7B4FE0", "#D54941", "#0594FA", "#B37F00"]
GREY = "#8A9099"
INK = "#1B1F27"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": "#D3D8E0",
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "text.color": INK,
        "xtick.color": "#5B6371",
        "ytick.color": "#5B6371",
        "axes.grid": True,
        "grid.color": "#EBEEF3",
        "grid.linewidth": 0.8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

PRETTY = {
    "S0": "S0 distance-only",
    "S1": "S1 + congestion",
    "S2": "S2 + delay risk",
    "S3": "S3 + both",
    "S4": "S4 adaptive",
}
INST_PRETTY = {
    "random-example_400": "random (open, 10% obstacles)",
    "room-example_150": "room (door bottlenecks)",
    "fulfill-example_2500": "warehouse (long corridors)",
    "maze-example_40": "maze (low connectivity)",
}
ORDER = ["S0", "S1", "S2", "S3", "S4"]


def _save(fig, name: str) -> List[str]:
    os.makedirs(FIGS, exist_ok=True)
    paths = []
    for ext in ("svg", "png"):
        p = os.path.join(FIGS, f"{name}.{ext}")
        fig.savefig(p, dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def _order_scheds(present: Sequence[str]) -> List[str]:
    return [s for s in ORDER if s in present] + sorted(set(present) - set(ORDER))


# ---------------------------------------------------------------------
# fig 1: main ablation
# ---------------------------------------------------------------------
def fig_ablation(rows: List[dict]) -> List[str]:
    agg = aggregate(rows, "num_task_finished")
    insts = sorted({a["instance"] for a in agg})
    scheds = _order_scheds({a["scheduler"] for a in agg})
    fig, axes = plt.subplots(1, len(insts), figsize=(3.6 * len(insts), 3.4), squeeze=False)
    for ax, inst in zip(axes[0], insts):
        sub = {a["scheduler"]: a for a in agg if a["instance"] == inst}
        vals = [sub[s]["mean"] for s in scheds if s in sub]
        errs = [sub[s]["ci95"] for s in scheds if s in sub]
        labels = [s for s in scheds if s in sub]
        colors = [PALETTE[ORDER.index(s) % len(PALETTE)] if s in ORDER else GREY for s in labels]
        bars = ax.bar(range(len(labels)), vals, yerr=errs, capsize=3.5,
                      color=colors, edgecolor="white", linewidth=0.8,
                      error_kw={"ecolor": "#9AA1AD", "lw": 1.1})
        base = sub["S0"]["mean"] if "S0" in sub else None
        for b, v, s in zip(bars, vals, labels):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8, color=INK,
                    fontweight="600" if s == "S0" else "400")
            if base and s != "S0":
                d = (v - base) / base * 100
                ax.text(b.get_x() + b.get_width() / 2, v * 0.5, f"{d:+.1f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="600")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(INST_PRETTY.get(inst, inst), fontsize=9.5)
        ax.set_ylim(0, max(vals) * 1.22 if vals else 1)
        if ax is axes[0][0]:
            ax.set_ylabel("tasks completed")
    fig.suptitle("Main ablation: tasks completed by assignment policy (mean ± 95% CI)",
                 fontsize=11, y=1.02)
    return _save(fig, "fig1_ablation")


# ---------------------------------------------------------------------
# fig 2: effect vs baseline
# ---------------------------------------------------------------------
def fig_effect(rows: List[dict]) -> List[str]:
    scheds = [s for s in _order_scheds({r["scheduler"] for r in rows}) if s != "S0"]
    insts = sorted({r["instance"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.6, 0.9 + 0.72 * len(scheds) * max(1, len(insts)) / 2))
    y = 0
    yticks, ylabels = [], []
    for inst in insts:
        for s in scheds:
            pcts = [x["pct"] for x in paired_diffs([r for r in rows if r["instance"] == inst], "S0", s, "num_task_finished")]
            if not pcts:
                continue
            m, h, _ = ci95(pcts)
            color = "#C0392B" if m < 0 else "#0E9A66"
            ax.barh(y, m, xerr=h, color=color, alpha=0.88, height=0.62,
                    capsize=3, error_kw={"ecolor": "#9AA1AD", "lw": 1.0})
            ax.text(m + (h if m >= 0 else -h) + (0.4 if m >= 0 else -0.4), y,
                    f"{m:+.1f}%", va="center", ha="left" if m >= 0 else "right",
                    fontsize=8, fontweight="600", color=color)
            yticks.append(y)
            ylabels.append(f"{s} · {INST_PRETTY.get(inst, inst).split(' ')[0]}")
            y += 1
        y += 0.45
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlabel("change in tasks completed vs S0 (%)")
    ax.set_title("Effect of each extension relative to the distance-only baseline", fontsize=11)
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig2_effect")


# ---------------------------------------------------------------------
# fig 3: weight sweeps
# ---------------------------------------------------------------------
def _sweep_series(rows: List[dict], param: str, sched: str):
    pts = defaultdict(list)
    for r in rows:
        if str(r["scheduler"]) != sched:
            continue
        v = r.get(param)
        if v in (None, "", 0.0) and param not in ("lam", "gam"):
            continue
        if v is None or v == "":
            continue
        pts[float(v)].append(float(r["num_task_finished"]))
    return dict(sorted(pts.items()))


def fig_weights(rows_lam: List[dict], rows_gam: List[dict]) -> List[str]:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    specs = [
        (axes[0], rows_lam, "lam", "S1", r"$\lambda$  (congestion weight)"),
        (axes[1], rows_gam, "gam", "S2", r"$\gamma$  (delay-risk weight)"),
    ]
    base_vals = {}
    for r in rows_lam + rows_gam:
        if str(r["scheduler"]) == "S0":
            base_vals.setdefault(str(r["instance"]), []).append(float(r["num_task_finished"]))
    for ax, rows, param, sched, label in specs:
        series = _sweep_series(rows, param, sched)
        if not series:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        xs = list(series.keys())
        ys = [mean(v) for v in series.values()]
        errs = [ci95(v)[1] for v in series.values()]
        ax.errorbar(xs, ys, yerr=errs, marker="o", ms=4.5, lw=1.7,
                    color=PALETTE[0], capsize=3, ecolor="#9AA1AD")
        bases = [mean(v) for v in base_vals.values()]
        if bases:
            b = mean(bases)
            ax.axhline(b, ls="--", lw=1.3, color=GREY)
            ax.text(xs[-1], b, " S0 baseline", va="bottom", ha="right",
                    fontsize=8, color="#5B6371")
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("tasks completed")
        ax.set_title(("S1 congestion sweep" if param == "lam" else "S2 delay sweep"),
                     fontsize=9.5)
    fig.suptitle("Weight sweeps: does the extra term ever pay for itself?", fontsize=11, y=1.02)
    return _save(fig, "fig3_weights")


# ---------------------------------------------------------------------
# fig 4 / 5: density and delay axes
# ---------------------------------------------------------------------
def _axis_plot(rows: List[dict], axis: str, xlabel: str, name: str, title: str) -> List[str]:
    scheds = _order_scheds({r["scheduler"] for r in rows})
    xs_all = sorted({float(r[axis]) for r in rows if r.get(axis) is not None})
    if not xs_all:
        return []
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for i, s in enumerate(scheds):
        pts = defaultdict(list)
        for r in rows:
            if str(r["scheduler"]) != s or r.get(axis) is None:
                continue
            pts[float(r[axis])].append(float(r["num_task_finished"]))
        if not pts:
            continue
        xs = sorted(pts)
        ys = [mean(pts[x]) for x in xs]
        errs = [ci95(pts[x])[1] for x in xs]
        ax.errorbar(xs, ys, yerr=errs, marker="o", ms=4.5, lw=1.8, capsize=3,
                    color=PALETTE[ORDER.index(s) % len(PALETTE)] if s in ORDER else GREY,
                    label=PRETTY.get(s, s), ecolor="#C7CCD6")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("tasks completed")
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, name)


# ---------------------------------------------------------------------
# fig 6: adaptive weight trajectory
# ---------------------------------------------------------------------
def fig_trajectory(trace: dict) -> List[str]:
    if not trace or not trace.get("lam"):
        return []
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True)
    t = trace["ticks"]
    axes[0].plot(t, trace["lam"], color=PALETTE[0], lw=1.6, label=r"$\lambda(t)$ congestion")
    axes[0].plot(t, trace["gam"], color=PALETTE[2], lw=1.6, label=r"$\gamma(t)$ delay risk")
    axes[0].set_ylabel("weight")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[0].set_title("S4 adaptive weights follow observable system pressure", fontsize=10.5)
    axes[1].plot(t, trace["backlog"], color=PALETTE[1], lw=1.4, label="backlog pressure")
    axes[1].plot(t, trace["block"], color=PALETTE[4], lw=1.4, label="blocked-robot share")
    axes[1].set_ylabel("normalised pressure")
    axes[1].set_xlabel("execution tick")
    axes[1].legend(fontsize=8, frameon=False)
    return _save(fig, "fig6_trajectory")


# ---------------------------------------------------------------------
# fig 7: the mechanism (pareto)
# ---------------------------------------------------------------------
def fig_pareto(rows: List[dict]) -> List[str]:
    scheds = _order_scheds({r["scheduler"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for s in scheds:
        sub = [r for r in rows if str(r["scheduler"]) == s]
        if not sub:
            continue
        moves = mean([float(r["mean_moves_per_robot"]) for r in sub])
        fin = mean([float(r["num_task_finished"]) for r in sub])
        ax.scatter(moves, fin, s=95,
                   color=PALETTE[ORDER.index(s) % len(PALETTE)] if s in ORDER else GREY,
                   edgecolor="white", linewidth=1.4, zorder=3, label=PRETTY.get(s, s))
        ax.annotate(s, (moves, fin), textcoords="offset points", xytext=(7, 5),
                    fontsize=8.5, fontweight="600")
    ax.set_xlabel("mean moves per robot  (travel work spent)")
    ax.set_ylabel("tasks completed")
    ax.set_title("The mechanism: extra term = extra detour, not less queueing", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, "fig7_pareto")


def fig_mechanism(rows: List[dict]) -> List[str]:
    """Detour per completed task -- the quantity that actually moves."""
    scheds = _order_scheds({r["scheduler"] for r in rows})
    vals, labels, colors = [], [], []
    for s in scheds:
        sub = [r for r in rows if str(r["scheduler"]) == s]
        if not sub:
            continue
        moves = mean([float(r["mean_moves_per_robot"]) * float(r["team_size"]) for r in sub])
        fin = mean([float(r["num_task_finished"]) for r in sub])
        vals.append(moves / fin if fin else 0)
        labels.append(s)
        colors.append(PALETTE[ORDER.index(s) % len(PALETTE)] if s in ORDER else GREY)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
    base = vals[0] if vals else 1
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}\n({(v-base)/base*100:+.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("robot moves per completed task")
    ax.set_title("Detour cost of each policy", fontsize=10.5)
    ax.set_ylim(0, max(vals) * 1.28 if vals else 1)
    return _save(fig, "fig8_mechanism")


# ---------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--core", default=os.path.join(RESULTS, "raw", "core.csv"))
    ap.add_argument("--lam", default=os.path.join(RESULTS, "raw", "sweep_lam.csv"))
    ap.add_argument("--gam", default=os.path.join(RESULTS, "raw", "sweep_gam.csv"))
    ap.add_argument("--density", default=os.path.join(RESULTS, "raw", "density.csv"))
    ap.add_argument("--delay", default=os.path.join(RESULTS, "raw", "delay_severity.csv"))
    ap.add_argument("--trace", default=os.path.join(RESULTS, "raw", "s4_trace.json"))
    args = ap.parse_args()

    made: List[str] = []
    if os.path.exists(args.core):
        rows = load_rows(args.core)
        print(f"core: {len(rows)} runs")
        made += fig_ablation(rows)
        made += fig_effect(rows)
        made += fig_pareto(rows)
        made += fig_mechanism(rows)

    if os.path.exists(args.lam) or os.path.exists(args.gam):
        rl = load_rows(args.lam) if os.path.exists(args.lam) else []
        rg = load_rows(args.gam) if os.path.exists(args.gam) else []
        # baseline rows are included in the sweep files too
        made += fig_weights(rl, rg)

    if os.path.exists(args.density):
        made += _axis_plot(load_rows(args.density), "team_size",
                           "fleet size (robots)", "fig4_density",
                           "Fleet density: when does congestion awareness start to matter?")
    if os.path.exists(args.delay):
        made += _axis_plot(load_rows(args.delay), "delay_p",
                           "per-tick delay probability", "fig5_delay",
                           "Delay severity: when does risk awareness start to matter?")
    if os.path.exists(args.trace):
        with open(args.trace, "r", encoding="utf-8") as fh:
            made += fig_trajectory(json.load(fh))

    for p in made:
        print("wrote", os.path.relpath(p, PROJECT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
