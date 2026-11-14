"""Produce every table and figure used by the HTML report and the paper.

    python3 analysis/make_figures.py

Inputs : results/runs/{tune,main,density,scale}/*/summary.json (+ assigns.csv.gz)
Outputs: results/tables/*.csv, results/figures/*.png|pdf, results/key_numbers.json
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.aggregate import KEY, holm, load, paired  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PROJECT, "results", "figures")
TAB = os.path.join(PROJECT, "results", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# reference palette (dataviz skill, light mode), fixed slot per scheduler
COL = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "ADAPT": "#eda100",
       "BANDIT": "#e87ba4", "DEFAULT": "#4a3aa7", "S0": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DOMAINS = ["maze", "room", "random", "city", "warehouse", "game"]
DELAYS = ["none", "base", "high"]
ORDER = ["DEFAULT", "S1", "S2", "S3", "ADAPT", "BANDIT"]
NAMES = {"DEFAULT": "Start-kit default", "S0": "S0  d", "S1": "S1  d+λC", "S2": "S2  d+γR",
         "S3": "S3  d+λC+γR", "ADAPT": "S4a adaptive", "BANDIT": "S4b bandit"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)


def base_sched(label):
    return label.split("(")[0].split("#")[0]


KEYNUM = {}


# ---------------------------------------------------------------------------
def main_ablation(df):
    m = df[(df.phase == "main") & (df.rep == 0)].copy()
    m["s"] = m.label.map(base_sched)
    m = m[m.label.str.contains("#") == False]  # noqa: E712
    # absolute table
    absd = m.groupby(["domain", "delay", "s"]).tasks_finished.agg(["mean", "std", "count"]).reset_index()
    absd.to_csv(os.path.join(TAB, "main_absolute.csv"), index=False)

    res = []
    m2 = m.copy()
    m2["label"] = m2["s"]
    for metric in ("tasks_finished", "service_mean", "service_p95", "load_cv", "assign_latency_mean"):
        p = paired(m2, ref="S0", metric=metric, by=("domain", "delay"))
        p["metric"] = metric
        res.append(p)
        q = paired(m2, ref="S0", metric=metric, by=("delay",))
        q["domain"] = "ALL"
        q["metric"] = metric
        res.append(q)
        r = paired(m2, ref="S0", metric=metric, by=("phase",))
        r["domain"], r["delay"] = "ALL", "ALL"
        r["metric"] = metric
        res.append(r.drop(columns=["phase"]))
    eff = pd.concat(res, ignore_index=True)
    eff["p_holm"] = np.nan
    sel = (eff.metric == "tasks_finished") & (eff.domain != "ALL")
    eff.loc[sel, "p_holm"] = holm(eff.loc[sel, "p_wilcoxon"].values)
    eff.to_csv(os.path.join(TAB, "main_effects.csv"), index=False)

    # figure: dot + CI, small multiples per delay level
    t = eff[(eff.metric == "tasks_finished") & (eff.domain != "ALL")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, dl in zip(axes, DELAYS):
        sub = t[t.delay == dl]
        for k, s in enumerate(ORDER):
            ss = sub[sub.label == s].set_index("domain").reindex(DOMAINS)
            y = np.arange(len(DOMAINS)) + (k - 2.5) * 0.12
            ax.errorbar(ss["mean"], y, xerr=ss["ci95"], fmt="o", ms=4, color=COL[s], elinewidth=1,
                        capsize=0, label=NAMES[s] if dl == "none" else None)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(DOMAINS)))
        ax.set_yticklabels(DOMAINS)
        ax.invert_yaxis()
        ax.set_title(f"delay = {dl}", color=INK, fontsize=10)
        ax.set_xlabel("Δ tasks finished vs S0 (%)")
        ax.set_xlim(-25, 25)
    fig.legend(loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Paired change in tasks finished relative to S0 (mean ± 95% CI, 10 seeds)",
                 color=INK, fontsize=10, x=0.01, ha="left", y=1.04)
    save(fig, "fig_main_ablation")

    # headline numbers
    for s in ORDER:
        r = eff[(eff.metric == "tasks_finished") & (eff.domain == "ALL") & (eff.delay == "ALL") & (eff.label == s)]
        if len(r):
            KEYNUM[f"overall_{s}"] = {k: float(r.iloc[0][k]) for k in ("mean", "ci95", "p_wilcoxon")} | {
                "wins": int(r.iloc[0]["wins"]), "losses": int(r.iloc[0]["losses"]), "n": int(r.iloc[0]["n"])}
    return m, eff


def noise_floor(df):
    m = df[(df.phase == "main") & (df.delay == "base") & (df.seed == 0)]
    m = m.assign(s=m.label.map(base_sched))
    m = m[m.s.isin(["S0", "S3"])]
    g = m.groupby(["domain", "s"]).tasks_finished.agg(["mean", "std", "min", "max", "count"]).reset_index()
    g["cv_pct"] = g["std"] / g["mean"] * 100
    g.to_csv(os.path.join(TAB, "noise_repeats.csv"), index=False)
    KEYNUM["repeat_cv_pct"] = g.groupby("domain").cv_pct.mean().round(2).to_dict()
    return g


def throughput_curves(df):
    m = df[(df.phase == "main") & (df.rep == 0) & (df.delay == "base")]
    m = m.assign(s=m.label.map(base_sched))
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    for ax, dom in zip(axes.flat, DOMAINS):
        for s in ["S0", "DEFAULT", "S3", "ADAPT"]:
            g = m[(m.domain == dom) & (m.s == s)]
            if g.empty:
                continue
            arr = np.array([np.cumsum(b[:30]) for b in g.throughput_bins100 if len(b) >= 30])
            x = np.arange(1, 31) * 100
            ax.plot(x, arr.mean(0), color=COL[s], label=NAMES[s], lw=1.6 if s != "S0" else 2.2)
        ax.set_title(dom, fontsize=10, color=INK)
        ax.set_xlabel("tick")
        ax.set_ylabel("cumulative tasks")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Cumulative tasks finished over time, base delays (mean of 10 seeds)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "fig_throughput_curves")


def density(df):
    d = df[((df.phase == "density") | ((df.phase == "main") & (df.delay == "base"))) & (df.rep == 0)]
    d = d[d.domain.isin(["maze", "room", "random", "warehouse"])]
    d = d.assign(label=d.label.map(base_sched))
    d = d[d.label.isin(["S0", "S1", "S2", "S3", "ADAPT"])]
    p = paired(d, ref="S0", metric="tasks_finished", by=("domain", "mult"))
    p.to_csv(os.path.join(TAB, "density_effects.csv"), index=False)
    absd = d.groupby(["domain", "mult", "label"]).agg(tasks=("tasks_finished", "mean"), team=("team", "first"),
                                                      stall=("stall_frac_mean", "mean")).reset_index()
    absd.to_csv(os.path.join(TAB, "density_absolute.csv"), index=False)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, dom in zip(axes, ["maze", "room", "random", "warehouse"]):
        for s in ["S1", "S2", "S3", "ADAPT"]:
            ss = p[(p.domain == dom) & (p.label == s)].sort_values("mult")
            ax.errorbar(ss["mult"], ss["mean"], yerr=ss["ci95"], color=COL[s], marker="o", ms=4, lw=1.5,
                        capsize=0, elinewidth=1, label=NAMES[s])
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(dom, fontsize=10, color=INK)
        ax.set_xlabel("fleet size × example")
        ax.set_xticks([0.5, 1, 1.5, 2])
    axes[0].set_ylabel("Δ tasks vs S0 (%)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Effect of fleet density (base delays, 10 seeds, mean ± 95% CI)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "fig_density")
    # S0 absolute throughput vs density (does the fleet saturate?)
    return p


def sensitivity(df):
    t = df[df.phase == "tune"].copy()
    p = paired(t, ref="S0", metric="tasks_finished", by=("domain",))
    p.to_csv(os.path.join(TAB, "tune_effects.csv"), index=False)
    doms = ["random", "room", "warehouse", "city"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), gridspec_kw={"wspace": 0.45})
    for ax, dom in zip(axes, doms):
        sub = p[p.domain == dom]
        grid = np.full((3, 3), np.nan)
        for i, l in enumerate([0.5, 1, 2]):
            for j, g in enumerate([0.5, 1, 2]):
                r = sub[sub.label == f"S3(l={l:g},g={g:g})"]
                if len(r):
                    grid[i, j] = r["mean"].iloc[0]
        lim = 12
        im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["0.5", "1", "2"])
        ax.set_yticks(range(3))
        ax.set_yticklabels(["0.5", "1", "2"])
        ax.set_xlabel("γ")
        ax.set_ylabel("λ")
        ax.grid(False)
        ax.set_title(dom, fontsize=10, color=INK)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Δ tasks vs S0 (%)")
    fig.suptitle("S3 weight sensitivity on the tuning seeds (100-104, base delays)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    save(fig, "fig_sensitivity")

    # 1-D sweeps
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, s, key in ((axes[0], "S1", "l"), (axes[1], "S2", "g")):
        for k, dom in enumerate(doms):
            sub = p[(p.domain == dom) & p.label.str.startswith(s)].copy()
            sub["w"] = sub.label.str.extract(rf"{key}=([\d.]+)").astype(float)
            sub = sub.sort_values("w")
            ax.plot(sub["w"], sub["mean"], marker="o", ms=4, lw=1.5,
                    color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][k], label=dom)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("λ" if s == "S1" else "γ")
        ax.set_title(NAMES[s], fontsize=10, color=INK)
    axes[0].set_ylabel("Δ tasks vs S0 (%)")
    axes[0].legend(fontsize=8)
    save(fig, "fig_sensitivity_1d")
    return p


def feature_validity(df):
    """Does C / R explain the part of the realised completion time that d misses?"""
    rows = []
    for _, r in df[(df.phase == "main") & (df.rep == 0) & (df.label.str.startswith("S3"))].iterrows():
        f = os.path.join(PROJECT, "results", "runs", r.phase, r.job_id, "assigns.csv.gz")
        if not os.path.exists(f):
            continue
        a = pd.read_csv(f)
        a = a[a.realised_ticks > 0]
        a["domain"], a["delay"], a["seed"], a["ctr"] = r.domain, r.delay, r.seed, r.agent_counter
        rows.append(a)
    if not rows:
        return None
    a = pd.concat(rows)
    out = []
    for (dom, dl), g in a.groupby(["domain", "delay"]):
        y = g.realised_ticks.values.astype(float)
        X0 = np.c_[np.ones(len(g)), g.d.values * g.ctr.values]
        b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        res = y - X0 @ b0
        r2_0 = 1 - res.var() / y.var()
        X1 = np.c_[X0, g.C.values, g.R.values]
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        r2_1 = 1 - (y - X1 @ b1).var() / y.var()
        out.append({"domain": dom, "delay": dl, "n": len(g), "ticks_per_step": float(b0[1]),
                    "r2_d": r2_0, "r2_dCR": r2_1,
                    "corr_resid_C": np.corrcoef(res, g.C)[0, 1] if g.C.std() > 0 else np.nan,
                    "corr_resid_R": np.corrcoef(res, g.R)[0, 1] if g.R.std() > 0 else np.nan,
                    "coef_C": float(b1[2]), "coef_R": float(b1[3])})
    fv = pd.DataFrame(out)
    fv.to_csv(os.path.join(TAB, "feature_validity.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, col, title in ((axes[0], "corr_resid_C", "C (congestion)"), (axes[1], "corr_resid_R", "R (delay risk)")):
        for k, dl in enumerate(DELAYS):
            ss = fv[fv.delay == dl].set_index("domain").reindex(DOMAINS)
            ax.bar(np.arange(len(DOMAINS)) + (k - 1) * 0.27, ss[col], width=0.25,
                   color=["#86b6ef", "#2a78d6", "#104281"][k], label=f"delay={dl}")
        ax.set_xticks(range(len(DOMAINS)))
        ax.set_xticklabels(DOMAINS, rotation=25)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(f"corr(residual of d-only fit, {title})", fontsize=9.5, color=INK)
    axes[0].legend(fontsize=8)
    fig.suptitle("Feature validity: do C and R predict the completion time that distance misses? (S3 runs)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, "fig_feature_validity")
    return fv


def delay_estimation(df):
    m = df[(df.phase == "main") & (df.sched.isin(["S2", "S3"])) & (df.delay != "none")]
    m = m.assign(p_hat=m.p_hat.astype(float), dur_hat=m.dur_hat.astype(float))
    g = m.groupby(["domain", "delay"]).agg(true_p=("true_p", "first"), p_hat=("p_hat", "mean"),
                                           true_dur=("true_dur", "first"), dur_hat=("dur_hat", "mean"),
                                           p_hat_sd=("p_hat", "std"), dur_hat_sd=("dur_hat", "std")).reset_index()
    g.to_csv(os.path.join(TAB, "delay_estimation.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for k, dl in enumerate(["base", "high"]):
        ss = g[g.delay == dl]
        mk = "o" if dl == "base" else "s"
        axes[0].scatter(ss.true_p, ss.p_hat, color=["#2a78d6", "#eb6834"][k], marker=mk, s=28, label=dl, zorder=3)
        axes[1].scatter(ss.true_dur, ss.dur_hat, color=["#2a78d6", "#eb6834"][k], marker=mk, s=28, label=dl, zorder=3)
        for _, r in ss.iterrows():
            axes[0].annotate(r.domain, (r.true_p, r.p_hat), fontsize=7, color=INK2, xytext=(3, 3),
                             textcoords="offset points")
    for ax, lab in ((axes[0], "per-agent onset prob. per tick"), (axes[1], "mean duration (ticks)")):
        lo, hi = ax.get_xlim()
        ax.plot([0, hi], [0, hi], color=INK2, lw=1, ls="--")
        ax.set_xlabel(f"true {lab}")
        ax.set_ylabel("online estimate")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("Online delay-model estimates vs the hidden delayConfig", x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.tight_layout()
    save(fig, "fig_delay_estimation")
    return g


def compute_cost(df):
    m = df[(df.phase == "main") & (df.rep == 0)]
    m = m.assign(s=m.label.map(base_sched))
    g = m.groupby(["domain", "s"]).agg(ms_mean=("sched_ms_mean", "mean"), ms_p95=("sched_ms_p95", "mean"),
                                       ms_max=("sched_ms_max", "max"), timeouts=("sched_timeouts", "sum"),
                                       entry_timeouts=("entry_timeouts", "sum"),
                                       sched_errors=("schedule_errors", "sum"),
                                       wall_s=("wall_s", "mean")).reset_index()
    g.to_csv(os.path.join(TAB, "compute_cost.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    ss = ["DEFAULT", "S0", "S1", "S2", "S3", "ADAPT", "BANDIT"]
    for k, s in enumerate(ss):
        v = g[g.s == s].set_index("domain").reindex(DOMAINS)
        ax.bar(np.arange(len(DOMAINS)) + (k - 3) * 0.11, v.ms_p95, width=0.1, color=COL[s], label=NAMES[s])
    ax.set_yscale("log")
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("scheduler ms per call (p95)")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Scheduler compute per planning update (budget ≈ 490 ms)", fontsize=10, color=INK, loc="left")
    save(fig, "fig_compute_cost")
    return g


def adaptive_traces(df):
    m = df[(df.phase == "main") & (df.rep == 0) & (df.sched.isin(["ADAPT", "BANDIT"]))]
    g = m.groupby(["domain", "delay", "sched"]).agg(lam_t=("lam_t_mean", "mean"), gam_t=("gam_t_mean", "mean"),
                                                    stall=("stall_frac_mean", "mean"),
                                                    delayed=("delay_frac_obs_mean", "mean")).reset_index()
    g.to_csv(os.path.join(TAB, "adaptive_weights.csv"), index=False)
    b = m[m.sched == "BANDIT"]
    pulls = np.zeros((len(DOMAINS) * 3, 6))
    labels = []
    for i, (dom, dl) in enumerate([(d, l) for d in DOMAINS for l in DELAYS]):
        labels.append(f"{dom}/{dl}")
        for arr in b[(b.domain == dom) & (b.delay == dl)].arm_pulls:
            for a in arr:
                pulls[i, a] += 1
    pulls = pulls / np.maximum(1, pulls.sum(1, keepdims=True))
    pd.DataFrame(pulls, index=labels, columns=["(0,0)", "(1,0)", "(0,1)", "(.5,.5)", "(1,1)", "(2,2)"]).to_csv(
        os.path.join(TAB, "bandit_arm_share.csv"))
    fig, ax = plt.subplots(figsize=(5.5, 6))
    im = ax.imshow(pulls, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["(0,0)", "(λ*,0)", "(0,γ*)", "½(λ*,γ*)", "(λ*,γ*)",
                        "2(λ*,γ*)"], fontsize=7.5, rotation=30)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="share of windows")
    ax.set_title("Discounted-UCB arm usage (10 seeds pooled)", fontsize=10, color=INK, loc="left")
    save(fig, "fig_bandit_arms")
    return g


def ray_usage():
    mans = []
    for f in glob.glob(os.path.join(PROJECT, "results", "runs", "*", "_manifest_*.json")):
        with open(f) as fh:
            mans.append(json.load(fh))
    rows = [{"phase": m["phase"], "jobs": m["n_run"], "failed": m["n_failed"], "wall_min": m["wall_s"] / 60,
             "cpus_per_job": m["cpus_per_job"], "finished_at": m["finished_at"]} for m in mans]
    t = pd.DataFrame(rows).sort_values("finished_at")
    t.to_csv(os.path.join(TAB, "ray_sweeps.csv"), index=False)
    return t


def scale_table(df):
    s = df[df.phase == "scale"]
    g = s.groupby("sched").agg(tasks=("tasks_finished", "mean"), ms_mean=("sched_ms_mean", "mean"),
                               ms_p95=("sched_ms_p95", "mean"), idle=("idle_agents_frac", "mean")).reset_index()
    g.to_csv(os.path.join(TAB, "scale_iron.csv"), index=False)
    return g


def spatial_heatmaps(df):
    """PlanViz-style wait maps: where do robots wait under S0 vs S3 (base delays)?"""
    from analysis.planviz_lite import heatmaps, read_map

    cases = [("room", "room.domain/maps/room-64-64-16.map"),
             ("warehouse", "warehouse.domain/maps/warehouse_long_corridor_large.map")]
    rows = []
    for dom, mp in cases:
        h, w, grid = read_map(os.path.join(PROJECT, "data", "instances", mp))
        acc = {}
        for s in ("S0", "S3"):
            wait_sum = np.zeros((h, w))
            sub = df[(df.phase == "main") & (df.domain == dom) & (df.delay == "base") & (df.rep == 0)
                     & (df.sched == s)].sort_values("seed").head(5)
            for _, r in sub.iterrows():
                occ, wait, _ = heatmaps(os.path.join(PROJECT, "results", "runs", "main", r.job_id,
                                                     "output.json.gz"), h, w)
                wait_sum += wait
                top = max(1, (~grid).sum() // 100)
                rows.append({"domain": dom, "sched": s, "seed": r.seed, "wait_share": wait.sum() / occ.sum(),
                             "obstacle_share": occ[grid].sum() / occ.sum(),
                             "top1pct_cells_wait_share": np.sort(wait.ravel())[::-1][:top].sum() / max(1, wait.sum())})
            acc[s] = wait_sum / max(1, len(sub))
        wide = w > 2 * h
        fig, axes = plt.subplots(3, 1, figsize=(10, 7.6)) if wide else plt.subplots(1, 3, figsize=(11, 3.6))
        vmax = np.percentile(np.r_[acc["S0"][~grid], acc["S3"][~grid]], 99.5)
        diff = acc["S3"] - acc["S0"]
        lim = np.percentile(np.abs(diff[~grid]), 99.5) or 1
        panels = [("S0", acc["S0"], "Blues", 0, vmax, "wait ticks / cell, S0"),
                  ("S3", acc["S3"], "Blues", 0, vmax, "wait ticks / cell, S3"),
                  ("diff", diff, "RdBu_r", -lim, lim, "S3 − S0 (red = more waiting under S3)")]
        for ax, (_, arr, cmap, lo, hi, title) in zip(axes, panels):
            ax.imshow(np.ma.masked_where(~grid, np.ones_like(arr)), cmap="Greys", vmin=0, vmax=3,
                      interpolation="nearest")
            im = ax.imshow(np.ma.masked_where(grid, arr), cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(f"{dom}: {title}", fontsize=9, color=INK, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
        fig.suptitle(f"Where robots wait ({dom}, base delays, mean of 5 seeds; grey = obstacles)",
                     x=0.01, ha="left", fontsize=10, color=INK)
        save(fig, f"fig_wait_{dom}")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(TAB, "wait_maps.csv"), index=False)
    KEYNUM["wait_share"] = {f"{d}/{s}": v for (d, s), v in t.groupby(["domain", "sched"]).wait_share.mean().round(4).items()}
    KEYNUM["decoder_obstacle_share_max"] = float(t.obstacle_share.max())
    return t


def secondary_metrics(eff):
    t = eff[(eff.domain != "ALL")].pivot_table(index=["metric", "label"], columns=["delay"], values="mean",
                                                aggfunc="mean")
    t.to_csv(os.path.join(TAB, "secondary_by_delay.csv"))


if __name__ == "__main__":
    df = load()
    print("runs loaded:", len(df), df.phase.value_counts().to_dict())
    KEYNUM["n_runs"] = int(len(df))
    KEYNUM["runs_by_phase"] = df.phase.value_counts().to_dict()
    KEYNUM["schedule_errors_total"] = int(df.schedule_errors.sum())
    KEYNUM["planner_errors_total"] = int(df.planner_errors.sum())
    KEYNUM["sim_ticks_total"] = int(df.ticks.sum())
    KEYNUM["cpu_hours"] = float(df.wall_s.sum() * 2 / 3600)
    if (df.phase == "tune").any():
        sensitivity(df)
    if (df.phase == "main").any():
        m, eff = main_ablation(df)
        noise_floor(df)
        throughput_curves(df)
        feature_validity(df)
        delay_estimation(df)
        compute_cost(df)
        adaptive_traces(df)
        secondary_metrics(eff)
        spatial_heatmaps(df)
    if (df.phase == "density").any():
        density(df)
    if (df.phase == "scale").any():
        scale_table(df)
    ray_usage()
    with open(os.path.join(PROJECT, "results", "key_numbers.json"), "w") as f:
        json.dump(KEYNUM, f, indent=1, default=float)
    print(json.dumps(KEYNUM, indent=1, default=float)[:3000])
