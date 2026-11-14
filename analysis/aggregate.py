"""Load per-run summaries into a DataFrame and compute paired statistics.

Pairing: two runs are paired when they share (domain, mult, delay, seed) --
i.e. identical map, robots, task stream and delay seed -- and differ only in the
scheduler.  Effects are reported as the paired relative change versus a
reference scheduler (S0 by default), with a t-based 95% CI across pairs and an
exact two-sided Wilcoxon signed-rank p-value.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(PROJECT, "results", "runs")
KEY = ["domain", "mult", "delay", "seed"]


def label(row):
    if row["sched"] in ("DEFAULT", "S0"):
        return row["sched"]
    rep = row["env"].get("REP") if isinstance(row["env"], dict) else None
    tag = row["sched"]
    if row["sched"] in ("S1", "S3", "ADAPT", "BANDIT") or row["lam"]:
        tag += f"(l={row['lam']:g}"
        tag += f",g={row['gam']:g})" if row["sched"] != "S1" else ")"
    elif row["sched"] == "S2":
        tag += f"(g={row['gam']:g})"
    if rep is not None:
        tag += f"#rep{rep}"
    return tag


def load(phases=None):
    rows = []
    pats = [os.path.join(RUNS, p, "*", "summary.json") for p in (phases or ["*"])]
    for pat in pats:
        for f in glob.glob(pat):
            with open(f) as fh:
                rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["label"] = df.apply(label, axis=1)
    df["rep"] = df["env"].apply(lambda e: e.get("REP", 0) if isinstance(e, dict) else 0)
    return df


def paired(df, ref="S0", metric="tasks_finished", by=("domain",), relative=True):
    """Paired effect of every label vs ``ref`` grouped by ``by``."""
    base = df[df["label"] == ref].set_index(KEY)[metric]
    out = []
    for keys, g in df[df["label"] != ref].groupby(list(by) + ["label"]):
        g = g.set_index(KEY)
        common = g.index.intersection(base.index)
        if len(common) == 0:
            continue
        x = g.loc[common, metric].astype(float).values
        y = base.loc[common].astype(float).values
        diff = (x - y) / np.where(y == 0, np.nan, y) * 100 if relative else x - y
        diff = diff[~np.isnan(diff)]
        n = len(diff)
        m = diff.mean() if n else np.nan
        half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        try:
            p = stats.wilcoxon(diff).pvalue if n >= 5 and np.any(diff != 0) else np.nan
        except ValueError:
            p = np.nan
        rec = dict(zip(list(by) + ["label"], keys if isinstance(keys, tuple) else (keys,)))
        rec.update({"n": n, "mean": m, "ci95": half, "wins": int((diff > 0).sum()),
                    "losses": int((diff < 0).sum()), "p_wilcoxon": p})
        out.append(rec)
    return pd.DataFrame(out)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaNs are left untouched)."""
    p = np.asarray(pvals, dtype=float)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adj = p.copy()
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj
