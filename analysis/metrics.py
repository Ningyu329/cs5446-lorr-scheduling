"""Metrics + statistics layer.

Two jobs:

1. **Summarise** raw run rows into per-configuration aggregates with 95%
   confidence intervals, using a paired design where possible (same instance,
   same seed, different scheduler).
2. **Test** differences with paired non-parametric tests and report effect
   sizes, because the proposal promises "robustness rather than tuning for a
   single scenario" and the anytime planner adds run-to-run variance.

Everything here is pure Python (+ optional pandas for IO) so the analysis can be
re-run from ``results/raw/*.csv`` alone.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# basic statistics
# ---------------------------------------------------------------------
def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def sem(xs: Sequence[float]) -> float:
    return stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value (small table, enough for n<=30)."""
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return 0.0
    if df in table:
        return table[df]
    # normal approximation for large df
    return 1.96 + 2.4 / max(1, df)


def ci95(xs: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, half-width, df)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    if n == 1:
        return mean(xs), 0.0, 0
    h = t_critical_95(n - 1) * sem(xs)
    return mean(xs), h, n - 1


def pct_change(new: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (new - base) / abs(base) * 100.0


# ---------------------------------------------------------------------
# Wilcoxon signed-rank (paired, two-sided)
# ---------------------------------------------------------------------
def wilcoxon_signed_rank(diffs: Sequence[float]) -> Tuple[float, float, int]:
    """Return (W statistic, approximate two-sided p, n_effective).

    Uses the normal approximation with tie correction, which is accurate for
    n >= 8 and still informative below that (we report n alongside).
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, d) if x < 0)
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return w, 1.0, n
    z = (w - mu) / sigma
    p = 2.0 * (1.0 - _phi(abs(z)))
    return w, min(1.0, p), n


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni correction; returns adjusted p-values in input order."""
    m = len(pvals)
    idx = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


# ---------------------------------------------------------------------
# loading / aggregation
# ---------------------------------------------------------------------
def load_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if not str(r.get("status", "")).startswith("ok"):
            continue
        d = {}
        for k, v in r.items():
            if v is None or v == "":
                d[k] = None
                continue
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def aggregate(rows: Sequence[dict], metric: str = "num_task_finished") -> List[dict]:
    """Group by (scheduler, instance) -> mean, CI, n."""
    groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        groups[(str(r["scheduler"]), str(r["instance"]))].append(float(r[metric]))
    out = []
    for (sch, inst), vals in sorted(groups.items()):
        m, h, df = ci95(vals)
        out.append(
            {
                "scheduler": sch,
                "instance": inst,
                "n": len(vals),
                "mean": m,
                "ci95": h,
                "sd": stdev(vals),
                "df": df,
            }
        )
    return out


def paired_diffs(rows: Sequence[dict], base: str, other: str, metric: str) -> List[dict]:
    """Pair up runs on (instance, seed) and return per-pair differences."""
    key = lambda r: (str(r["instance"]), int(float(r["seed"])))
    by: Dict[Tuple[str, Tuple[str, int]], float] = {}
    for r in rows:
        by[(str(r["scheduler"]), key(r))] = float(r[metric])
    out = []
    for (sch, k), v in by.items():
        if sch != other:
            continue
        b = by.get((base, k))
        if b is None:
            continue
        out.append(
            {
                "instance": k[0],
                "seed": k[1],
                "base": b,
                f"{other}": v,
                "diff": v - b,
                "pct": pct_change(v, b),
            }
        )
    return sorted(out, key=lambda x: (x["instance"], x["seed"]))


def compare(
    rows: Sequence[dict],
    base: str,
    others: Sequence[str],
    metric: str = "num_task_finished",
) -> List[dict]:
    """Paired comparison of each scheduler against ``base``."""
    results = []
    raw_p = []
    for o in others:
        d = paired_diffs(rows, base, o, metric)
        if not d:
            continue
        diffs = [x["diff"] for x in d]
        w, p, n = wilcoxon_signed_rank(diffs)
        m, h, _df = ci95(diffs)
        results.append(
            {
                "base": base,
                "compare": o,
                "metric": metric,
                "n_pairs": n,
                "mean_diff": m,
                "ci95_diff": h,
                "mean_pct": mean([x["pct"] for x in d]),
                "median_pct": sorted(x["pct"] for x in d)[len(d) // 2],
                "wins": sum(1 for x in diffs if x > 0),
                "losses": sum(1 for x in diffs if x < 0),
                "wilcoxon_W": w,
                "p_value": p,
            }
        )
        raw_p.append(p)
    if raw_p:
        adj = holm(raw_p)
        for r, a in zip(results, adj):
            r["p_holm"] = a
            r["significant_05"] = a < 0.05
    return results


def format_ci(m: float, h: float, digits: int = 1) -> str:
    if h <= 0:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {h:.{digits}f}"
