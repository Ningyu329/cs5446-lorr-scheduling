#!/usr/bin/env python3
"""Regenerate simple summary figures using the small public derived CSVs.

This does NOT reproduce the original multi-panel full-run plots exactly; those
were created with `analysis/make_figures.py` from per-run logs.
"""
from pathlib import Path
import csv
from collections import defaultdict
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"published_results"/"tables"
DST=ROOT/"published_results"/"regenerated"
DST.mkdir(parents=True,exist_ok=True)

# Warehouse density effects of S3 vs S0 (domain/seed-aggregated means).
with (SRC/"density_effects.csv").open(newline="") as f:
    rows=list(csv.DictReader(f))
sub=sorted((r for r in rows if r["domain"]=="warehouse" and r["label"]=="S3"),key=lambda r:float(r["mult"]))
assert len(sub)==4
x=[float(r["mult"]) for r in sub]
y=[float(r["mean"]) for r in sub]
e=[float(r["ci95"]) for r in sub]
fig,ax=plt.subplots(figsize=(6,3.6))
ax.errorbar(x,y,yerr=e,fmt="o-",capsize=4)
ax.axhline(0,lw=1,ls="--")
ax.set(xlabel="Fleet size multiplier",ylabel="S3 vs S0 (% tasks)",title="Warehouse density (original 10 seeds)")
fig.tight_layout();fig.savefig(DST/"warehouse_density_from_csv.png",dpi=180);plt.close(fig)

# Task-level duration proxy by domain, 180 historical BANDIT runs.
with (ROOT/"published_results"/"followup"/"bandit_audit_180.csv").open(newline="") as f:
    rows=list(csv.DictReader(f))
by=defaultdict(list)
for r in rows:by[r["run"].split("-m",1)[0]].append(float(r["median_task_ticks"]))
order=["maze","room","random","city","warehouse","game"]
fig,ax=plt.subplots(figsize=(8,3.5))
ax.bar(order,[statistics.median(by[k]) for k in order]);ax.axhline(100,lw=1,ls="--",label="Arm window (100 ticks)")
ax.set(ylabel="Median task duration (ticks)",title="BANDIT feedback timescale (task proxy)")
ax.legend(frameon=False);fig.tight_layout();fig.savefig(DST/"bandit_lag_from_csv.png",dpi=180);plt.close(fig)
print("Wrote two figures to",DST)
