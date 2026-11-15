"""Print a paired comparison table for one finished phase (standard library only).

    python3 scripts/quick_summary.py [phase]      # default phase: quick

Runs sharing (domain, mult, delay, seed) see identical robots, tasks and delays,
so each scheduler is compared with S0 on the same instance.
"""

from __future__ import annotations

import glob
import argparse
import json
import os
import sys
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", nargs="?", default="quick")
    ap.add_argument("--config", help="require every run in this experiment config to be complete")
    args = ap.parse_args()
    phase = args.phase
    root = os.path.join(PROJECT, "results", "runs", phase)
    if args.config:
        from expand import expand_config

        jobs = expand_config(args.config)
        if not jobs or any(j["phase"] != phase for j in jobs):
            sys.exit("config must contain jobs for the requested phase")
        files = [os.path.join(root, j["job_id"], "summary.json") for j in jobs]
        missing = [os.path.basename(os.path.dirname(f)) for f in files
                   if not (os.path.isfile(f) and os.path.isfile(os.path.join(os.path.dirname(f), "DONE")))]
        if missing:
            sys.exit(f"incomplete experiment: {len(missing)} run(s) missing DONE/summary.json: " + ", ".join(missing))
    else:
        files = [f for f in glob.glob(os.path.join(root, "*", "summary.json"))
                 if os.path.isfile(os.path.join(os.path.dirname(f), "DONE"))]
    if not files:
        sys.exit(f"no finished runs under results/runs/{phase}/ -- run scripts/ray_sweep.py first")
    runs = []
    for path in files:
        with open(path) as f:
            runs.append(json.load(f))
    if len({r["horizon"] for r in runs}) != 1:
        sys.exit("cannot compare runs with different horizons; use separate phases")

    by_inst = defaultdict(dict)
    for r in runs:
        row = by_inst[(r["domain"], r["mult"], r["delay"], r["seed"])]
        if r["sched"] in row:
            sys.exit("multiple weight/config variants for one scheduler; select a config with --config")
        row[r["sched"]] = r
    scheds = sorted({r["sched"] for r in runs}, key=lambda s: (s != "DEFAULT", s))

    print(f"Tasks finished ({phase}, {runs[0]['horizon']} ticks)\n")
    head = f"{'instance':<26}" + "".join(f"{s:>10}" for s in scheds)
    print(head)
    print("-" * len(head))
    changes = defaultdict(list)
    for key in sorted(by_inst):
        row = by_inst[key]
        name = f"{key[0]} m{key[1]:g} {key[2]} s{key[3]}"
        print(f"{name:<26}" + "".join(f"{row[s]['tasks_finished']:>10}" if s in row else f"{'-':>10}" for s in scheds))
        if "S0" in row and row["S0"]["tasks_finished"] > 0:
            for s in scheds:
                if s in row and s != "S0":
                    changes[s].append(row[s]["tasks_finished"] / row["S0"]["tasks_finished"] - 1)
        elif "S0" in row:
            print("  percentage change undefined: S0 finished zero tasks")

    print("\nMean paired change in tasks finished vs S0")
    for s in scheds:
        if s in changes:
            xs = changes[s]
            print(f"  {s:<8} {100 * sum(xs) / len(xs):+6.2f}%   ({len(xs)} pairs)")
    errs = sum(r["planner_errors"] + r["schedule_errors"] + r["entry_timeouts"] for r in runs)
    print(f"\n{len(runs)} runs, planner/schedule errors + timeouts: {errs}")
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
