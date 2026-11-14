"""Print a paired comparison table for one finished phase (standard library only).

    python3 scripts/quick_summary.py [phase]      # default phase: quick

Runs sharing (domain, mult, delay, seed) see identical robots, tasks and delays,
so each scheduler is compared with S0 on the same instance.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "quick"
    files = glob.glob(os.path.join(PROJECT, "results", "runs", phase, "*", "summary.json"))
    if not files:
        sys.exit(f"no finished runs under results/runs/{phase}/ -- run scripts/ray_sweep.py first")
    runs = [json.load(open(f)) for f in files]

    by_inst = defaultdict(dict)
    for r in runs:
        by_inst[(r["domain"], r["mult"], r["delay"], r["seed"])][r["sched"]] = r
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
        if "S0" in row:
            for s in scheds:
                if s in row and s != "S0":
                    changes[s].append(row[s]["tasks_finished"] / row["S0"]["tasks_finished"] - 1)

    print("\nMean paired change in tasks finished vs S0")
    for s in scheds:
        if s in changes:
            xs = changes[s]
            print(f"  {s:<8} {100 * sum(xs) / len(xs):+6.2f}%   ({len(xs)} pairs)")
    errs = sum(r["planner_errors"] + r["schedule_errors"] + r["entry_timeouts"] for r in runs)
    print(f"\n{len(runs)} runs, planner/schedule errors + timeouts: {errs}")


if __name__ == "__main__":
    main()
