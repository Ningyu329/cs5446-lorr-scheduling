"""Diagnostic: does each congestion/risk feature actually predict arrival delay?

For every assignment decision we record, for the chosen task:

    predicted  d(i,j)        (actions * agentCounter)
    predicted  C(i,j), R(i,j)
    realised   ticks from assignment to first-errand arrival

Then we correlate ``realised - predicted`` (the *surprise*, i.e. the delay that
``d`` failed to foresee) with each feature.  A useful feature must be
**positively** correlated with the surprise: high predicted congestion/risk
should coincide with arrivals taking longer than the free-flow distance
suggests.

This is the offline validation gate from the plan (WP3): if a feature has no
correlation here, adding it to the score can only add noise, and a negative
end-to-end result is a *feature* problem, not a scheduler bug.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lorrsim.instance import load_instance
from lorrsim.schedulers import BaseScheduler, route_of
from lorrsim.world import Simulation


class RecordingScheduler(BaseScheduler):
    """Wraps S0 but records the feature vector for every assignment it makes."""

    name = "record"

    def __init__(self, kinds=("c1", "c2", "c3", "c4", "mix"), **kw):
        super().__init__(**kw)
        self.kinds = kinds
        self.records: List[dict] = []
        self._pending: Dict[int, dict] = {}

    def reset(self, sim) -> None:
        super().reset(sim)
        self.records = []
        self._pending = {}

    def plan(self, sim, free_agents, current_schedule, budget_ms):
        sched = super().plan(sim, free_agents, current_schedule, budget_ms)
        feats = sim.features
        pool = sim.taskmgr.pool
        for a in free_agents:
            tid = sched[a]
            if tid is None or tid < 0 or tid not in pool:
                continue
            t = pool[tid]
            rt = route_of(sim, sim.robots[a].loc, t)
            rec = {
                "agent": a,
                "task": tid,
                "t0": sim.tick,
                "d_actions": sim.grid.dist(sim.robots[a].loc, t.next_loc),
                "agent_counter": sim.inst.agent_counter,
                "route_len": len(rt),
                "delay_duty": feats.delay_duty(a),
                "delay_heat": feats.route_delay_heat(rt),
                "drift": feats.drift(sim, a),
            }
            for k in self.kinds:
                if k in ("c1", "c2", "c3", "c4", "mix"):
                    rec["C_" + k] = feats.congestion(a, rt, k)
                elif k in ("r1", "r2", "r3", "mix"):
                    rec["R_" + k] = feats.delay_risk(sim, a, t, rt, k)
            self._pending[a] = rec
        return sched

    def note_arrival(self, sim, agent: int, tick: int) -> None:
        rec = self._pending.pop(agent, None)
        if rec is not None:
            rec["t_arrive"] = tick
            self.records.append(rec)


def spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation (no scipy needed)."""
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--out", default="results/feature_validity.json")
    args = ap.parse_args()

    inst = load_instance(args.instance)
    sched = RecordingScheduler(shortlist_k=32, route_stride=1)
    sim = Simulation(inst, sched, seed=args.seed, horizon=args.horizon)
    orig_arrive = sim._arrive

    def arrive(r, loc):
        before = r.task_id
        t = sim.taskmgr.pool.get(before) if before >= 0 else None
        first = t is not None and t.idx_next_loc == 0 and t.locations[0] == loc
        orig_arrive(r, loc)
        if first:
            sched.note_arrival(sim, r.id, sim.tick)

    sim._arrive = arrive
    res = sim.run()

    recs = [r for r in sched.records if "t_arrive" in r]
    if not recs:
        print("no arrival records", file=sys.stderr)
        return 1

    # surprise = realised - free-flow prediction
    for r in recs:
        pred = r["d_actions"] * r["agent_counter"]
        r["pred"] = pred
        r["realised"] = r["t_arrive"] - r["t0"]
        r["surprise"] = r["realised"] - pred

    out = {
        "instance": inst.name,
        "seed": args.seed,
        "horizon": args.horizon,
        "n_records": len(recs),
        "num_task_finished": res.num_task_finished,
        "mean_surprise": round(statistics.mean(r["surprise"] for r in recs), 3),
        "median_surprise": round(statistics.median(r["surprise"] for r in recs), 3),
        "correlations_with_surprise": {},
    }
    sur = [r["surprise"] for r in recs]
    for key in sorted(recs[0].keys()):
        if key.startswith(("C_", "R_")) or key in ("delay_duty", "delay_heat", "drift", "route_len", "pred"):
            vals = [r[key] for r in recs]
            out["correlations_with_surprise"][key] = round(spearman(vals, sur), 4)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
