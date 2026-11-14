"""Experiment runner: sweeps schedulers x instances x seeds and writes tidy CSV.

Usage
-----
    python3 scripts/run_experiments.py --config experiments/configs/core.yaml
    python3 scripts/run_experiments.py --preset core --workers 6
    python3 scripts/run_experiments.py --preset sweep_lam --workers 6

Every run writes one row to ``results/raw/runs.csv`` plus a per-run JSON in
``results/raw/<run_id>/manifest.json`` recording the exact configuration, so
any number in the report can be traced back and reproduced.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT, "data", "instances")
RESULTS = os.path.join(PROJECT, "results")

# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------
#: three maps chosen to span the structural axes of the study
CORE_INSTANCES = [
    "random.domain/random-example_400.json",       # open, 10% obstacles
    "room.domain/room-example_150.json",           # door bottlenecks
    "warehouse.domain/fulfill-example_2500.json",  # long corridors
]

#: small instances for fast iteration / CI
FAST_INSTANCES = [
    "maze.domain/maze-example_40.json",
    "room.domain/room-example_150.json",
]

PRESETS: Dict[str, dict] = {
    # ---- main ablation: S0..S4 on three maps, 3 seeds -------------------
    "core": {
        "instances": CORE_INSTANCES,
        "schedulers": ["S0", "S1", "S2", "S3", "S4"],
        "seeds": [0, 1, 2],
        "horizon": 800,
        "sched_budget_ms": 100,
        "shortlist_k": 32,
    },
    # ---- smoke test for CI ---------------------------------------------
    "fast": {
        "instances": FAST_INSTANCES,
        "schedulers": ["S0", "S1", "S3"],
        "seeds": [0],
        "horizon": 300,
        "sched_budget_ms": 100,
        "shortlist_k": 32,
    },
    # ---- lambda sweep (congestion weight) -------------------------------
    "sweep_lam": {
        "instances": ["random.domain/random-example_400.json",
                      "room.domain/room-example_150.json"],
        "schedulers": ["S1"],
        "seeds": [0, 1],
        "horizon": 600,
        "sched_budget_ms": 100,
        "shortlist_k": 32,
        "param_grid": {"lam": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]},
    },
    # ---- gamma sweep (delay-risk weight) --------------------------------
    "sweep_gam": {
        "instances": ["random.domain/random-example_400.json",
                      "room.domain/room-example_150.json"],
        "schedulers": ["S2"],
        "seeds": [0, 1],
        "horizon": 600,
        "sched_budget_ms": 100,
        "shortlist_k": 32,
        "param_grid": {"gam": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]},
    },
    # ---- density sweep (the key "when does it help" axis) ---------------
    "density": {
        "instances": ["random.domain/random-example_400.json"],
        "schedulers": ["S0", "S1", "S3"],
        "seeds": [0, 1],
        "horizon": 600,
        "sched_budget_ms": 100,
        "shortlist_k": 32,
        "team_size_grid": [100, 200, 300, 400],
    },
    # ---- delay-severity sweep -------------------------------------------
    "delay_severity": {
        "instances": ["random.domain/random-example_400.json"],
        "schedulers": ["S0", "S2", "S3"],
        "seeds": [0, 1],
        "horizon": 600,
        "sched_budget_ms": 100,
        "shortlist_k": 32,
        "delay_grid": [0.0, 0.01, 0.03, 0.06],
    },
}


def run_one(job: dict) -> dict:
    """Execute a single (instance, scheduler, seed, params) run."""
    from lorrsim.instance import load_instance
    from lorrsim.instance import DelayConfig
    from lorrsim.runner import run_sim

    t0 = time.time()
    try:
        inst = load_instance(os.path.join(DATA, job["instance"]))
        if job.get("team_size") or job.get("delay_p") is not None:
            dc = inst.delay_config
            if job.get("delay_p") is not None:
                dc = dc.scaled(p_delay=job["delay_p"])
            inst = inst.with_params(
                team_size=job.get("team_size"), delay_config=dc
            )
        res = run_sim(
            inst,
            job["scheduler"],
            seed=job["seed"],
            horizon=job["horizon"],
            sched_budget_ms=job["sched_budget_ms"],
            scheduler_kwargs=job.get("scheduler_kwargs") or {},
        )
        row = res.as_row()
        row.update(
            {
                "run_id": job["run_id"],
                "domain": inst.domain,
                "wall_s": round(time.time() - t0, 2),
                "status": "ok",
                "lam": job.get("scheduler_kwargs", {}).get("lam", ""),
                "gam": job.get("scheduler_kwargs", {}).get("gam", ""),
            }
        )
        return {"job": job, "row": row}
    except Exception as exc:  # noqa: BLE001
        return {
            "job": job,
            "row": {
                "run_id": job["run_id"],
                "instance": os.path.basename(job["instance"]),
                "scheduler": job["scheduler"],
                "seed": job["seed"],
                "status": f"error: {exc}",
                "wall_s": round(time.time() - t0, 2),
                "traceback": traceback.format_exc(),
            },
        }


def build_jobs(cfg: dict, preset: str) -> List[dict]:
    jobs: List[dict] = []
    grid = cfg.get("param_grid") or {}
    keys = list(grid) if grid else [None]
    combos = []
    if grid:
        combos = [dict(zip(grid.keys(), vals)) for vals in itertools.product(*grid.values())]
    else:
        combos = [{}]

    team_sizes = cfg.get("team_size_grid") or [None]
    delay_ps = cfg.get("delay_grid") or [None]

    for inst, sched, seed, combo, ts, dp in itertools.product(
        cfg["instances"], cfg["schedulers"], cfg["seeds"], combos, team_sizes, delay_ps
    ):
        tag = []
        if combo:
            tag.append("_".join(f"{k}{v}" for k, v in combo.items()))
        if ts is not None:
            tag.append(f"n{ts}")
        if dp is not None:
            tag.append(f"p{dp}")
        name = os.path.splitext(os.path.basename(inst))[0]
        run_id = "--".join(
            [preset, name, sched, f"s{seed}"] + tag
        )
        jobs.append(
            {
                "run_id": run_id,
                "instance": inst,
                "scheduler": sched,
                "seed": seed,
                "horizon": cfg["horizon"],
                "sched_budget_ms": cfg["sched_budget_ms"],
                "scheduler_kwargs": {**combo, "shortlist_k": cfg.get("shortlist_k", 32)},
                "team_size": ts,
                "delay_p": dp,
                "preset": preset,
            }
        )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="core", choices=sorted(PRESETS))
    ap.add_argument("--config", default=None, help="YAML/JSON config overriding a preset")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.config:
        p = args.config
        with open(p, "r", encoding="utf-8") as fh:
            if p.endswith(".json"):
                cfg.update(json.load(fh))
            else:  # minimal flat YAML: key: value  (lists as [a, b])
                for line in fh:
                    line = line.split("#", 1)[0].strip()
                    if not line or ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    v = v.strip()
                    try:
                        cfg[k.strip()] = json.loads(v)
                    except json.JSONDecodeError:
                        cfg[k.strip()] = v

    jobs = build_jobs(cfg, args.preset)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"preset={args.preset}  jobs={len(jobs)}  workers={args.workers}", flush=True)

    os.makedirs(os.path.join(RESULTS, "raw"), exist_ok=True)
    out_csv = args.out or os.path.join(RESULTS, "raw", f"{args.preset}.csv")
    rows: List[dict] = []
    t0 = time.time()

    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            r = run_one(job)
            rows.append(r["row"])
            print(f"[{i}/{len(jobs)}] {job['run_id']}: {r['row'].get('num_task_finished')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                rows.append(r["row"])
                done += 1
                j = futs[fut]
                print(
                    f"[{done}/{len(jobs)}] {j['run_id']}: fin={r['row'].get('num_task_finished')} "
                    f"({r['row'].get('wall_s')}s)",
                    flush=True,
                )

    # ---- write tidy CSV ------------------------------------------------
    import csv

    keys = [
        "run_id", "instance", "domain", "scheduler", "seed", "status",
        "team_size", "density", "delay_p", "delay_event", "agent_counter",
        "ticks", "num_task_finished", "throughput", "mean_wait", "p95_wait",
        "max_wait", "load_cv", "load_gini", "total_delay_ticks",
        "mean_moves_per_robot", "sched_calls", "sched_ms_mean", "sched_ms_p95",
        "sched_ms_max", "sched_budget_hit", "assign_attempts", "lam", "gam", "wall_s",
    ]
    extra = sorted({k for r in rows for k in r} - set(keys))
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + extra, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- manifest ------------------------------------------------------
    man = {
        "preset": args.preset,
        "config": {k: v for k, v in cfg.items()},
        "n_jobs": len(jobs),
        "n_ok": sum(1 for r in rows if str(r.get("status")) == "ok"),
        "n_error": sum(1 for r in rows if str(r.get("status", "")).startswith("error")),
        "wall_s": round(time.time() - t0, 2),
        "python": sys.version.split()[0],
        "csv": os.path.relpath(out_csv, PROJECT),
    }
    man_path = os.path.splitext(out_csv)[0] + ".manifest.json"
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)

    print(f"\nwrote {out_csv}")
    print(json.dumps(man, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
