"""Run one LoRR simulation job and reduce its output to a compact summary.

``run_job(job)`` is pure with respect to the project tree: it writes into
``results/runs/<phase>/<job_id>/`` only

* ``summary.json``  -- every scalar metric used by the analysis
* ``assigns.csv.gz`` -- one row per scheduler assignment with predicted d/C/R and
  the realised completion time (feature-validity analysis)
* ``output.json.gz`` -- the start-kit output (``-c 2``: paths, events, tasks,
  delay intervals), loadable by PlanViz
* ``stats.json.gz`` -- scheduler telemetry (per-call timing and weights)
* ``run.log``       -- the simulator log

and finally ``DONE``.  A job directory with ``DONE`` is never recomputed.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
BIN = os.path.join(PROJECT, "third_party", "Start-Kit", "build", "lifelong")
RUNS = os.path.join(PROJECT, "results", "runs")


def job_dir(job):
    return os.path.join(RUNS, job["phase"], job["job_id"])


def _pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _gini(xs):
    xs = sorted(xs)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * s)


def summarise(out, stats, job, wall_s):
    team = out["teamSize"]
    horizon = out["makespan"]
    ctr = out.get("agentMaxCounter", 1)
    tasks = {t[0]: (t[1], len(t[2]) // 2) for t in out.get("tasks", [])}
    finish = {}
    per_agent = [0] * team
    for t, agent, task, seq in out.get("events", []):
        if task in tasks and seq == tasks[task][1]:
            finish[task] = t
            per_agent[agent] += 1
    service = [finish[k] - tasks[k][0] for k in finish]

    bins = [0] * (horizon // 100 + 1)
    for t in finish.values():
        bins[min(t // 100, len(bins) - 1)] += 1

    delay_ticks = sum(b - a for iv in out.get("delayIntervals", []) for a, b in iv)

    calls = stats.get("calls", []) if stats else []
    ms = [c[4] for c in calls]
    first_assign = {}
    for a in (stats.get("assigns", []) if stats else []):
        first_assign.setdefault(a[2], a)
    latency = [first_assign[k][0] - tasks[k][0] for k in first_assign if k in tasks]
    lam_t = [c[5] for c in calls if c[3] != -1]
    gam_t = [c[6] for c in calls if c[3] != -1]
    mean_load = sum(per_agent) / team if team else 0
    sd_load = math.sqrt(sum((x - mean_load) ** 2 for x in per_agent) / team) if team else 0

    s = {k: job[k] for k in ("job_id", "phase", "domain", "mult", "delay", "seed", "sched", "lam", "gam", "horizon")}
    s.update({
        "env": job.get("env", {}),
        "team": team,
        "ticks": horizon,
        "agent_counter": ctr,
        "tasks_finished": out["numTaskFinished"],
        "throughput_per_1k": 1000.0 * out["numTaskFinished"] / max(1, horizon),
        "tasks_per_agent": out["numTaskFinished"] / max(1, team),
        "planner_errors": out["numPlannerErrors"],
        "schedule_errors": out["numScheduleErrors"],
        "entry_timeouts": out["numEntryTimeouts"],
        "service_mean": sum(service) / len(service) if service else float("nan"),
        "service_p50": _pct(service, 0.5),
        "service_p95": _pct(service, 0.95),
        "assign_latency_mean": sum(latency) / len(latency) if latency else float("nan"),
        "assign_latency_p95": _pct(latency, 0.95),
        "load_cv": sd_load / mean_load if mean_load else float("nan"),
        "load_gini": _gini(per_agent),
        "idle_agents_frac": sum(1 for x in per_agent if x == 0) / team if team else float("nan"),
        "delay_ticks_total": delay_ticks,
        "delay_frac_ticks": delay_ticks / max(1, team * horizon),
        "throughput_bins100": bins,
        "sched_calls": len(ms),
        "sched_ms_mean": sum(ms) / len(ms) if ms else float("nan"),
        "sched_ms_p95": _pct(ms, 0.95),
        "sched_ms_max": max(ms) if ms else float("nan"),
        "sched_timeouts": sum(c[12] for c in calls) if calls else 0,
        "lam_t_mean": sum(lam_t) / len(lam_t) if lam_t else float("nan"),
        "gam_t_mean": sum(gam_t) / len(gam_t) if gam_t else float("nan"),
        "stall_frac_mean": (sum(c[7] for c in calls if c[3] != -1) / max(1, len(lam_t))) if lam_t else float("nan"),
        "delay_frac_obs_mean": (sum(c[8] for c in calls if c[3] != -1) / max(1, len(lam_t))) if lam_t else float("nan"),
        "p_hat": stats.get("p_hat") if stats else None,
        "dur_hat": stats.get("dur_hat") if stats else None,
        "true_p": None, "true_dur": None,
        "arm_pulls": stats.get("arm_pulls", []) if stats else [],
        "wall_s": wall_s,
    })

    rows = []
    for k, a in first_assign.items():
        t, agent, task, d, C, R = a
        rows.append((t, agent, task, d, C, R, finish.get(task, -1) - t if task in finish else -1))
    return s, rows


def run_job(job, force=False):
    jd = job_dir(job)
    if os.path.exists(os.path.join(jd, "DONE")) and not force:
        with open(os.path.join(jd, "summary.json")) as f:
            return json.load(f)
    from make_instances import make_instance  # noqa: E402

    inst = make_instance(job["domain"], job["mult"], job["delay"], job["seed"])
    os.makedirs(jd, exist_ok=True)
    # scratch lives outside the project tree; LORR_SCRATCH overrides the system temp dir
    scratch = os.path.join(os.environ.get("LORR_SCRATCH", tempfile.gettempdir()), "lorr_jobs", job["phase"],
                           job["job_id"])
    os.makedirs(scratch, exist_ok=True)
    out_json = os.path.join(scratch, "output.json")
    stats_json = os.path.join(scratch, "stats.json")
    log = os.path.join(scratch, "run.log")
    for p in (out_json, stats_json):
        if os.path.exists(p):
            os.remove(p)

    env = dict(os.environ)
    env.update({
        "LORR_SCHED": job["sched"],
        "LORR_LAMBDA": str(job["lam"]),
        "LORR_GAMMA": str(job["gam"]),
        "LORR_HORIZON": str(job["horizon"]),
        "LORR_STATS": stats_json,
        "LORR_SEED": str(job["seed"]),
    })
    for k, v in job.get("env", {}).items():
        env[k] = str(v)
    cmd = [BIN, "-i", inst, "-o", out_json, "-s", str(job["horizon"]), "-c", "2", "-d", "3",
           "-l", log]
    t0 = time.time()
    with open(os.path.join(scratch, "stdout.txt"), "w") as so:
        rc = subprocess.run(cmd, env=env, stdout=so, stderr=subprocess.STDOUT,
                            timeout=job["horizon"] * 0.6 + 900).returncode
    wall = time.time() - t0
    if rc != 0 or not os.path.exists(out_json):
        raise RuntimeError(f"{job['job_id']}: lifelong rc={rc}, see {scratch}")

    with open(out_json) as f:
        out = json.load(f)
    stats = {}
    if os.path.exists(stats_json):
        with open(stats_json) as f:
            stats = json.load(f)
    s, rows = summarise(out, stats, job, wall)
    with open(inst) as f:
        dc = json.load(f)["delayConfig"]
    s["true_p"] = dc["pDelay"]
    s["true_dur"] = (dc["minDelay"] + dc["maxDelay"]) / 2 if dc["pDelay"] > 0 else 0.0
    s["instance"] = os.path.relpath(inst, PROJECT)
    # recorded relative to the project root / job directory, so it is portable
    shown = {out_json: "output.json", log: "run.log"}
    s["cmd"] = " ".join(shown.get(c, os.path.relpath(c, PROJECT) if c.startswith(PROJECT) else c) for c in cmd)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["t_assign", "agent", "task", "d", "C", "R", "realised_ticks"])
    w.writerows(rows)
    with gzip.open(os.path.join(jd, "assigns.csv.gz"), "wt") as f:
        f.write(buf.getvalue())
    for src, dst in ((out_json, "output.json.gz"), (stats_json, "stats.json.gz")):
        if os.path.exists(src):
            with open(src, "rb") as fi, gzip.open(os.path.join(jd, dst), "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
    if os.path.exists(log):
        shutil.copy(log, os.path.join(jd, "run.log"))
    with open(os.path.join(jd, "summary.json"), "w") as f:
        json.dump(s, f)
    open(os.path.join(jd, "DONE"), "w").close()
    shutil.rmtree(scratch, ignore_errors=True)
    return s


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="run one job locally")
    ap.add_argument("--domain", default="room")
    ap.add_argument("--mult", type=float, default=1)
    ap.add_argument("--delay", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sched", default="S0")
    ap.add_argument("--lam", type=float, default=0)
    ap.add_argument("--gam", type=float, default=0)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--phase", default="debug")
    a = ap.parse_args()
    from expand import expand_config

    cfg = {"phase": a.phase, "horizon": {"default": a.horizon},
           "grids": [{"domains": [a.domain], "mults": [a.mult], "delays": [a.delay], "seeds": [a.seed],
                      "schedulers": [{"sched": a.sched, "lam": a.lam, "gam": a.gam}]}]}
    job = expand_config(cfg)[0]
    s = run_job(job, force=True)
    print(json.dumps({k: v for k, v in s.items() if k != "throughput_bins100"}, indent=1))
