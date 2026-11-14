"""Expand an experiment config into a flat, deterministic list of jobs.

A config is a JSON object::

    {
      "phase": "main",
      "horizon": {"default": 3000},
      "grids": [
        {"domains": [...], "mults": [1], "delays": ["none","base","high"],
         "seeds": [0,1,2,3,4],
         "schedulers": [{"sched": "S0"}, {"sched": "S3", "lam": 1, "gam": 1}, ...]}
      ]
    }

Each job is a dict; ``job_id`` is a stable string so finished jobs can be skipped
when a sweep is resumed.
"""

from __future__ import annotations

import itertools
import json


def sched_tag(s):
    tag = s["sched"]
    if "lam" in s or "gam" in s:
        tag += f"-l{s.get('lam', 0):g}-g{s.get('gam', 0):g}"
    for k in sorted(s.get("env", {})):
        tag += f"-{k}{s['env'][k]}"
    return tag


def expand_config(path_or_obj):
    cfg = path_or_obj
    if isinstance(path_or_obj, str):
        with open(path_or_obj) as f:
            cfg = json.load(f)
    hz = cfg.get("horizon", {})
    jobs, seen = [], set()
    for g in cfg["grids"]:
        for dom, mult, delay, seed, s in itertools.product(
                g["domains"], g.get("mults", [1]), g.get("delays", ["base"]), g["seeds"], g["schedulers"]):
            job = {
                "phase": cfg["phase"],
                "domain": dom, "mult": mult, "delay": delay, "seed": seed,
                "sched": s["sched"], "lam": s.get("lam", 0.0), "gam": s.get("gam", 0.0),
                "env": s.get("env", {}),
                "horizon": g.get("horizon", hz.get(dom, hz.get("default", 3000))),
            }
            job["job_id"] = f"{dom}-m{mult:g}-{delay}-s{seed}--{sched_tag(s)}"
            if job["job_id"] not in seen:
                seen.add(job["job_id"])
                jobs.append(job)
    return jobs
