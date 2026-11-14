"""Generate LoRR instances for the density / delay / seed sweeps.

Every generated instance keeps the *map* and the *task-location distribution* of
one official LoRR 2026 example instance and varies only what the study varies:

* fleet size   -- ``teamSize = round(mult * base_team)`` with fresh start cells
                  drawn from the largest connected component of the map;
* task stream  -- errand counts and errand cells are resampled (with
                  replacement) from the official task file, so warehouse tasks
                  still go to stations/shelves and city tasks to the same streets;
* delays       -- ``none`` (pDelay = 0), ``base`` (official delayConfig) or
                  ``high`` (4x event rate, 2x duration range, delay load
                  pDelay*E[dur] capped at 0.5);
* seed         -- drives both the sampling above and ``delayConfig.seed``.

Agent and task files depend only on (domain, mult, seed), so every scheduler and
delay level sees exactly the same robots and tasks (common random numbers).

Usage::

    python3 scripts/make_instances.py --config experiments/configs/main.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
from collections import deque

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# official example instances: data/instances if present, else the copy shipped with the start-kit
BASE = os.path.join(PROJECT, "data", "instances")
if not os.path.isdir(BASE):
    BASE = os.path.join(PROJECT, "third_party", "Start-Kit", "example_problems")
GEN = os.path.join(PROJECT, "data", "generated")

# domain -> official example instance (relative to data/instances)
DOMAINS = {
    "maze": "maze.domain/maze-example_40.json",
    "room": "room.domain/room-example_150.json",
    "random": "random.domain/random-example_400.json",
    "city": "city.domain/bos-example_600.json",
    "warehouse": "warehouse.domain/fulfill-example_2500.json",
    "game": "game.domain/orz-example_1800.json",
    "iron": "iron_harvest.domain/iron-example_10000.json",
}

DELAY_LEVELS = ("none", "base", "high")
HIGH_RHO_CAP = 0.5


def read_map(path):
    with open(path) as f:
        lines = f.read().splitlines()
    h = int(lines[1].split()[1])
    w = int(lines[2].split()[1])
    grid = lines[4:4 + h]
    free = bytearray(h * w)
    for r, row in enumerate(grid):
        for c, ch in enumerate(row[:w]):
            if ch not in "@T":
                free[r * w + c] = 1
    return h, w, free


def largest_component(h, w, free):
    seen = bytearray(h * w)
    best = []
    for s in range(h * w):
        if not free[s] or seen[s]:
            continue
        comp, q = [], deque([s])
        seen[s] = 1
        while q:
            u = q.popleft()
            comp.append(u)
            r, c = divmod(u, w)
            for v, ok in ((u - w, r > 0), (u + w, r < h - 1), (u - 1, c > 0), (u + 1, c < w - 1)):
                if ok and free[v] and not seen[v]:
                    seen[v] = 1
                    q.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def read_tasks(path):
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l and not l.startswith("#")]
    n = int(lines[0])
    return [[int(x) for x in l.split(",")] for l in lines[1:1 + n]]


def base_info(domain):
    rel = DOMAINS[domain]
    jpath = os.path.join(BASE, rel)
    ddir = os.path.dirname(jpath)
    with open(jpath) as f:
        cfg = json.load(f)
    return jpath, ddir, cfg


_cache = {}


def domain_geometry(domain):
    if domain not in _cache:
        _, ddir, cfg = base_info(domain)
        h, w, free = read_map(os.path.join(ddir, cfg["mapFile"]))
        comp = largest_component(h, w, free)
        tasks = read_tasks(os.path.join(ddir, cfg["taskFile"]))
        compset = set(comp)
        locs = [x for t in tasks for x in t if x in compset]
        lens = [len(t) for t in tasks]
        _cache[domain] = (h, w, comp, locs, lens, cfg)
    return _cache[domain]


def instance_name(domain, mult, delay, seed):
    return f"{domain}-m{mult:g}-{delay}-s{seed}"


def make_instance(domain, mult, delay, seed, out_root=GEN):
    """Write (if absent) and return the path of one generated instance JSON."""
    h, w, comp, locs, lens, cfg = domain_geometry(domain)
    team = max(1, int(round(mult * cfg["teamSize"])))
    team = min(team, len(comp) // 2)
    ddir = os.path.join(out_root, domain)
    os.makedirs(os.path.join(ddir, "agents"), exist_ok=True)
    os.makedirs(os.path.join(ddir, "tasks"), exist_ok=True)
    maps_link = os.path.join(ddir, "maps")
    if not os.path.exists(maps_link):
        src = os.path.join(BASE, os.path.dirname(DOMAINS[domain]), "maps")
        os.symlink(os.path.relpath(src, ddir), maps_link)

    stem = f"{domain}-m{mult:g}-s{seed}"
    afile = os.path.join(ddir, "agents", stem + ".agents")
    tfile = os.path.join(ddir, "tasks", stem + ".tasks")
    if not (os.path.exists(afile) and os.path.exists(tfile)):
        rng = random.Random(f"{domain}|{mult}|{seed}")
        starts = rng.sample(comp, team)
        n_tasks = max(len(lens), 10 * team)
        tasks = []
        for _ in range(n_tasks):
            k = rng.choice(lens)
            errands = [rng.choice(locs)]
            while len(errands) < k:
                x = rng.choice(locs)
                if x != errands[-1]:
                    errands.append(x)
            tasks.append(errands)
        for path, body in ((afile, [str(s) for s in starts]),
                           (tfile, [",".join(map(str, t)) for t in tasks])):
            tmp = path + f".tmp{os.getpid()}-{threading.get_ident()}"
            with open(tmp, "w") as f:
                f.write("# version for LoRR 2026\n%d\n%s\n" % (len(body), "\n".join(body)))
            os.replace(tmp, path)

    dc = dict(cfg["delayConfig"])
    dc["seed"] = int(seed)
    if delay == "none":
        dc["pDelay"] = 0.0
    elif delay == "high":
        # 8x the delay load rho = pDelay * E[duration] (4x rate, 2x durations), with rho
        # capped at HIGH_RHO_CAP: room/city already sit at rho~0.2, and 8x that leaves
        # every robot permanently delayed (zero throughput for any scheduler).
        dc["minDelay"] = 2 * dc["minDelay"]
        dc["maxDelay"] = 2 * dc["maxDelay"]
        mean_dur = (dc["minDelay"] + dc["maxDelay"]) / 2
        dc["pDelay"] = min(4 * dc["pDelay"], HIGH_RHO_CAP / mean_dur)
    out = {
        "mapFile": cfg["mapFile"],
        "agentFile": os.path.relpath(afile, ddir),
        "teamSize": team,
        "taskFile": os.path.relpath(tfile, ddir),
        "numTasksReveal": cfg["numTasksReveal"],
        "version": cfg["version"],
        "agentCounter": cfg["agentCounter"],
        "agentSize": cfg["agentSize"],
        "delayConfig": dc,
    }
    jpath = os.path.join(ddir, instance_name(domain, mult, delay, seed) + ".json")
    if not os.path.exists(jpath):
        tmp = jpath + f".tmp{os.getpid()}-{threading.get_ident()}"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, jpath)
    return jpath


def describe():
    rows = []
    for d in DOMAINS:
        h, w, comp, locs, lens, cfg = domain_geometry(d)
        rows.append({
            "domain": d, "map": cfg["mapFile"], "rows": h, "cols": w,
            "free_cells_lcc": len(comp), "base_team": cfg["teamSize"],
            "density": round(cfg["teamSize"] / len(comp), 4),
            "distinct_errand_cells": len(set(locs)),
            "errands_per_task_mean": round(sum(lens) / len(lens), 2),
            "tasks_in_file": len(lens), "numTasksReveal": cfg["numTasksReveal"],
            "agentCounter": cfg["agentCounter"], "delayConfig": cfg["delayConfig"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="experiment config JSON (uses its 'grid' entries)")
    ap.add_argument("--describe", action="store_true", help="write data/instance_summary.json")
    a = ap.parse_args()
    if a.describe:
        rows = describe()
        with open(os.path.join(PROJECT, "data", "instance_summary.json"), "w") as f:
            json.dump(rows, f, indent=2)
        for r in rows:
            print(r)
    if a.config:
        from expand import expand_config  # noqa: E402  (scripts/expand.py)
        n = 0
        for job in expand_config(a.config):
            make_instance(job["domain"], job["mult"], job["delay"], job["seed"])
            n += 1
        print(f"instances ready for {n} jobs")


if __name__ == "__main__":
    main()
