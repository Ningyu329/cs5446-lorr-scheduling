"""Run an experiment config on the Ray cluster (resumable).

    python3 scripts/ray_sweep.py experiments/configs/tune.json [--max-concurrent 600] [--local N]

Each job reserves ``--cpus`` (default 2) logical CPUs, because the start-kit runs
the planner thread and the simulator thread concurrently and its planner is
wall-clock bounded: oversubscribing cores would change results, not just speed.
Finished jobs (``DONE`` marker) are skipped, so a sweep can be re-launched
after interruption.  A sweep-level manifest with job counts and timing is
written to ``results/runs/<phase>/_manifest_<t>.json`` (no host or node
identifiers are recorded).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

from expand import expand_config  # noqa: E402
from lorr_job import job_dir, run_job  # noqa: E402
from make_instances import make_instance  # noqa: E402

BIG = {"warehouse": 4, "game": 6, "iron": 24, "city": 3}  # GB reserved per job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-concurrent", type=int, default=600)
    ap.add_argument("--cpus", type=float, default=2)
    ap.add_argument("--local", type=int, default=0, help="run with a local process pool instead of Ray")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    jobs = expand_config(a.config)
    todo = [j for j in jobs if not os.path.exists(os.path.join(job_dir(j), "DONE"))]
    phase = jobs[0]["phase"]
    print(f"{phase}: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    # instances are generated once on the driver so workers never race on files
    for j in todo:
        make_instance(j["domain"], j["mult"], j["delay"], j["seed"])
    if a.dry or not todo:
        return

    t0 = time.time()
    failed = []
    if a.local:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(a.local) as ex:
            futs = {ex.submit(run_job, j): j for j in todo}
            for k, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    failed.append((futs[f]["job_id"], repr(e)))
                print(f"[{k}/{len(todo)}] {futs[f]['job_id']}", flush=True)
    else:
        import ray

        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False, logging_level="ERROR")

        @ray.remote(max_retries=1)
        def remote_job(job):
            import sys as _s

            _s.path.insert(0, os.path.join(PROJECT, "scripts"))
            from lorr_job import run_job as rj

            return rj(job)

        pending, queue = {}, list(todo)
        # longest jobs first keeps the tail short
        queue.sort(key=lambda j: (-j["horizon"] * (1 + BIG.get(j["domain"], 1)), j["job_id"]))
        done = 0
        while queue or pending:
            while queue and len(pending) < a.max_concurrent:
                j = queue.pop(0)
                mem = BIG.get(j["domain"], 1) * (j["mult"] if j["mult"] > 1 else 1)
                ref = remote_job.options(num_cpus=a.cpus, memory=int(mem * 2**30)).remote(j)
                pending[ref] = j
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=60)
            for r in ready:
                j = pending.pop(r)
                done += 1
                try:
                    s = ray.get(r)
                    msg = f"tasks={s['tasks_finished']} wall={s['wall_s']:.0f}s"
                except Exception as e:  # noqa: BLE001
                    failed.append((j["job_id"], repr(e)[:400]))
                    msg = "FAILED"
                print(f"[{done}/{len(todo)} {time.time() - t0:.0f}s] {j['job_id']} {msg}", flush=True)

    man = {
        "phase": phase, "config": os.path.relpath(os.path.abspath(a.config), PROJECT),
        "n_jobs": len(jobs), "n_run": len(todo), "n_failed": len(failed), "failed": failed,
        "wall_s": round(time.time() - t0, 1), "cpus_per_job": a.cpus,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.join(PROJECT, "results", "runs", phase), exist_ok=True)
    mp = os.path.join(PROJECT, "results", "runs", phase, f"_manifest_{int(t0)}.json")
    with open(mp, "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "failed"}, indent=1))
    if failed:
        print("FAILED:", *failed[:20], sep="\n  ")


if __name__ == "__main__":
    main()
