"""Regressions for first-run setup and honest quick-start success reporting."""

import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_instances
import quick_summary
import ray_sweep


def write_run(tmp_path, sched, count=10, done=True, errors=0):
    directory = tmp_path / "results" / "runs" / "quick" / sched
    directory.mkdir(parents=True)
    data = dict(domain="maze", mult=1, delay="base", seed=0, sched=sched,
                horizon=100, tasks_finished=count, planner_errors=errors,
                schedule_errors=0, entry_timeouts=0)
    (directory / "summary.json").write_text(json.dumps(data))
    if done:
        (directory / "DONE").touch()
    return directory


def test_summary_ignores_unfinished_runs(tmp_path, monkeypatch, capsys):
    write_run(tmp_path, "S0")
    write_run(tmp_path, "S3", done=False)
    monkeypatch.setattr(quick_summary, "PROJECT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["quick_summary.py"])
    quick_summary.main()
    text = capsys.readouterr().out
    assert "1 runs" in text and "S3" not in text


def test_summary_requires_every_config_job(tmp_path, monkeypatch):
    config = tmp_path / "quick.json"
    config.write_text(json.dumps({"phase": "quick", "grids": [{"domains": ["maze"],
        "seeds": [0], "schedulers": [{"sched": "S0"}, {"sched": "S3"}]}]}))
    monkeypatch.setattr(quick_summary, "PROJECT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["quick_summary.py", "--config", str(config)])
    with pytest.raises(SystemExit, match="incomplete experiment: 2"):
        quick_summary.main()


def test_summary_handles_zero_baseline(tmp_path, monkeypatch, capsys):
    write_run(tmp_path, "S0", count=0)
    write_run(tmp_path, "S3", count=0)
    monkeypatch.setattr(quick_summary, "PROJECT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["quick_summary.py"])
    quick_summary.main()
    assert "percentage change undefined" in capsys.readouterr().out


def test_summary_errors_are_not_success(tmp_path, monkeypatch):
    write_run(tmp_path, "S0", errors=1)
    monkeypatch.setattr(quick_summary, "PROJECT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["quick_summary.py"])
    with pytest.raises(SystemExit) as exc:
        quick_summary.main()
    assert exc.value.code == 1


def test_summary_rejects_different_horizons(tmp_path, monkeypatch):
    write_run(tmp_path, "S0")
    directory = write_run(tmp_path, "S3")
    data = json.loads((directory / "summary.json").read_text())
    data["horizon"] = 200
    (directory / "summary.json").write_text(json.dumps(data))
    monkeypatch.setattr(quick_summary, "PROJECT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["quick_summary.py"])
    with pytest.raises(SystemExit, match="different horizons"):
        quick_summary.main()


def test_fresh_concurrent_instances_share_map_link(tmp_path, monkeypatch):
    """Force both first-time workers past the existence check before symlink."""
    base = tmp_path / "base"
    (base / "maze.domain" / "maps").mkdir(parents=True)
    monkeypatch.setattr(make_instances, "BASE", str(base))
    cfg = dict(teamSize=2, numTasksReveal=1.5, version="2026 LoRR", agentCounter=3,
               agentSize=1.0, mapFile="maps/test.map",
               delayConfig=dict(seed=0, pDelay=0.01, minDelay=2, maxDelay=4))
    monkeypatch.setattr(make_instances, "domain_geometry",
                        lambda domain: (2, 3, list(range(6)), list(range(6)), [2], cfg))
    barrier = threading.Barrier(2)
    original_symlink = os.symlink

    def simultaneous_symlink(src, dst):
        barrier.wait(timeout=5)
        original_symlink(src, dst)

    monkeypatch.setattr(os, "symlink", simultaneous_symlink)
    generated = str(tmp_path / "generated")
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        paths = list(pool.map(lambda delay: make_instances.make_instance("maze", 1, delay, 7, generated),
                              ["base", "none"]))
    assert len(paths) == 2
    configs = [json.loads(Path(p).read_text()) for p in paths]
    assert configs[0]["agentFile"] == configs[1]["agentFile"]
    assert configs[0]["taskFile"] == configs[1]["taskFile"]
    assert configs[1]["delayConfig"]["pDelay"] == 0
    assert (Path(generated) / "maze" / "maps").is_dir()


def test_sweep_reports_failed_job_as_failure(tmp_path, monkeypatch):
    job = dict(phase="quick", job_id="failed-job", domain="maze", mult=1, delay="base", seed=0)
    monkeypatch.setattr(ray_sweep, "PROJECT", str(tmp_path))
    monkeypatch.setattr(ray_sweep, "expand_config", lambda config: [job])
    monkeypatch.setattr(ray_sweep, "job_dir", lambda job: str(tmp_path / "missing"))
    monkeypatch.setattr(ray_sweep, "make_instance", lambda *args: None)

    def fail(job):
        raise RuntimeError("simulator failed")

    monkeypatch.setattr(ray_sweep, "run_job", fail)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    monkeypatch.setattr(sys, "argv", ["ray_sweep.py", "quick.json", "--local", "1"])
    with pytest.raises(SystemExit) as exc:
        ray_sweep.main()
    assert exc.value.code == 1
    manifests = list((tmp_path / "results" / "runs" / "quick").glob("_manifest_*.json"))
    assert json.loads(manifests[0].read_text())["n_failed"] == 1


def test_failed_build_does_not_accept_stale_binary(tmp_path):
    """A failing compile must fail even when a previous executable exists."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts" / "build_startkit.sh", scripts)
    (tmp_path / "sched").mkdir()
    for name in ("TaskScheduler.cpp", "cda_scheduler.cpp", "cda_scheduler.h"):
        (tmp_path / "sched" / name).write_text("test fixture")
    kit = tmp_path / "third_party" / "Start-Kit"
    for name in ("src", "inc", "build"):
        (kit / name).mkdir(parents=True)
    (kit / "build" / "lifelong").touch()
    tools = tmp_path / "bin"
    tools.mkdir()
    stubs = {
        "git": "echo 9bc75d9dd9883de049294ab2981bf325bbe19d49\n",
        "python3": "exit 1\n",
        "cmake": 'if [ "$1" = "--build" ]; then echo "intentional compile failure"; exit 23; fi\nexit 0\n',
    }
    for name, body in stubs.items():
        tool = tools / name
        tool.write_text("#!/bin/sh\n" + body)
        tool.chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"], PYTHON="python3")
    result = subprocess.run(["bash", str(scripts / "build_startkit.sh")], env=env,
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "intentional compile failure" in result.stdout
