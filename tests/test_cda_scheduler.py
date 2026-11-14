"""Integration tests for the CDA scheduler inside the official start-kit.

These run the real ``lifelong`` binary for a few hundred ticks, so they take
~40 s (all modes run in parallel).  Run with ``python3 -m pytest tests/test_cda_scheduler.py``.
They pin the two bugs found during the study (schedule errors from overwriting
the shared schedule, and the unsorted fallback) plus the core design invariants.
"""

import gzip
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
BIN = os.path.join(PROJECT, "third_party", "Start-Kit", "build", "lifelong")
MODES = ["DEFAULT", "S0", "S1", "S2", "S3", "ADAPT", "BANDIT"]
HORIZON = 400

pytestmark = pytest.mark.skipif(not os.path.exists(BIN), reason="run scripts/build_startkit.sh first")


def _run(mode, delay, tmp):
    from make_instances import make_instance

    inst = make_instance("room", 1, delay, 7)
    out, st = os.path.join(tmp, f"{mode}-{delay}.json"), os.path.join(tmp, f"{mode}-{delay}.stats.json")
    env = dict(os.environ, LORR_SCHED=mode, LORR_LAMBDA="1", LORR_GAMMA="2", LORR_HORIZON=str(HORIZON),
               LORR_STATS=st)
    subprocess.run([BIN, "-i", inst, "-o", out, "-s", str(HORIZON), "-c", "1", "-d", "3"], env=env,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    with open(out) as f:
        o = json.load(f)
    with open(st) as f:
        s = json.load(f)
    return o, s


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("cda"))
    jobs = [(m, "base") for m in MODES] + [("S2", "none"), ("S3", "none")]
    with ThreadPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(lambda j: _run(j[0], j[1], tmp), jobs))
    return dict(zip(jobs, res))


@pytest.mark.parametrize("mode", MODES)
def test_no_schedule_or_planner_errors(runs, mode):
    o, _ = runs[(mode, "base")]
    assert o["numScheduleErrors"] == 0, o["scheduleErrors"][:5]
    assert o["numPlannerErrors"] == 0
    assert o["numTaskFinished"] > 0


@pytest.mark.parametrize("mode", ["S0", "S1", "S2", "S3"])
def test_every_free_agent_gets_a_task_at_t0(runs, mode):
    o, s = runs[(mode, "base")]
    first = s["calls"][0]  # [t, n_free_agents, n_free_tasks, n_assigned, ...]
    assert first[3] == min(first[1], first[2])


def test_no_assignment_is_far_beyond_the_nearest(runs):
    """Fallback must pick the nearest unused task, not an arbitrary one."""
    _, s = runs[("S0", "base")]
    t0 = [a for a in s["assigns"] if a[0] == 0]
    ds = sorted(a[3] for a in t0)
    # with 1.5 tasks per robot on a 64x64 map, the worst t=0 match stays local
    assert ds[-1] < 4 * ds[len(ds) // 2] + 64


def test_s0_and_s1_have_no_delay_term(runs):
    for mode in ("S0", "S1"):
        _, s = runs[(mode, "base")]
        assert all(c[6] == 0 for c in s["calls"])  # gamma_t


def test_risk_vanishes_without_delays(runs):
    _, s = runs[("S2", "none")]
    assert s["onsets"] == 0
    assert all(a[5] == 0 for a in s["assigns"])  # R column


def test_delay_load_estimate(runs):
    """p_hat * dur_hat recovers the hidden delay load pDelay * E[dur] (room: 0.01 * 20)."""
    _, s = runs[("S3", "base")]
    load = s["p_hat"] * s["dur_hat"]
    assert 0.12 < load < 0.35


def test_bandit_pulls_every_arm_first(runs):
    _, s = runs[("BANDIT", "base")]
    assert s["arm_pulls"][:6] == [0, 1, 2, 3, 4, 5][: len(s["arm_pulls"][:6])]
