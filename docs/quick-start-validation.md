# Quick-start validation — 15 November 2026

This check covers first-run setup, the small demonstration, and the existing
tests. It does not rerun the full study or change the C++ scheduling policy.
The starting repository revision was `706fdea`; the simulator was built from
the required Start-Kit commit `9bc75d9dd9883de049294ab2981bf325bbe19d49`.

## Environment and build

Validated on Apple M2, macOS 15.7.4, AppleClang 17, CMake 3.30.4, Python 3.12.2,
Boost 1.82.0, and pybind11 3.1.0. Python packages were installed into a project
virtual environment. This machine already had Boost in an Anaconda prefix;
the build used `CMAKE_PREFIX_PATH=/opt/anaconda3` and
`PYTHON="$PWD/.venv/bin/python" JOBS=4 scripts/build_startkit.sh`.

The real `lifelong` executable compiled successfully. Ubuntu package commands
are documented in the README but were not executed on this Mac.

## Automated tests

Command:

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

Result: **43 passed in 40.95 seconds; no skipped tests.** These comprise
19 Python-prototype checks, 16 real-simulator integration checks, and eight
new first-run/failure-reporting regressions. All seven scheduler modes are
covered by the existing integration tests.

The first fresh integration run exposed a `FileExistsError` when several
workers created the shared map symlink simultaneously. The fix accepts an
identical link created by another worker; a regression test forces that race
on a new temporary directory.

## Small experiment

Commands:

```bash
.venv/bin/python scripts/ray_sweep.py experiments/configs/quick.json --local 3
.venv/bin/python scripts/quick_summary.py --config experiments/configs/quick.json
```

The six configured runs finished in **177.8 seconds**, with **zero failed jobs**.
Both seeds used 400 robots on the official random map, base delays, and a
1,000-tick horizon.

| Instance | DEFAULT | S0 | S3 |
|---|---:|---:|---:|
| random m1 base s0 | 686 | 741 | 736 |
| random m1 base s1 | 652 | 694 | 692 |

Mean paired task-count changes relative to S0 were −6.74% for DEFAULT and
−0.48% for S3 (two pairs each). Planner errors, schedule errors, and entry
timeouts were **all zero**. These are smoke-test observations, not a new
statistical conclusion; another machine can produce different counts because
the planner has a wall-clock budget.

Rerunning the same sweep printed `quick: 6 jobs, 0 to run`, confirming that
completed jobs are reused.

## Changes made during validation

* The build script now propagates compiler failures, checks the pinned
  simulator revision, selects the requested Python environment, and permits
  configurable build parallelism and extra CMake arguments.
* Instance generation handles simultaneous creation of the same map link.
* A failed experiment sweep returns a failing exit status.
* The summary ignores unfinished jobs, can require all jobs in a supplied
  config, rejects ambiguous or mixed-horizon comparisons, handles a zero-task
  S0 baseline, and returns failure when simulator health counters are nonzero.
* README instructions now include a virtual environment, minimal dependencies,
  success criteria, troubleshooting, and the distinction between a quick
  demonstration and the full statistical study. The headline percentage uses
  the same reference denominator as its existing results table.

Generated maps, simulator sources, binaries, and run outputs remain ignored by
Git. No report or paper draft is included.
