# Congestion- and Delay-Aware Scheduling for Lifelong MAPF (CS5446 Team 27)

Controlled study of task scheduling in the League of Robot Runners 2026 start-kit:
`S = d + λC + γR` (S0–S3 ablation, adaptive S4a, bandit S4b) against the start-kit default.

This repository contains the code only. The report, paper, instance data (`data/`), run results (`results/`)
and third-party sources (`third_party/`) are not included; `scripts/build_startkit.sh` clones the start-kit,
and `data/instances` must be populated with the official LoRR 2026 example domains before running.

## Layout
| Path | Content |
|---|---|
| `sched/` | C++ scheduler (`cda_scheduler.{h,cpp}`) + start-kit `TaskScheduler.cpp` dispatcher (`LORR_SCHED`, `LORR_LAMBDA`, `LORR_GAMMA`, ...) |
| `third_party/Start-Kit` | official LoRR Start-Kit v3.1.0 (commit 9bc75d9), patched by `scripts/build_startkit.sh` |
| `third_party/Benchmark-Archive` | LoRR 2023/2024 benchmark archive (reference) |
| `data/instances` | official 2026 example domains; `data/generated` seeded/density/delay variants; `data/instance_summary.json` |
| `scripts/` | `make_instances.py`, `expand.py`, `lorr_job.py` (one run), `ray_sweep.py` (Ray, resumable) |
| `experiments/configs` | `tune.json`, `main.json`, `density.json`, `scale.json` |
| `results/runs/<phase>/<job>` | summary, per-assignment features, PlanViz-loadable output, telemetry for all 2,326 runs |
| `results/tables`, `results/figures`, `results/key_numbers.json` | analysis outputs (`analysis/make_figures.py`) |
| `results/_invalid_v1_bugged_matcher` | archived first pass produced with two scheduler bugs (see its README); not used |
| `lorrsim/`, `results/raw`, `tests/test_lorrsim.py` | earlier pure-Python prototype simulator (kept for reference) |
| `tests/test_cda_scheduler.py` | integration tests against the real start-kit binary (skipped until the start-kit is built) |
| `scripts/sanitize_artifacts.py` | strips infrastructure identifiers from artefacts; `--check` gate |

## Reproduce
```bash
export PROJECT_ROOT="$(pwd)"                       # run everything from the project root
scripts/build_startkit.sh                         # clones the start-kit if absent (set https_proxy if your network needs one)
for c in tune main density scale; do python3 scripts/ray_sweep.py experiments/configs/$c.json; done
python3 analysis/make_figures.py
python3 -m pytest tests/
```
Each run reserves 2 CPUs: the planner is wall-clock bounded, so oversubscription changes results.
`ray_sweep.py` connects to `$RAY_ADDRESS` (default `auto`); use `--local N` for a local process pool instead.
Per-run scratch files go to `$LORR_SCRATCH` (default: the system temp dir).

## Artefact hygiene
Run artefacts record no host names, IP addresses or absolute paths.
`python3 scripts/sanitize_artifacts.py` rewrites older artefacts and `--check` fails if any such pattern remains.
Build outputs (`third_party/Start-Kit/build/`), Python caches and LaTeX auxiliary files are not distributed; `scripts/build_startkit.sh` and `latexmk` regenerate them.
