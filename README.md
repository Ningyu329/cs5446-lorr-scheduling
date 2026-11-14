# Congestion- and Delay-Aware Scheduling for Lifelong MAPF

**CS5446 Reinforcement Learning and Sequential Decision Making · AY2026/27 Sem 1 · Team 27**
Competition project: [League of Robot Runners (LoRR) 2026](https://github.com/MAPF-Competition/Start-Kit), Task Scheduling Track.

We built a task scheduler for the official LoRR 2026 C++ start-kit that scores each robot–task pair with

```
S = d + λC + γR        d = distance,  C = congestion exposure,  R = delay risk
```

and evaluated it, together with two adaptive extensions, in 2,326 full simulations.
**Main result:** the *matching rule* matters far more than the extra score terms.
A global greedy matcher on distance alone finishes **5.9% ± 1.3% more tasks** than the start-kit default
(156 of 180 paired wins), while adding congestion and delay terms changes throughput by only −0.4% to +0.5%, which is not significant.

---

## Problem background

* **Lifelong multi-agent path finding.** Hundreds to 10,000 robots share a grid map (warehouse, city, game maps…).
  Tasks keep arriving; each task is a chain of errand cells that a robot must visit in order.
* **Only the scheduler is ours.** In the Task Scheduling Track the path planner and executor are fixed.
  At least once per simulated second the scheduler sees the robots (positions, headings, `inDelay` flags) and the
  task pool, and assigns free robots to free, unopened tasks. The objective is tasks finished within the horizon.
* **Hidden execution delays.** LoRR 2026 randomly stops robots for several ticks. Only the delay *family* is public,
  so the delay rate and duration must be estimated online.
* **Research question.** When, and by how much, do congestion- and delay-aware assignments improve throughput over
  distance-only assignment?

We model this as a partially observable sequential decision problem and study myopic score-based policies
`π_t = argmin_π Σ_i S_θ(i, π(i) | o_t)` with interpretable weights `θ = (λ, γ)` that are fixed, state-adaptive, or learned online.

## Method

| Scheduler | Score | Notes |
|---|---|---|
| `DEFAULT` | d | start-kit baseline: robots in hash-set order each take their nearest task |
| `S0` | d | **our global greedy matcher** (shared by S0–S4b): each robot proposes its 16 nearest tasks, all proposals are sorted, best pairs are committed first |
| `S1` | d + λC | congestion: route length × (robot density + 2 × stalled-robot density) + competitors heading to the same blocks |
| `S2` | d + γR | delay risk: own remaining delay + delay exposure in corridors + delayed robots on the route; delay rate/duration estimated online from `inDelay` flags |
| `S3` | d + λC + γR | full score, (λ\*, γ\*) = (2, 1) tuned on separate seeds |
| `ADAPT` (S4a) | d + λₜC + γₜR | weights scale with the observed share of stalled / delayed robots |
| `BANDIT` (S4b) | d + λₜC + γₜR | discounted-UCB bandit over 6 weight multipliers, one arm per 100-tick window |

All terms are measured in action steps, so λ and γ are dimensionless.

## Key findings

| vs S0 (180 paired runs: 6 maps × 3 delay levels × 10 held-out seeds) | Δ tasks finished (95% CI) | Holm-adjusted p |
|---|---|---|
| Start-kit default | −5.88% ± 1.31 | 1.9 × 10⁻¹⁸ |
| S1  d + λC | +0.45% ± 1.26 | 1.00 |
| S2  d + γR | −0.06% ± 1.25 | 1.00 |
| S3  d + λC + γR | +0.17% ± 1.28 | 1.00 |
| S4a adaptive | −0.41% ± 1.14 | 0.35 |
| S4b bandit | +0.24% ± 1.18 | 1.00 |

* On the 10,000-robot iron-harvest map, S0 finishes 2.2× as many tasks as the default (837 vs 382).
* The only hint of a congestion effect is in the warehouse at 1.5× fleet size, at the onset of congestion collapse:
  +4.4% ± 2.6% (p = 0.004), which is not significant after Holm correction.
* Why the terms do not help: LoRR delay onsets are i.i.d. over robots and ticks, so delay history does not predict future delays;
  and waiting concentrates at junctions that every route must cross. Given d, the C and R terms add at most 0.018 to the
  R² of a completion-time model.

---

## Requirements

* Linux or macOS, a C++17 compiler, CMake ≥ 3.16, Boost (program_options, system, filesystem, log), pybind11
* Python ≥ 3.9 (running experiments needs only the standard library)

```bash
# Ubuntu / Debian
sudo apt install build-essential cmake libboost-all-dev pybind11-dev python3 python3-pip
# macOS (Homebrew)
brew install cmake boost pybind11

# optional: analysis/figures, tests, Ray cluster sweeps
pip install -r requirements.txt
```

## Quick start (about 5 minutes)

```bash
git clone https://github.com/Ningyu329/cs5446-lorr-scheduling.git
cd cs5446-lorr-scheduling

scripts/build_startkit.sh          # clones the official start-kit (pinned commit), installs our scheduler, builds ./lifelong (~1 min)
python3 scripts/ray_sweep.py experiments/configs/quick.json --local 3   # 6 runs, ~3 min with 3 parallel workers
python3 scripts/quick_summary.py   # paired comparison table
```

`quick.json` runs the 400-robot `random` map with the official delays for 1,000 ticks on 2 seeds, comparing
`DEFAULT`, `S0` and `S3`. Expected output (counts vary slightly between machines, because the planner is wall-clock bounded):

```
Tasks finished (quick, 1000 ticks)

instance                     DEFAULT        S0        S3
--------------------------------------------------------
random m1 base s0                674       744       736
random m1 base s1                648       706       696

Mean paired change in tasks finished vs S0
  DEFAULT   -8.81%   (2 pairs)
  S3        -1.25%   (2 pairs)

6 runs, planner/schedule errors + timeouts: 0
```

DEFAULT should come out clearly below S0 (finding 1), and S3 should be within a few percent of S0 (finding 2).
Two seeds are far too few for inference: the full study uses 10 held-out seeds per cell.

**A single run** (any map, scheduler and weights):

```bash
python3 scripts/lorr_job.py --domain maze --delay base --sched S3 --lam 2 --gam 1 --horizon 500
```

Maps: `maze` (40 robots), `room` (150), `random` (400), `city` (600), `warehouse` (2,500), `game` (1,800), `iron` (10,000).
Delay levels: `none`, `base` (official configuration), `high` (8× delay load, capped).

**Tests** (35 tests, ~40 s; the scheduler integration tests run the real binary):

```bash
python3 -m pytest tests/
```

## Full reproduction

The simulator advances one tick per 100 ms of wall-clock time, so a 3,000-tick run takes about 5 minutes regardless of CPU speed,
and each run should get **2 dedicated CPU cores**: the planner is time-bounded, so oversubscription changes results, not just speed.

| Config | Runs | What |
|---|---|---|
| `experiments/configs/quick.json` | 6 | smoke test above |
| `experiments/configs/tune.json` | 400 | λ, γ sweeps on tuning seeds 100–104 |
| `experiments/configs/main.json` | 1,320 | 6 maps × 3 delay levels × 7 schedulers × held-out seeds 0–9, plus noise-floor repeats |
| `experiments/configs/density.json` | 600 | fleet size 0.5×, 1.5×, 2× on 4 maps |
| `experiments/configs/scale.json` | 6 | iron harvest, 10,000 robots |

```bash
python3 scripts/ray_sweep.py experiments/configs/main.json             # on a Ray cluster ($RAY_ADDRESS, default "auto")
python3 scripts/ray_sweep.py experiments/configs/main.json --local 8   # or a local process pool (8 runs ≈ 16 cores)
python3 analysis/make_figures.py                                        # tables and figures (needs requirements.txt)
```

The full study took about 349 CPU-hours. Sweeps are resumable: finished runs (with a `DONE` marker) are skipped.
Each run writes `results/runs/<phase>/<job>/` containing `summary.json` (all metrics), `assigns.csv.gz`
(every assignment with predicted d/C/R and the realised completion time), `output.json.gz` (start-kit output, loadable in PlanViz),
`stats.json.gz` (scheduler telemetry) and `run.log`. Scratch files go to `$LORR_SCRATCH` (default: the system temp dir).

## Repository layout

| Path | Content |
|---|---|
| `sched/` | C++ scheduler (`cda_scheduler.{h,cpp}`) and the start-kit `TaskScheduler.cpp` dispatcher |
| `scripts/build_startkit.sh` | clones LoRR Start-Kit v3.1.0 (commit 9bc75d9) into `third_party/`, installs the scheduler, builds |
| `scripts/make_instances.py` | seeded instance variants (fleet size, delay level, seed) of the official 2026 example maps |
| `scripts/lorr_job.py`, `scripts/ray_sweep.py`, `scripts/expand.py` | one run / resumable sweep of a config / config expansion |
| `scripts/quick_summary.py` | dependency-free paired comparison table for a finished phase |
| `scripts/sanitize_artifacts.py` | strips host names, IPs and absolute paths from run artefacts (`--check` gate) |
| `experiments/configs/` | the configs listed above |
| `analysis/` | paired statistics (Wilcoxon, Holm), tables and figures |
| `lorrsim/`, `scripts/run_experiments.py`, `scripts/feature_validity.py`, `tests/test_lorrsim.py` | earlier pure-Python prototype simulator (kept for reference; not used for the reported results) |
| `tests/test_cda_scheduler.py` | integration tests against the real start-kit binary (pin two bugs found during the study) |

Instances are generated from the official example problems in `third_party/Start-Kit/example_problems`
(or from `data/instances` if that directory exists). Agent and task files depend only on (map, fleet size, seed),
so every scheduler and delay level sees identical robots and tasks, and all comparisons are paired.

The scheduler is configured through environment variables read by the `lifelong` binary:
`LORR_SCHED` (`DEFAULT`, `S0`, `S1`, `S2`, `S3`, `ADAPT`, `BANDIT`), `LORR_LAMBDA`, `LORR_GAMMA`, `LORR_K` (shortlist size),
`LORR_BANDIT_WINDOW`, `LORR_SEED`, `LORR_HORIZON`, `LORR_STATS` (telemetry output path); see `sched/cda_scheduler.cpp`.

The report, paper, generated data and the results of the full study are not included in this repository.

## Team

Han Zeyuan · Su Jiaqi · Tong Yupeng · Wang Ningyu · Wang Shuntao · Yang Junyao
