# CS5446 Team27 — Integrated Follow-up Study

This package supplements (does not overwrite) the existing LoRR/CS5446 study.
It contains **three linked investigations**:

- **A. Criticality** — is the relative benefit of congestion/risk awareness larger at warehouse 1.5× density than at 0.5×?
- **B. Avoidability** — do extra room-to-room passages change the benefit of congestion-aware assignment?
- **C. Delayed feedback** — how often does task completion arrive after the Bandit's 100-tick arm-selection window?

Important: A includes both congestion and risk (S3 vs S0); B isolates congestion (S1 vs S0). C is a *task-completion proxy*, whereas the Bandit rewards errands, so do not interpret it as precise causal credit assignment.

## Installation in the existing Linux/WSL project

Extract the ZIP **into the project root**. The ZIP contains a directory named `followup/`, so do not unzip *inside* an extra `followup` directory.

```bash
cd ~/cs5446-lorr-scheduling
unzip -o /path/to/CS5446_followup_study.zip -d .
```

You need `scripts/` and a built `third_party/Start-Kit/build/lifelong` (run
`scripts/build_startkit.sh` first). The public repository's `make_instances.py`
uses the official Start-Kit example problems when `data/instances/` is absent.
The bottleneck runner accepts either location for the original Room map.
Published summaries are in `published_results/followup/`; the raw full runs are **not** in this repository.

## C — Offline Bandit diagnosis (start here; no simulation)

Use the bundled, already-checked example from the connected Drive:

```bash
cd ~/cs5446-lorr-scheduling
python3 followup/study.py audit-bandit --runs followup/sample_bandit --out followup/sample_bandit/new_report.csv
```

Expected for that **one existing run**: 327 completed tasks; 97.55% complete in later 100-tick windows; 83.18% finish under a different arm; median task time 496 ticks. These are **task-level proxies** and not errand-level credit-assignment estimates.

If the original full `results/runs/main` is available locally (e.g. in the team's archive), audit all available BANDIT runs:

```bash
python3 followup/study.py audit-bandit --runs results/runs/main
```

If no runs are found, the Drive results have **not** been synced locally. Do not claim that the full 180-run audit has been done. The bundled sample still works without them.

## A — Density interaction: independent seeds

The existing 0.5× and 1.5× density results suggest an exploratory interaction. Do **not** treat those same seeds as an independent confirmation. We use *new* seeds 20–29, with fixed existing weights `(lambda,gamma)=(2,1)`, warehouse domain, base delays, S0 and S3, 3000 ticks.

First a 3-seed functionality pilot (12 simulations):

```bash
cd ~/cs5446-lorr-scheduling
python3 scripts/ray_sweep.py followup/configs/criticality_pilot.json --local 1
python3 followup/study.py criticality-report
```

Then only if the pilot is sound, complete the prespecified 10-seed comparison (40 total simulations; prior 12 are skipped via `DONE` files):

```bash
python3 scripts/ray_sweep.py followup/configs/criticality_confirm.json --local 1
python3 followup/study.py criticality-report
```

`--local 1` deliberately runs *one simulation at a time* to avoid planner wall-clock interference. Large warehouse runs may be heavy on a laptop. Re-running the same command resumes the job list, and original results under `results/runs/main` and `results/runs/density` are untouched.

Primary estimand: within-seed **interaction**, `relative-effect(S3 vs S0 at 1.5×) - relative-effect(S3 vs S0 at 0.5×)`; not simply whether a single condition p-value is below 0.05. If the effect does not replicate, report this accurately.

## B — Alternative-passages controlled pilot

The script creates a **new** Room map (`data/followup_maps/room-64-64-16-bypass.map`) by opening 24 blocked cells which have free neighbors on opposite sides. The original map stays unchanged. The SAME generated robot starts, task file, delay seed, and fixed planner are used for each original-vs-bypass comparison.

Start with a dry-run that checks the source map and input generator:

```bash
cd ~/cs5446-lorr-scheduling
python3 followup/study.py bottleneck --seeds 20 21 22 --dry
```

Then execute only if the dry run succeeded:

```bash
python3 followup/study.py bottleneck --seeds 20 21 22
python3 followup/study.py bottleneck-report
```

This is 3 seeds × 2 map versions × 2 schedulers (S0/S1) = 12 simulations. For a more serious result, expand to at least 10 **prespecified** seeds (20–29); the runner skips completed summaries. Results are saved under `results/followup_bottleneck/`, not into the team's original `results/runs` tree.

**Scientific limitation:** opening passages also changes graph distances, so a throughput interaction is evidence about topological context, not alone a causal proof of congestion avoidability. Inspect whether the second passage is actually used and waiting shifts before making a strong claim.

## Combining conclusions

A: identify a potential *density regime* in which S3 adds value.
B: manipulate *topological alternatives* while isolating C via S1/S0.
C: identify a potential *feedback timing* limitation of online bandit adaptation.

Interpret all three as exploratory mechanism probes unless independently replicated. Report all conditions and negative outcomes, avoid choosing only favorable seeds, and keep test parameters frozen.

## Known limits

- This package was tested against the original repository's schema and Drive sample logs, but **the full LoRR simulations have not been executed in this workspace**. They must be run in an environment with the built Start-Kit executable and original data files.
- Full 2026 competition online submission is not relevant to this course follow-up.
- The scripts do not install dependencies or overwrite existing scheduler/experiment code.
- `criticality-report` uses a Student t confidence interval; for n<2 its CI is undefined, and small pilot n is for functionality, not definitive significance.
- `bottleneck` writes a bypass instance JSON in `data/generated/room/` without overwriting the original generated JSON. If those generated files exist already, they are reused.

## Published follow-up data and observed pilot outcomes

- `published_results/followup/bandit_audit_180.csv`: historical 180-run task-duration proxy; medians across runs: 840 ticks, 99.28% cross-window, 84.17% different-arm. **Not causal evidence of learning failure.**
- `published_results/followup/bottleneck_pilot.csv`: seed 20--22, original vs 24 added passages, S0/S1. Mean topology interaction approximately -0.69 percentage points (high uncertainty); the hypothesis of better S1 relative performance with bypasses was **not supported**.
- `published_results/followup/low_density_pilot.csv`: three new 0.5× warehouse seeds (S0/S3); no independent 1.5× confirmation. The separate 1.5× local pilot failed because of WSL OOM, so no throughput values are reported for those failed jobs.
- Prior 0.5× versus 1.5× density evidence (`published_results/tables/density_effects.csv`) is **exploratory**; bootstrap re-sampling of these same observations is not a fresh experiment.

All follow-up data are separate from the original 2,326-run total. The 2026-11-15 analysis is exploratory; there was no competition submission.
