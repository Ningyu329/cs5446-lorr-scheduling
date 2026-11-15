# Published numerical summaries and visualizations

These are the **original derived tables and figure images** from the team's completed
2,326-run LoRR experiment, plus separately labeled follow-up observations.
Files were copied from the team's project archive; the original raw per-run simulator
outputs (the many `results/runs/` directories) are deliberately omitted.

- `tables/*.csv`, `tables/key_numbers.json`: released aggregate statistics,
  ablations, density study, feature validity, runtime/noise, tuning and scale.
- `../assets/figures/fig_*.png`: original project figures; `fig_followup_bandit_lag.png`
  is newly plotted from `followup/bandit_audit_180.csv`.
- `followup/*.csv`: the new **exploratory** A/B/C summaries, separate from the
  2,326 main-study simulations. B has only 3 paired seeds; C is a task-level
  proxy rather than per-errand reward credit assignment.

## Reproduction tiers

1. **Smoke test** (`experiments/configs/quick.json`): 6 real simulations;
   does not reproduce publication numbers.
2. **View/reanalyze released summary tables**: no simulator or cluster required.
   Run `python3 analysis/plot_published_tables.py` (requires `matplotlib`) to
   create independent plots directly from the checked-in derived tables.
3. **Rebuild the exact original published figures**: requires the entire original
   `results/runs/` archive (not on GitHub), then run `analysis/make_figures.py`.
   For history and provenance, see the team report and original experiment configs.
4. **Rerun the full experiments**: scripts and configs are public, but this costs
   substantial compute and results vary with the wall-clock bounded planner.

## Interpretation cautions

- Original main pooled S1/S2/S3/ADAPT/BANDIT throughput effects did **not**
  survive Holm multiple-testing adjustment.
- The 1.5× warehouse +4.45% signal was selected after inspecting many cells;
  the original multiple-testing-corrected comparisons did not pass a 0.05 cutoff.
- Nonparametric bootstrap confidence intervals from those original 10 seeds
  describe uncertainty in those observations; they are *not* a confirmation
  with independently generated seeds.
- Follow-up A's six 0.5× runs are valid but the six 1.5× runs were OOM-killed,
  so they do not estimate a new high-density effect.
