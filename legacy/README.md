# Legacy

Code and results here are **not part of the current paper draft** (the one
whose Tables 1–8 the rest of this repository reproduces). Kept for reference
rather than deleted.

- `s12_joint_training.py` — joint multi-task training experiment (λ_dyn /
  λ_loc weighted loss over dynamics + localization). Its docstring labels it
  "Supplement Section S12", but no such section or table exists in the
  current paper. `results/s12_final/joint_training_results.json` holds its
  most recent run.
- `stale_runs/table1_results_full/` — an earlier, incomplete Table 1 CFD run
  (has `table1.csv`/`.tex` but no model checkpoints), superseded by
  `results/table1/cfd/`.
- `stale_runs/s10_non_final/` — an earlier run of the Table 4 correlation/MI
  sweep, superseded by `results/table4/s10_sweep/` (the "final" rerun).
