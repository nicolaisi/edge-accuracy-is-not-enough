# Edge Accuracy Is Not Enough: Why Dynamics-Learned Structure Fails to Transfer to Inverse Problems
### Reproduction Guide

This guide maps every table in the paper to the exact commands needed to reproduce it, on two physical domains: **CFD** (hydrogen leak detection, Navier–Stokes) and **Acoustic** (synthetic source localization, Helmholtz equation).

Scripts and output directories are named directly after the paper's table numbers (`make_table1.py` → `results/table1/`, ..., `make_table8.py` → `results/table8/`). Each `make_tableN.py` supports two modes: `--run` performs the (often multi-hour) experiment(s) that table needs, and the default (no flag) mode combines whatever results already exist into the paper-formatted table. Tables 2 and 3 are pure derivatives of Tables 7 and 1 respectively and have no independent `--run` step.

**The only scripts you run directly are `make_table1.py` … `make_table8.py`, at the repo root.** Everything else — the shared training engine, data loaders, and standalone data/figure-prep utilities — lives in `src/` and is invoked automatically by the `make_tableN.py` scripts (or by hand only if you want finer control; commands below always show the `src/` path where needed). Always run commands from the repo root.

---

## Table of Contents

1. [Repository layout](#repository-layout)
2. [Prerequisites](#prerequisites)
3. [Data](#data)
4. [Reproducing each table](#reproducing-each-table)
   - [Table 1 — Architecture comparison](#table-1--architecture-comparison)
   - [Table 2 — Data scaling (summary)](#table-2--data-scaling-summary)
   - [Table 3 — Transfer outcomes vs. Jaccard similarity](#table-3--transfer-outcomes-vs-jaccard-similarity)
   - [Table 4 — Additional structure baselines](#table-4--additional-structure-baselines)
   - [Table 5 — Sensor-count scaling](#table-5--sensor-count-scaling)
   - [Table 6 — Out-of-distribution generalization](#table-6--out-of-distribution-generalization)
   - [Table 7 — Full data-scaling curve](#table-7--full-data-scaling-curve)
   - [Table 8 — NRI threshold / top-k ablation](#table-8--nri-threshold--top-k-ablation)
5. [Figure 1 — Structural misalignment](#figure-1--structural-misalignment)
6. [Acoustic domain](#acoustic-domain)
7. [Expected runtimes](#expected-runtimes)
8. [Legacy](#legacy)

---

## Repository layout

```
.
├── # ── One script per paper table (run these) ──────────────────────
├── make_table1.py   → results/table1/   # Architecture comparison (CFD + Acoustic)
├── make_table2.py   → results/table2/   # Data scaling, m=10 vs m=180 (derived from Table 7)
├── make_table3.py   → results/table3/   # Transfer outcomes vs. Jaccard similarity
├── make_table4.py   → results/table4/   # Correlation-Graph / MI-Graph baselines
├── make_table5.py   → results/table5/   # Sensor-count scaling, n=10/15/20
├── make_table6.py   → results/table6/   # Out-of-distribution generalization
├── make_table7.py   → results/table7/   # Full six-point data-scaling curve (m=10..180)
├── make_table8.py   → results/table8/   # NRI threshold / top-k ablation
│
├── # ── src/ — everything the make_tableN.py scripts run for you ────
├── src/
│   ├── run_localizer_experiment.py  # Trains/evaluates one (domain, n_sensors, n_scenarios)
│   │                                 # condition for all 4 architectures; writes table1.csv-format
│   │                                 # output. Invoked by every make_tableN.py above.
│   ├── table_utils.py               # Shared per-architecture RMSE aggregation helper
│   ├── localizer_architectures.py   # MLP / Green's / Attention / NRI-Graph
│   ├── load_cfd_netcdf.py           # CFD NetCDF loader
│   ├── domain_config.py             # Domain-specific constants (bounds, OOD split values, ...)
│   ├── train_nri_spatial.py         # NRI pre-training (produces the NRI checkpoints)
│   ├── s10_structure_baselines.py   # Correlation-graph / MI-graph baselines (feeds Table 4)
│   ├── prepare_supplement_data.py   # Data prep for s10_structure_baselines.py
│   ├── visualize_structure_comparison.py  # Fig. 1 — structural misalignment; also emits the
│   │                                       # Jaccard JSON that Table 3/4 consume
│   ├── generate_acoustic.py         # Synthetic Helmholtz data generator
│   ├── acoustic_loader.py           # CFD-compatible loader for acoustic data
│   ├── visualize_acoustic.py        # Acoustic domain diagnostics (not table/figure-specific)
│   └── run_acoustic_experiments.py  # Master orchestration for the acoustic --run steps
│
├── # ── Legacy (not part of the current paper) ──────────────────────
├── legacy/                         # See legacy/README.md
│
├── # ── Data (not committed — see .gitignore) ────────────────────────
├── cfd_data/                       # 180 CFD scenarios (NetCDF, ~GB)
├── acoustic_data/                  # 180 acoustic scenarios (pickle, ~11 s to regenerate)
│
├── # ── Pre-trained NRI checkpoints ──────────────────────────────────
├── checkpoints/
│   ├── cfd/                        # nri_edge_probs_spatial_{m}.pt
│   └── acoustic/                   # nri_edge_probs_spatial_{m}.pt
│
└── README.md
```

`src/` is a plain directory, not an installed package — each `make_tableN.py` adds it to `sys.path` at startup, so `python make_table1.py` (etc.) works from the repo root with no extra setup (`PYTHONPATH`, `pip install -e .`, ...).

> **Not committed to git:** `cfd_data/`, `acoustic_data/`, all `*.pt`/`*.pth` checkpoints, and `*.zip` archives. See `.gitignore`.

---

## Prerequisites

```bash
# Core
pip install numpy pandas scipy matplotlib seaborn tqdm netCDF4 scikit-learn

# PyTorch — pick the build matching your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
# pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

Python 3.9+ is required. All experiments were run on a single NVIDIA A100; CPU runs are slower but fully supported.

---

## Data

### CFD data (hydrogen leak, 180 scenarios)

The CFD scenarios (180 total: 12 positions × 5 leak rates × 3 ventilation levels) are hosted on Zenodo, split across four archives. Download and extract all four into the same `cfd_data/` directory:

```bash
mkdir -p cfd_data

# Part 1
wget -O cfd_part1.zip "https://zenodo.org/records/18740541?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijg2NDhhYWJjLTQ3NzEtNGRjMy04NjAwLTViMWJjMGY3ZTE4ZiIsImRhdGEiOnt9LCJyYW5kb20iOiIyNTU4Yzc3NWJjNTQxZDdlMTU4YTAwODMxMTIzYjdmYiJ9.0Qq3T_B63QbFURN3hTEyna5z8hqe2oovJwoMGMWnHvPYDV_B8xJBV6x_3Tx-fC-N2_74kSeTj2G_EW-cH4WJSA"

# Part 2
wget -O cfd_part2.zip "https://zenodo.org/records/18745400?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE4YWZjZjUzLTEwZWUtNDAwZi1hOTUxLTkyNTdiMzM0Y2Y1ZSIsImRhdGEiOnt9LCJyYW5kb20iOiIyYWQxNmVhOWNjOGFmODgwOTgwMDhlZmMzOGE1OGQ1NiJ9.yWC726vlbl6beGJmaw6KaZqSGQcpBaNUfYjmEYW5UWmo3_WaX16lEkM-ZMgjUM1aQABsebX6Uih31LKznZYRZA"

# Part 3
wget -O cfd_part3.zip "https://zenodo.org/records/18754599?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjExMWM0NDM2LThiMTYtNDE1Ni04MzJiLTgyNDkzZTUwZmI4MCIsImRhdGEiOnt9LCJyYW5kb20iOiIyMzE1YmIwOWYzNmI3ODU2YTExNjZjNmVjYjBmY2MwMSJ9.6XYu9JalR1Cv5L3IsscsNQTaiF2hThlQp6fOeoOOXEYoHFxLPfEGcQUJHTnwwnzs312orpeup1GA60HTvoUJEg"

# Part 4
wget -O cfd_part4.zip "https://zenodo.org/records/18754802?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJkMGY2ZjAxLTFmNGEtNDIyMy1iM2IyLTczYTkzNzkzNjRhMiIsImRhdGEiOnt9LCJyYW5kb20iOiIxYWMwMTEwODA2YjBkY2Q4MmJiMjQ2MWI1ZDQyNjVhMyJ9.jGMm-XIAknrqB9ulWHlghYm-8B71lYLoNivtnt4fTT2pUVN4jhHcfK4ZMzUnF4ZYWnETDOqnw0PtrkNOujhX-A"

# Extract all four into cfd_data/ and remove the zip files
for i in 1 2 3 4; do
  unzip -q cfd_part${i}.zip -d cfd_data/
  rm cfd_part${i}.zip
done
```

After extraction, `cfd_data/` should contain 180 scenario subdirectories (e.g. `cfd_data/ach3m100p1/visout00001.nc`, ...).

Sanity check:

```bash
python src/load_cfd_netcdf.py cfd_data/ 4    # load and summarise 4 scenarios
```

### Acoustic data (Helmholtz, 180 scenarios — fully reproducible)

The acoustic scenarios are generated from scratch in ~11 seconds on CPU (no external data needed):

```bash
python src/generate_acoustic.py --out acoustic_data/
python src/acoustic_loader.py --data_dir acoustic_data/   # verify format
```

---

## Reproducing each table

The NRI checkpoint must be trained before any experiment using the `nri` architecture; `src/run_localizer_experiment.py` raises a clear error naming the missing checkpoint if one isn't found. It looks for `checkpoints/{domain}/nri_edge_probs_spatial_{m}.pt`, derived automatically from `--domain` (pass `--nri_edge_dir` to override). To (re)train one for `m` scenarios:

```bash
python src/train_nri_spatial.py --domain cfd --data_dir cfd_data/ --n_scenarios 180 --n_queries 32 --epochs 100
# writes checkpoints/cfd/nri_edge_probs_spatial_180.pt (add --out to override)
```

The migrated results already on disk (`results/table1/`, `results/table5/`, `results/table6/`, `results/table7/`, `results/table8/`, `results/table4/s10_sweep/`) are real, previously-computed runs. For Tables 3–8, `--run` skips any condition that already has results unless you pass `--force` (Tables 1 and 2 have no `--run` experiment of their own — see their sections below). **Before running `--run` on any table, check whether its output directory already has a `table1_results_raw.csv`** — if it does, `--run` is a no-op and safe; if it doesn't, `--run` starts real training that can take hours.

> **Reruns are seeded but not bit-exact.** Each of the 10 seeds fixes the sensor-placement sampling, the train/val/test split, and model initialization, but `DataLoader(..., shuffle=True)`'s per-epoch order depends on unpinned global RNG state, and no `torch.use_deterministic_algorithms` flag is set, so CPU floating-point reduction order can vary run to run. Over 100 epochs those small differences compound. Expect the same architecture ranking and RMSE within roughly ±0.01–0.1 m of a prior run, not identical values — the already-migrated `results/table1/cfd/table1.csv`, for instance, is close to but not exactly the paper's reported numbers for the same reason.

### Table 1 — Architecture comparison

**Paper reference:** Table 1 (n=15 sensors, m=180 scenarios, CFD + Acoustic)

```bash
# Prerequisite runs (skip if results/table1/{cfd,acoustic}/ already exist)
python src/run_localizer_experiment.py --full --domain cfd --n_sensors 15 --output results/table1/cfd
python src/run_localizer_experiment.py --full --domain acoustic --n_sensors 15 --output results/table1/acoustic

# Merge into the paper's two-domain table
python make_table1.py
```

Output: `results/table1/table1.csv`, `results/table1/table1.tex`.

### Table 2 — Data scaling (summary)

**Paper reference:** Table 2 (m=10 vs. m=180, CFD) — a two-row summary of Table 7's data, with an Improvement row and a paired significance test (NRI vs. Attention).

```bash
python make_table7.py --run     # only if results/table7/m{10,180} don't already exist
python make_table2.py
```

Output: `results/table2/table2.csv`, `results/table2/table2.tex`.

### Table 3 — Transfer outcomes vs. Jaccard similarity

**Paper reference:** Table 3 (Section 5.1/5.4) — RMSE, degradation vs. Learned Attention, and Jaccard similarity to Learned Attention's graph, for {CFD, Acoustic} × {Green's Function, NRI Learned Graph}.

```bash
python make_table3.py --run     # generates the per-domain Jaccard JSON if missing
python make_table3.py
```

Output: `results/table3/table3.csv`, `results/table3/table3.tex`.

### Table 4 — Additional structure baselines

**Paper reference:** Table 4 — Correlation-Graph and Mutual-Information-Graph baselines, alongside MLP/NRI-Graph reused from Table 1.

```bash
python make_table4.py --run     # prepares supplement_data.pt and runs the correlation/MI sweep if missing
python make_table4.py
```

Output: `results/table4/table4.csv`, `results/table4/table4.tex`.

### Table 5 — Sensor-count scaling

**Paper reference:** Table 5 (n ∈ {10, 15, 20}, m=180, both domains). n=15 is reused from Table 1 rather than rerun.

```bash
python make_table5.py --run --domain both     # runs n=10/n=20 for CFD and acoustic
python make_table5.py --domain both
```

Output: `results/table5/table5_cfd.csv`/`.tex`, `results/table5/table5_acoustic.csv`/`.tex`.

### Table 6 — Out-of-distribution generalization

**Paper reference:** Table 6 (Appendix B.3) — train on ACH∈{6,10} (or the acoustic absorption equivalent), test on the held-out condition, both domains.

```bash
python make_table6.py --run --domain both
python make_table6.py --domain both
```

Output: `results/table6/table6_cfd.csv`/`.tex`, `results/table6/table6_acoustic.csv`/`.tex`.

> **Only the acoustic OOD run currently has data migrated into this repo** (`results/table6/acoustic_{train,test}/`) — `make_table6.py --domain acoustic` works immediately. The CFD OOD run has never been produced in this repo: `python make_table6.py --run --domain cfd` (or `--domain both`) will genuinely start a ~2–3 hour training run, not a no-op — don't run it unless you mean to.

### Table 7 — Full data-scaling curve

**Paper reference:** Table 7 (Appendix C.3) — all six m values (10, 25, 50, 100, 150, 180), all four architectures.

```bash
python make_table7.py --run
python make_table7.py
```

Output: `results/table7/table7.csv`, `results/table7/table7.tex`.

### Table 8 — NRI threshold / top-k ablation

**Paper reference:** Table 8 (Appendix C.4) — τ ∈ {0.1, 0.2, 0.3, 0.5, 0.7} and top-k ∈ {2, 3, 4}, CFD only, NRI-Graph only.

```bash
python make_table8.py --run --plot
python make_table8.py --plot
```

Output: `results/table8/table8.csv`, `results/table8/table8.tex`, `results/table8/figures/*.pdf`.

---

## Figure 1 — Structural misalignment

Generates the three-panel Green's/NRI/Learned-Attention comparison, and (as a side effect) the `*_jaccard.json` files that `make_table3.py` and `make_table4.py` consume:

```bash
python src/visualize_structure_comparison.py --domain cfd \
  --nri-path checkpoints/cfd/nri_edge_probs_spatial_180.pt \
  --attention-model results/table1/cfd/model_attention_seed0.pt \
  --n-sensors 15 --output results/table3/structure_comparison_cfd

python src/visualize_structure_comparison.py --domain acoustic \
  --nri-path checkpoints/acoustic/nri_edge_probs_spatial_180.pt \
  --attention-model results/table1/acoustic/model_attention_seed0.pt \
  --n-sensors 15 --output results/table3/structure_comparison_acoustic
```

---

## Acoustic domain

```bash
# Generate data (~11 s)
python src/generate_acoustic.py --out acoustic_data/

# Train NRI on acoustic dynamics (writes checkpoints/acoustic/nri_edge_probs_spatial_180.pt)
python src/train_nri_spatial.py --domain acoustic --data_dir acoustic_data/ \
  --n_scenarios 180 --n_queries 32 --epochs 100

# Master script: runs Table 1 (acoustic), Table 5 (acoustic), Table 6 (acoustic) in sequence
python src/run_acoustic_experiments.py --test    # smoke test, ~5 min
python src/run_acoustic_experiments.py --full    # ~4-6 hours
```

---

## Expected runtimes

All table runtimes below are measured on a MacBook Pro (CPU only) — the full reproduction pipeline has been run end to end on this machine. Only acoustic data generation itself is still an unverified estimate.

| Table | Run step | Runtime (MacBook Pro) |
|---|---|---|
| 1 | `src/run_localizer_experiment.py --full --domain cfd` | **1h 33m (measured)** |
| 1 | `src/run_localizer_experiment.py --full --domain acoustic` | **50 min (measured)**, consistent across 2 runs |
| 2 | none (derived from Table 7) | — |
| 3 | `make_table3.py --run` (2 figure generations, cheap) | **~4 s + combine (measured, both domains)** |
| 4 | `make_table4.py --run` (data prep + correlation/MI sweep) | **~1 min (measured)** — all 70 seeded runs are lightweight; not the ~2h originally estimated |
| 5 | `make_table5.py --run --domain acoustic` (n=10 + n=20) | **1h 41m (measured)** |
| 5 | `make_table5.py --run --domain cfd` (n=10 + n=20) | **1h 40m (measured)** |
| 6 | `make_table6.py --run --domain acoustic` (train+test) | **~55 min (measured)** |
| 6 | `make_table6.py --run --domain cfd` (train+test) | **51 min (measured)** |
| 7 | `make_table7.py --run` (6 m values, CFD only) | **2h 31m (measured)** — well under the ~8h originally estimated |
| 8 | `make_table8.py --run` (5 τ + 3 top-k, CFD only) | **1h 34m (measured)** — well under the ~4–5h originally estimated |
| Acoustic data generation | `src/generate_acoustic.py` | ~11 s (estimated) |

> **Table 6 checkpoint-leakage note (fixed)**: `run_localizer_experiment.py`'s NRI checkpoint is looked up purely by scenario *count* (`nri_edge_probs_spatial_{m}.pt`), which used to silently break for Table 6's OOD split — the OOD train/test scenarios are a *filtered* subset (by ventilation/absorption condition), not "the first m scenarios" that `train_nri_spatial.py --n_scenarios m` would naively select. This first surfaced as an acoustic NRI-Graph OOD gap of +68% (vs. the paper's −3.7%). Fixed in `make_table6.py --run` itself: it now always (re)trains the NRI checkpoint from the exact saved `results/table6/scenarios/{domain}_{train,test}_scenarios.pkl` split before training the localizer, for both domains, automatically — no manual step needed. Confirmed correct on both domains: acoustic NRI-Graph gap now −2.0% and CFD NRI-Graph gap now −3.4%, both matching the paper's sign and ballpark.

"Measured" entries are confirmed by an actual run; "estimated" entries haven't been timed on this machine yet and are rough projections — update them here as you collect real numbers.

---

## Legacy

`legacy/` holds code and results from an earlier paper draft that are not referenced by the current Tables 1–8 — see `legacy/README.md`.
