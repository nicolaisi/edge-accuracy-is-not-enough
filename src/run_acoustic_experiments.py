#!/usr/bin/env python3
"""
Run All Acoustic Domain Experiments
====================================

Master script that runs the complete acoustic experiment pipeline,
producing data for the acoustic columns/rows of Tables 1, 5, and 6.

Prerequisites:
    - generate_acoustic.py in repo root (from synthetic_data/)
    - acoustic_loader.py in repo root
    - domain_config.py in repo root
    - run_localizer_experiment.py in repo root

Usage:
    # Quick smoke test  (~5 min)
    python src/run_acoustic_experiments.py --test

    # Full paper experiments (~4-6 hours total)
    python src/run_acoustic_experiments.py --full

    # Individual steps
    python src/run_acoustic_experiments.py --full --only generate
    python src/run_acoustic_experiments.py --full --only nri
    python src/run_acoustic_experiments.py --full --only table1
    python src/run_acoustic_experiments.py --full --only table5
    python src/run_acoustic_experiments.py --full --only table6
"""

import subprocess
import sys
import time
import argparse
import os
from pathlib import Path


def run(cmd, desc, allow_fail=False):
    """Run a shell command with timing."""
    print(f"\n{'─'*70}")
    print(f"▶ {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*70}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        if allow_fail:
            print(f"⚠ {desc} — exited with {result.returncode} ({elapsed:.0f}s)")
            return False
        else:
            print(f"✗ {desc} — FAILED ({elapsed:.0f}s)")
            sys.exit(1)
    print(f"✓ {desc} — {elapsed:.0f}s")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Run acoustic domain experiments")
    p.add_argument("--test", action="store_true", help="Quick smoke test")
    p.add_argument("--full", action="store_true", help="Full paper experiments")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--only", default=None,
                   choices=["generate", "nri", "table1", "table5", "table6"],
                   help="Run only this step")
    p.add_argument("--data_dir", default="acoustic_data",
                   help="Where to store generated acoustic data")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.test and not args.full:
        print("Specify --test or --full")
        sys.exit(1)

    mode = "test" if args.test else "full"
    data_dir = args.data_dir
    device = args.device
    do_all = args.only is None

    print("=" * 70)
    print(f"ACOUSTIC DOMAIN EXPERIMENTS — {mode.upper()}")
    print("=" * 70)

    total_t0 = time.time()

    # ── Step 1: Generate acoustic data ───────────────────────────────────
    if do_all or args.only == "generate":
        os.makedirs(data_dir, exist_ok=True)

        if args.test:
            run([sys.executable, "src/generate_acoustic.py",
                 "--out", data_dir, "--n_positions", "4"],
                "Generate acoustic data (4 positions, quick)")
        else:
            run([sys.executable, "src/generate_acoustic.py",
                 "--out", data_dir],
                "Generate acoustic data (12 positions, 180 scenarios)")

    # ── Step 2: Train NRI on acoustic dynamics ───────────────────────────
    if do_all or args.only == "nri":
        # Determine scenario count
        import pickle, glob
        pkl_path = os.path.join(data_dir, "all_scenarios.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                n_sc = len(pickle.load(f))
        else:
            n_sc = len(glob.glob(os.path.join(data_dir, "scenario_*.pkl")))

        ckpt_dir = "checkpoints/acoustic"
        os.makedirs(ckpt_dir, exist_ok=True)

        nri_cmd = [
            sys.executable, "src/train_nri_spatial.py",
            "--domain", "acoustic",
            "--data_dir", data_dir,
            "--n_scenarios", str(n_sc),
            "--output_dir", ckpt_dir,
            "--device", device,
        ]
        if args.test:
            nri_cmd += ["--epochs", "10"]

        run(nri_cmd, f"Train NRI on acoustic data (m={n_sc})")

    # ── Step 3: Table 1 — architecture comparison (n=15) ────────────────
    if do_all or args.only == "table1":
        t1_output = "results/table1/acoustic_test" if args.test else "results/table1/acoustic"
        if os.path.exists(os.path.join(t1_output, "table1_results_raw.csv")):
            print(f"\nTable 1 (acoustic) already has results at {t1_output}/, skipping "
                  f"(delete it first if you want to rerun)")
        else:
            t1_cmd = [
                sys.executable, "src/run_localizer_experiment.py",
                "--test" if args.test else "--full",
                "--domain", "acoustic",
                "--data_dir", data_dir,
                "--n_sensors", "15",
                # --nri_edge_dir omitted: the engine derives checkpoints/acoustic/
                # automatically from --domain.
                "--output", t1_output,
                "--device", device,
            ]
            run(t1_cmd, "Table 1 — acoustic architecture comparison")
        run([sys.executable, "make_table1.py"], "Merge Table 1 (CFD + acoustic)", allow_fail=True)

    # ── Step 4: Table 5 — sensor scaling (n=10, 20; n=15 reused from Table 1) ──
    if do_all or args.only == "table5":
        run([sys.executable, "make_table5.py", "--run", "--domain", "acoustic",
             "--device", device] + (["--force"] if args.test else []),
            "Table 5 — acoustic sensor-count scaling")

    # ── Step 5: Table 6 — OOD generalisation ─────────────────────────────
    if do_all or args.only == "table6":
        t6_cmd = [sys.executable, "make_table6.py", "--run", "--domain", "acoustic",
                   "--data-dir", data_dir, "--device", device]
        if args.test:
            t6_cmd.append("--test")
        run(t6_cmd, "Table 6 — acoustic OOD generalisation", allow_fail=True)

    # Note: Table 8 (NRI threshold/top-k ablation) is CFD-only in the paper —
    # there is no acoustic step for it here.

    # ── Done ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"✓ ALL ACOUSTIC EXPERIMENTS COMPLETE — {total_elapsed/60:.1f} min")
    print(f"{'='*70}")

    print(f"\nOutputs:")
    for d in ["results/table1/acoustic", "results/table5/acoustic_n10", "results/table5/acoustic_n20",
              "results/table6/acoustic_train", "results/table6/acoustic_test"]:
        print(f"  {d}/table1.csv")
    print(f"\nNext: python make_table1.py / make_table5.py / make_table6.py")


if __name__ == "__main__":
    main()
