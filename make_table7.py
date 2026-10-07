#!/usr/bin/env python3
"""
Table 7 (Appendix C.3): Full six-point data-scaling curve.

Paper format — one row per m, one column per architecture:
    m   | MLP | Green's | Attention | NRI-Graph

CFD only. --run performs the six m-sweep experiments (subprocess calls into
run_localizer_experiment.py); combine mode (default) aggregates whatever
m{M}/ directories already exist. Table 2 is a strict two-row derivative of
this same data (see make_table2.py) and does not need its own --run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from table_utils import ARCH_NAMES, aggregate_rmse, fmt, load_raw, ordered_archs

M_VALUES = [10, 25, 50, 100, 150, 180]


def run(output_root: str, device: str, force: bool):
    for m in M_VALUES:
        run_dir = Path(output_root) / f"m{m}"
        if not force and (run_dir / "table1_results_raw.csv").exists():
            print(f"m={m}: results already exist at {run_dir}, skipping (use --force to rerun)")
            continue
        nri_ckpt = Path("checkpoints/cfd") / f"nri_edge_probs_spatial_{m}.pt"
        if not nri_ckpt.exists():
            print(f"m={m}: SKIPPED — no NRI checkpoint at {nri_ckpt}. "
                  f"Train it first: python src/train_nri_spatial.py --n_scenarios {m} "
                  f"--out {nri_ckpt}")
            continue
        cmd = [
            sys.executable, "src/run_localizer_experiment.py", "--full",
            "--n_scenarios", str(m), "--n_sensors", "15",
            "--output", str(run_dir), "--device", device,
        ]
        print(f"m={m}: running {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def combine(output_root: str, output_dir: str):
    rows = []
    archs_seen = set()
    per_m_agg = {}
    for m in M_VALUES:
        run_dir = Path(output_root) / f"m{m}"
        try:
            raw = load_raw(run_dir)
        except FileNotFoundError as e:
            print(f"m={m}: {e}")
            continue
        agg = aggregate_rmse(raw)
        per_m_agg[m] = agg
        archs_seen |= set(agg)

    archs = ordered_archs(archs_seen)
    for m, agg in per_m_agg.items():
        row = {"m": m}
        for arch in archs:
            row[ARCH_NAMES[arch]] = fmt(agg[arch]["mean"], agg[arch]["std"]) if arch in agg else "—"
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("m").reset_index(drop=True)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table7.csv", index=False)
    with open(out / "table7.tex", "w") as f:
        f.write(table.to_latex(index=False, escape=False,
                                caption="Full data-scaling curve (CFD domain).",
                                label="tab:data-scaling-full"))

    print("=" * 70)
    print("TABLE 7: FULL DATA-SCALING CURVE")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved to {out}/table7.csv, {out}/table7.tex")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                         help="Run the six m-sweep experiments (skips any m that already has results)")
    parser.add_argument("--force", action="store_true",
                         help="With --run, rerun even m values that already have results")
    parser.add_argument("--output-root", type=str, default="results/table7",
                         help="Directory containing per-m subdirectories m10/, m25/, ...")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.run:
        run(args.output_root, args.device, args.force)

    combine(args.output_root, args.output_root)


if __name__ == "__main__":
    main()
