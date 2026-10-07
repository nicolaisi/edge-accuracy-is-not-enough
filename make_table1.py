#!/usr/bin/env python3
"""
Table 1: Architecture comparison (CFD + Acoustic).

Paper format:
    Localiser          | CFD RMSE (m) | Acoustic RMSE (m)
    MLP (no structure) | ...          | ...
    Green's Function   | ...          | ...
    Learned Attention  | ...          | ...
    NRI-Graph          | ...          | ...

This merges two independent single-domain runs (produced by
src/run_localizer_experiment.py --full --domain {cfd,acoustic} --n_sensors 15)
into the paper's one two-domain table. There is no experiment specific to
this script; --run only points at the two prerequisite runs if missing.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from table_utils import ARCH_NAMES, aggregate_rmse, fmt, load_raw, ordered_archs


def combine(cfd_dir: str, acoustic_dir: str, output_dir: str):
    cfd_raw = load_raw(cfd_dir)
    acoustic_raw = load_raw(acoustic_dir)

    cfd_agg = aggregate_rmse(cfd_raw)
    acoustic_agg = aggregate_rmse(acoustic_raw)

    archs = ordered_archs(set(cfd_agg) | set(acoustic_agg))

    rows = []
    for arch in archs:
        cfd = cfd_agg.get(arch)
        acoustic = acoustic_agg.get(arch)
        rows.append({
            "Localiser": ARCH_NAMES[arch],
            "CFD RMSE (m)": fmt(cfd["mean"], cfd["std"]) if cfd else "—",
            "Acoustic RMSE (m)": fmt(acoustic["mean"], acoustic["std"]) if acoustic else "—",
        })

    table = pd.DataFrame(rows)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table1.csv", index=False)
    with open(out / "table1.tex", "w") as f:
        f.write(table.to_latex(index=False, escape=False,
                                caption="Structure comparison across domains.",
                                label="tab:architecture-comparison"))

    print("=" * 70)
    print("TABLE 1: ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved to {out}/table1.csv, {out}/table1.tex")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                         help="No independent experiment for Table 1 — this only "
                              "checks that the two prerequisite domain runs exist.")
    parser.add_argument("--cfd-dir", type=str, default="results/table1/cfd")
    parser.add_argument("--acoustic-dir", type=str, default="results/table1/acoustic")
    parser.add_argument("--output-dir", type=str, default="results/table1")
    args = parser.parse_args()

    if args.run:
        missing = [d for d in (args.cfd_dir, args.acoustic_dir)
                   if not (Path(d) / "table1_results_raw.csv").exists()]
        if missing:
            print("Table 1 has no independent --run step. Missing prerequisite run(s):")
            for d in missing:
                domain = "cfd" if "cfd" in d else "acoustic"
                print(f"  python src/run_localizer_experiment.py --full --domain {domain} "
                      f"--n_sensors 15 --output {d}")
            raise SystemExit(1)
        print("Both prerequisite runs already exist — nothing to run. "
              "Continuing to combine.")

    combine(args.cfd_dir, args.acoustic_dir, args.output_dir)


if __name__ == "__main__":
    main()
