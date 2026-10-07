#!/usr/bin/env python3
"""
Table 5: Sensor-count scaling (n=10, 15, 20).

Paper format (per domain):
    n  | MLP | Green's | Attention | NRI-Graph

n=15 is reused from Table 1's runs (results/table1/{cfd,acoustic}/) rather
than rerun. --run performs the n=10/n=20 sweeps for one or both domains;
combine mode (default) assembles the per-domain tables from whatever n10/n15/n20
directories already exist.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from table_utils import ARCH_NAMES, aggregate_rmse, fmt, load_raw, ordered_archs

N_VALUES = [10, 15, 20]


def dirs_for_domain(domain: str, output_root: str):
    prefix = "acoustic_" if domain == "acoustic" else ""
    n15 = f"results/table1/{domain}"
    return {
        10: Path(output_root) / f"{prefix}n10",
        15: Path(n15),
        20: Path(output_root) / f"{prefix}n20",
    }


def run(domain: str, output_root: str, device: str, force: bool):
    for n in (10, 20):
        run_dir = dirs_for_domain(domain, output_root)[n]
        if not force and (run_dir / "table1_results_raw.csv").exists():
            print(f"{domain} n={n}: results already exist at {run_dir}, skipping (use --force to rerun)")
            continue
        cmd = [
            sys.executable, "src/run_localizer_experiment.py", "--full",
            "--n_sensors", str(n), "--domain", domain,
            "--output", str(run_dir), "--device", device,
        ]
        print(f"{domain} n={n}: running {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def combine_domain(domain: str, output_root: str, output_dir: str):
    dirs = dirs_for_domain(domain, output_root)
    per_n_agg = {}
    archs_seen = set()
    for n, run_dir in dirs.items():
        try:
            raw = load_raw(run_dir)
        except FileNotFoundError as e:
            print(f"{domain} n={n}: {e}")
            continue
        agg = aggregate_rmse(raw)
        per_n_agg[n] = agg
        archs_seen |= set(agg)

    archs = ordered_archs(archs_seen)
    rows = []
    for n, agg in sorted(per_n_agg.items()):
        row = {"n": n}
        for arch in archs:
            row[ARCH_NAMES[arch]] = fmt(agg[arch]["mean"], agg[arch]["std"]) if arch in agg else "—"
        rows.append(row)
    table = pd.DataFrame(rows)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"table5_{domain}.csv", index=False)
    with open(out / f"table5_{domain}.tex", "w") as f:
        f.write(table.to_latex(index=False, escape=False,
                                caption=f"Sensor-count scaling ({domain} domain).",
                                label=f"tab:sensor-scaling-{domain}"))

    print("=" * 70)
    print(f"TABLE 5 ({domain.upper()}): SENSOR-COUNT SCALING")
    print("=" * 70)
    print(table.to_string(index=False))

    # n=10 -> n=20 improvement, printed for reference (not saved separately)
    if 10 in per_n_agg and 20 in per_n_agg:
        print("\nChange (n=10 -> n=20):")
        for arch in ordered_archs(set(per_n_agg[10]) & set(per_n_agg[20])):
            r10, r20 = per_n_agg[10][arch]["mean"], per_n_agg[20][arch]["mean"]
            print(f"  {ARCH_NAMES[arch]:20s}: {r10:.2f} -> {r20:.2f} ({(r20 - r10) / r10 * 100:+.0f}%)")
    print()
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the n=10/n=20 sweeps")
    parser.add_argument("--force", action="store_true", help="With --run, rerun even if results exist")
    parser.add_argument("--domain", type=str, default="both", choices=["cfd", "acoustic", "both"])
    parser.add_argument("--output-root", type=str, default="results/table5")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    domains = ["cfd", "acoustic"] if args.domain == "both" else [args.domain]

    if args.run:
        for d in domains:
            run(d, args.output_root, args.device, args.force)

    for d in domains:
        combine_domain(d, args.output_root, args.output_root)

    print(f"Saved to {args.output_root}/table5_{{domain}}.csv/.tex")


if __name__ == "__main__":
    main()
