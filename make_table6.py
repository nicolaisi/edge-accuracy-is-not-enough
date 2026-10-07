#!/usr/bin/env python3
"""
Table 6 (Appendix B.3): Out-of-distribution generalization.

Paper format (per domain):
    Localiser | Train RMSE | Test RMSE | Gap

--run splits scenarios by a domain condition (ventilation ACH for CFD,
absorption-mapped-to-ACH for acoustic; both read train/test values from
domain_config.py rather than hardcoding them), trains on the train split and
evaluates on the held-out test split via run_localizer_experiment.py's
--scenarios-file pickle path, then combine mode (default) computes the OOD
gap per architecture from the two resulting run directories.
"""

import argparse
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from domain_config import get_domain_config
from table_utils import ARCH_NAMES, aggregate_rmse, fmt, load_raw, ordered_archs


def get_ach(scenario):
    if scenario.get("ventilation") is not None:
        return scenario["ventilation"]
    for key in ["ach", "ACH", "ventilation_rate", "air_changes_per_hour"]:
        if scenario.get(key) is not None:
            return scenario[key]
    name = scenario.get("name")
    if name and "ach" in name.lower():
        match = re.search(r"ach(\d+)", name.lower())
        if match:
            return int(match.group(1))
    return None


def load_and_split_scenarios(data_dir: str, domain: str, n_scenarios: int = 180):
    """Ventilation split, using train/test ACH values from domain_config.py."""
    domain_cfg = get_domain_config(domain)
    train_values = domain_cfg["ood_train_values"]
    test_values = domain_cfg["ood_test_values"]

    print(f"\nLoading {n_scenarios} scenarios from {data_dir} (domain={domain})...")
    if domain == "acoustic":
        from acoustic_loader import load_acoustic_scenarios_as_cfd
        scenarios = load_acoustic_scenarios_as_cfd(data_dir, n_scenarios)
    else:
        from load_cfd_netcdf import load_cfd_from_netcdf_quick
        scenarios = load_cfd_from_netcdf_quick(data_dir, n_scenarios)
    print(f"✓ Loaded {len(scenarios)} scenarios")

    train = [s for s in scenarios if get_ach(s) in train_values]
    test = [s for s in scenarios if get_ach(s) in test_values]
    print(f"  Train: ventilation∈{train_values} -> {len(train)} scenarios")
    print(f"  Test:  ventilation∈{test_values} -> {len(test)} scenarios")

    if not train or not test:
        raise ValueError(f"Empty train or test set! Train: {len(train)}, Test: {len(test)}")
    return train, test


def train_nri_checkpoint_for_split(scenarios_file: Path, domain: str, device: str) -> None:
    """Train the NRI checkpoint for exactly this scenario split.

    Table 6's train/test scenarios are a *filtered* subset (by ventilation/
    absorption condition), not "the first m scenarios" that
    train_nri_spatial.py --n_scenarios m would naively select on its own —
    the checkpoint is looked up purely by scenario count, so a checkpoint
    trained the naive way would silently mismatch this split (wrong
    scenarios, same filename) rather than error. Always (re)train on the
    exact saved split; it's a few seconds, so there's no reason to skip it.
    """
    cmd = [sys.executable, "src/train_nri_spatial.py", "--domain", domain,
           "--scenarios-file", str(scenarios_file), "--n_queries", "32",
           "--epochs", "100", "--device", device]
    print(f"Training NRI checkpoint for this split: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_phase(scenarios_file: str, output_dir: str, test_mode: bool, device: str, domain: str) -> bool:
    cmd = [
        sys.executable, "src/run_localizer_experiment.py",
        "--test" if test_mode else "--full",
        "--scenarios-file", scenarios_file,
        "--output", output_dir,
        "--domain", domain,
        "--device", device,
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Phase failed (output: {output_dir})")
        return False
    print(f"✓ Completed in {(time.time() - start) / 60:.1f} min -> {output_dir}/table1.csv")
    return True


def run(domain: str, output_root: str, data_dir: str, device: str, test_mode: bool, force: bool = False):
    train_dir = Path(output_root) / f"{domain}_train"
    test_dir = Path(output_root) / f"{domain}_test"
    if not force and (train_dir / "table1_results_raw.csv").exists() \
            and (test_dir / "table1_results_raw.csv").exists():
        print(f"{domain}: results already exist at {train_dir} and {test_dir}, "
              f"skipping (use --force to rerun)")
        return

    train_scenarios, test_scenarios = load_and_split_scenarios(data_dir, domain)
    if test_mode:
        train_scenarios, test_scenarios = train_scenarios[:10], test_scenarios[:5]

    scenarios_dir = Path(output_root) / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    train_file = scenarios_dir / f"{domain}_train_scenarios.pkl"
    test_file = scenarios_dir / f"{domain}_test_scenarios.pkl"
    with open(train_file, "wb") as f:
        pickle.dump(train_scenarios, f)
    with open(test_file, "wb") as f:
        pickle.dump(test_scenarios, f)

    train_nri_checkpoint_for_split(train_file, domain, device)
    train_nri_checkpoint_for_split(test_file, domain, device)

    if not run_phase(str(train_file), f"{output_root}/{domain}_train", test_mode, device, domain):
        raise SystemExit(1)
    if not run_phase(str(test_file), f"{output_root}/{domain}_test", test_mode, device, domain):
        raise SystemExit(1)


def combine(domain: str, output_root: str):
    train_dir = f"{output_root}/{domain}_train"
    test_dir = f"{output_root}/{domain}_test"
    try:
        train_raw = load_raw(train_dir)
        test_raw = load_raw(test_dir)
    except FileNotFoundError as e:
        print(f"{domain}: {e}\n  Run: python make_table6.py --run --domain {domain}")
        return None

    train_agg = aggregate_rmse(train_raw)
    test_agg = aggregate_rmse(test_raw)
    archs = ordered_archs(set(train_agg) & set(test_agg))

    rows = []
    for arch in archs:
        train_mean, test_mean = train_agg[arch]["mean"], test_agg[arch]["mean"]
        gap_pct = (test_mean - train_mean) / train_mean * 100
        rows.append({
            "Localiser": ARCH_NAMES[arch],
            "Train RMSE (m)": fmt(train_agg[arch]["mean"], train_agg[arch]["std"]),
            "Test RMSE (m)": fmt(test_agg[arch]["mean"], test_agg[arch]["std"]),
            "Gap": f"{gap_pct:+.1f}%",
        })
    table = pd.DataFrame(rows).sort_values("Test RMSE (m)", key=lambda s: s.str.split("±").str[0].astype(float))

    out = Path(output_root)
    table.to_csv(out / f"table6_{domain}.csv", index=False)
    with open(out / f"table6_{domain}.tex", "w") as f:
        f.write(table.to_latex(index=False, escape=False,
                                caption=f"Out-of-distribution generalization ({domain} domain).",
                                label=f"tab:ood-{domain}"))

    print("=" * 70)
    print(f"TABLE 6 ({domain.upper()}): OUT-OF-DISTRIBUTION GENERALIZATION")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved to {out}/table6_{domain}.csv, {out}/table6_{domain}.tex")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the train/test split experiments")
    parser.add_argument("--force", action="store_true", help="With --run, rerun even if results exist")
    parser.add_argument("--test", action="store_true", help="With --run, use quick-test mode")
    parser.add_argument("--domain", type=str, default="both", choices=["cfd", "acoustic", "both"])
    parser.add_argument("--data-dir", type=str, default=None,
                         help="Defaults to cfd_data/ or acoustic_data/ based on --domain")
    parser.add_argument("--output-root", type=str, default="results/table6")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    domains = ["cfd", "acoustic"] if args.domain == "both" else [args.domain]

    if args.run:
        for d in domains:
            data_dir = args.data_dir or f"{d}_data/"
            run(d, args.output_root, data_dir, args.device, args.test, args.force)

    for d in domains:
        combine(d, args.output_root)


if __name__ == "__main__":
    main()
