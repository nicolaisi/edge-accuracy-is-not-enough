#!/usr/bin/env python3
"""
Table 2: Data scaling (m=10 vs m=180).

Paper format:
    m           | MLP  | Green's | Attention | NRI-Graph
    10          | ...  | ...     | ...       | ...
    180         | ...  | ...     | ...       | ...
    Improvement | ...% | ...%    | ...%      | ...%***

CFD only. Pure derivative of Table 7's m=10 and m=180 runs — reads those two
run directories directly, so it doesn't require results/table7/table7.csv to
exist first. No independent --run step.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from table_utils import ARCH_NAMES, aggregate_rmse, fmt, load_raw, ordered_archs

M_LOW, M_HIGH = 10, 180


def combine(m_low_dir: str, m_high_dir: str, output_dir: str):
    raw_low = load_raw(m_low_dir)
    raw_high = load_raw(m_high_dir)

    agg_low = aggregate_rmse(raw_low)
    agg_high = aggregate_rmse(raw_high)
    archs = ordered_archs(set(agg_low) & set(agg_high))

    row_low = {"m": M_LOW}
    row_high = {"m": M_HIGH}
    row_improve = {"m": "Improvement"}

    for arch in archs:
        row_low[ARCH_NAMES[arch]] = fmt(agg_low[arch]["mean"], agg_low[arch]["std"])
        row_high[ARCH_NAMES[arch]] = fmt(agg_high[arch]["mean"], agg_high[arch]["std"])

        improvement_pct = (agg_low[arch]["mean"] - agg_high[arch]["mean"]) / agg_low[arch]["mean"] * 100
        row_improve[ARCH_NAMES[arch]] = f"{improvement_pct:.1f}%"

    # Significance: is NRI-Graph's per-seed *percentage* improvement (m10 -> m180)
    # different from Learned Attention's, via a paired t-test on matched seeds.
    # (Percentage, not raw-meter, deltas — matches the paper's "1.5 ± 0.6% vs.
    # 16.6 ± 1.9%" framing, since NRI and Attention have very different RMSE scales.)
    if "nri" in archs and "attention" in archs:
        low_by_seed = raw_low.set_index(["architecture", "seed"])["rmse_mean"]
        high_by_seed = raw_high.set_index(["architecture", "seed"])["rmse_mean"]
        common_seeds = sorted(
            set(raw_low.loc[raw_low.architecture == "nri", "seed"])
            & set(raw_high.loc[raw_high.architecture == "nri", "seed"])
            & set(raw_low.loc[raw_low.architecture == "attention", "seed"])
            & set(raw_high.loc[raw_high.architecture == "attention", "seed"])
        )
        pct_nri = [(low_by_seed[("nri", s)] - high_by_seed[("nri", s)]) / low_by_seed[("nri", s)] * 100
                   for s in common_seeds]
        pct_attn = [(low_by_seed[("attention", s)] - high_by_seed[("attention", s)]) / low_by_seed[("attention", s)] * 100
                    for s in common_seeds]
        _, p_value = ttest_rel(pct_nri, pct_attn)
        stars = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
        if stars:
            row_improve[ARCH_NAMES["nri"]] += stars
        print(f"\nPaired t-test (NRI vs. Attention per-seed % improvement, n={len(common_seeds)} seeds): "
              f"p = {p_value:.2e}{' ' + stars if stars else ''}")

    table = pd.DataFrame([row_low, row_high, row_improve])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table2.csv", index=False)
    with open(out / "table2.tex", "w") as f:
        f.write(table.to_latex(index=False, escape=False,
                                caption="Data scaling: RMSE (m) as a function of training scenarios m (CFD domain).",
                                label="tab:data-scaling"))

    print("=" * 70)
    print("TABLE 2: DATA SCALING")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved to {out}/table2.csv, {out}/table2.tex")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                         help="No independent experiment for Table 2 — run make_table7.py --run "
                              "first (it owns the m-sweep), then make_table2.py reads the same m10/m180 runs.")
    parser.add_argument("--m-low-dir", type=str, default=f"results/table7/m{M_LOW}")
    parser.add_argument("--m-high-dir", type=str, default=f"results/table7/m{M_HIGH}")
    parser.add_argument("--output-dir", type=str, default="results/table2")
    args = parser.parse_args()

    if args.run:
        print("Table 2 has no independent --run step; run make_table7.py --run first, "
              "then make_table2.py will read the same m10/m180 data.")
        raise SystemExit(1)

    combine(args.m_low_dir, args.m_high_dir, args.output_dir)


if __name__ == "__main__":
    main()
