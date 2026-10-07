#!/usr/bin/env python3
"""
Table 8 (Appendix C.4): NRI ablation over edge threshold and top-k connectivity.

Paper format — one combined table, one row per configuration:
    Configuration | Avg. degree | Sparsity | RMSE (m)

CFD, NRI-Graph only. Unifies what used to be two separate scripts
(run_multi_threshold_experiment.py + run_multi_threshold_experiment_topk.py,
which wrote two separate un-merged tables) into one --run covering both the
tau-sweep and the top-k sweep, and one combine step writing a single
results/table8/table8.csv/.tex. --plot absorbs visualize_all_results.py's
three figures, reading the real combined table and the live Table 1 CFD
baseline instead of a hardcoded dict.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from table_utils import fmt, load_raw

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7]
DEFAULT_TOPKS = [2, 3, 4]


def create_threshold_checkpoint(base_checkpoint: str, tau: float, ckpt_dir: Path) -> Path:
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    edge_prob = checkpoint["edge_prob"].clone()
    edge_prob[edge_prob < tau] = 0.0
    row_sums = edge_prob.sum(dim=1, keepdim=True)
    edge_prob = torch.where(row_sums > 0, edge_prob / (row_sums + 1e-8), edge_prob)

    n_total = edge_prob.numel()
    n_zero = (edge_prob == 0).sum().item()
    sparsity = n_zero / n_total * 100
    avg_degree = (n_total - n_zero) / edge_prob.shape[0]

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"nri_edge_probs_tau_{tau:.1f}.pt"
    torch.save({
        "edge_prob": edge_prob,
        "query_points": checkpoint.get("query_points"),
        "threshold": tau,
        "sparsity": sparsity,
        "avg_degree": avg_degree,
    }, path)
    print(f"tau={tau:.1f}: avg_degree={avg_degree:.1f}, sparsity={sparsity:.1f}% -> {path}")
    return path


def create_topk_checkpoint(base_checkpoint: str, k: int, ckpt_dir: Path) -> Path:
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    edge_prob = checkpoint["edge_prob"].clone()
    n = edge_prob.shape[0]
    edge_prob_topk = torch.zeros_like(edge_prob)
    for i in range(n):
        row = edge_prob[i].clone()
        row[i] = -1e9
        top_idx = torch.topk(row, k=k, dim=0).indices
        edge_prob_topk[i, top_idx] = edge_prob[i, top_idx]
    row_sums = edge_prob_topk.sum(dim=1, keepdim=True)
    edge_prob_topk = torch.where(row_sums > 0, edge_prob_topk / (row_sums + 1e-8), edge_prob_topk)

    n_total = edge_prob_topk.numel()
    n_zero = (edge_prob_topk == 0).sum().item()
    sparsity = n_zero / n_total * 100
    avg_degree = (n_total - n_zero) / n

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"nri_edge_probs_topk_{k}.pt"
    torch.save({
        "edge_prob": edge_prob_topk,
        "query_points": checkpoint.get("query_points"),
        "topk": k,
        "sparsity": sparsity,
        "avg_degree": avg_degree,
    }, path)
    print(f"top-{k}: avg_degree={avg_degree:.1f}, sparsity={sparsity:.1f}% -> {path}")
    return path


def run(output_root: str, base_checkpoint: str, thresholds, topks, data_dir: str, device: str, force: bool):
    out = Path(output_root)
    ckpt_dir = out / "checkpoints"

    for tau in thresholds:
        run_dir = out / f"tau_{tau:.1f}"
        if not force and (run_dir / "table1_results_raw.csv").exists():
            print(f"tau={tau}: already run, skipping (use --force to rerun)")
            continue
        ckpt = create_threshold_checkpoint(base_checkpoint, tau, ckpt_dir)
        cmd = [sys.executable, "src/run_localizer_experiment.py", "--full",
               "--output", str(run_dir), "--data_dir", data_dir, "--device", device,
               "--nri_edge_path", str(ckpt), "--arch", "nri"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    for k in topks:
        run_dir = out / f"topk_{k}"
        if not force and (run_dir / "table1_results_raw.csv").exists():
            print(f"topk={k}: already run, skipping (use --force to rerun)")
            continue
        ckpt = create_topk_checkpoint(base_checkpoint, k, ckpt_dir)
        cmd = [sys.executable, "src/run_localizer_experiment.py", "--full",
               "--output", str(run_dir), "--data_dir", data_dir, "--device", device,
               "--nri_edge_path", str(ckpt), "--arch", "nri"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def _degree_sparsity(ckpt_path: Path):
    if not ckpt_path.exists():
        return float("nan"), float("nan")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("avg_degree", float("nan")), ckpt.get("sparsity", float("nan"))


def combine(output_root: str, thresholds, topks):
    out = Path(output_root)
    ckpt_dir = out / "checkpoints"
    rows = []

    for tau in thresholds:
        run_dir = out / f"tau_{tau:.1f}"
        try:
            raw = load_raw(run_dir)
        except FileNotFoundError as e:
            print(f"tau={tau}: {e}")
            continue
        nri = raw[raw.architecture == "nri"]["rmse_mean"]
        avg_deg, sparsity = _degree_sparsity(ckpt_dir / f"nri_edge_probs_tau_{tau:.1f}.pt")
        rows.append({
            "Configuration": f"τ = {tau:.1f}",
            "Avg. degree": f"{avg_deg:.1f}" if np.isfinite(avg_deg) else "N/A",
            "Sparsity": f"{sparsity:.0f}%" if np.isfinite(sparsity) else "N/A",
            "RMSE (m)": fmt(nri.mean(), 0.0 if len(nri) < 2 else nri.std()),
        })

    for k in topks:
        run_dir = out / f"topk_{k}"
        try:
            raw = load_raw(run_dir)
        except FileNotFoundError as e:
            print(f"topk={k}: {e}")
            continue
        nri = raw[raw.architecture == "nri"]["rmse_mean"]
        avg_deg, sparsity = _degree_sparsity(ckpt_dir / f"nri_edge_probs_topk_{k}.pt")
        rows.append({
            "Configuration": f"Top-{k} connectivity",
            "Avg. degree": f"{avg_deg:.1f}" if np.isfinite(avg_deg) else "N/A",
            "Sparsity": f"{sparsity:.0f}%" if np.isfinite(sparsity) else "N/A",
            "RMSE (m)": fmt(nri.mean(), 0.0 if len(nri) < 2 else nri.std()),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out / "table8.csv", index=False)
    with open(out / "table8.tex", "w") as f:
        f.write(table.to_latex(index=False, escape=False,
                                caption="NRI-Graph ablation over edge threshold and top-k connectivity (CFD domain).",
                                label="tab:nri-ablation"))

    print("=" * 70)
    print("TABLE 8: NRI THRESHOLD / TOP-K ABLATION")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved to {out}/table8.csv, {out}/table8.tex")
    return table


def plot(output_root: str, baseline_dir: str):
    import matplotlib.pyplot as plt

    out = Path(output_root)
    table = pd.read_csv(out / "table8.csv")
    baseline = pd.read_csv(Path(baseline_dir) / "table1.csv")

    def baseline_rmse(name_substr):
        row = baseline[baseline["Localiser"].str.contains(name_substr)] \
            if "Localiser" in baseline.columns else baseline[baseline["Localizer"].str.contains(name_substr)]
        mean, std = row.iloc[0].filter(like="RMSE").values[0].split("±")
        return float(mean), float(std)

    attn_mean, _ = baseline_rmse("Attention")
    greens_mean, _ = baseline_rmse("Green")
    mlp_mean, _ = baseline_rmse("MLP")

    def parse_rmse(s):
        mean, std = s.split("±")
        return float(mean), float(std)

    table[["rmse_mean", "rmse_std"]] = table["RMSE (m)"].apply(lambda s: pd.Series(parse_rmse(s)))
    table["avg_degree"] = pd.to_numeric(table["Avg. degree"], errors="coerce")

    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure: performance vs avg degree
    is_topk = table["Configuration"].str.contains("Top-")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(table.loc[~is_topk, "avg_degree"], table.loc[~is_topk, "rmse_mean"],
                yerr=table.loc[~is_topk, "rmse_std"], fmt="o", color="#E74C3C",
                markersize=8, capsize=4, label="NRI-Graph (threshold)")
    ax.errorbar(table.loc[is_topk, "avg_degree"], table.loc[is_topk, "rmse_mean"],
                yerr=table.loc[is_topk, "rmse_std"], fmt="s", color="#E67E22",
                markersize=8, capsize=4, label="NRI-Graph (top-k)")
    ax.axhline(attn_mean, color="#27AE60", linestyle="--", label="Learned Attention")
    ax.axhline(greens_mean, color="#3498DB", linestyle="--", label="Green's Function")
    ax.axhline(mlp_mean, color="#7F8C8D", linestyle="--", label="MLP (no structure)")
    ax.set_xlabel("Average degree (connectivity)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("NRI-Graph performance vs. graph connectivity")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(figures_dir / "performance_vs_degree.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: threshold sensitivity bar chart
    tau_rows = table[~is_topk].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(tau_rows["Configuration"], tau_rows["rmse_mean"], yerr=tau_rows["rmse_std"],
           color="#E74C3C", alpha=0.8, edgecolor="black", capsize=5)
    ax.axhline(attn_mean, color="#27AE60", linestyle="--", label="Learned Attention")
    ax.axhline(greens_mean, color="#3498DB", linestyle="--", label="Green's Function")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("NRI-Graph threshold sensitivity")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    plt.tight_layout()
    plt.savefig(figures_dir / "threshold_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: comprehensive comparison (all configs + baselines)
    all_rows = pd.concat([
        table[["Configuration", "rmse_mean", "rmse_std"]],
        pd.DataFrame([
            {"Configuration": "Learned Attention", "rmse_mean": attn_mean, "rmse_std": 0.0},
            {"Configuration": "Green's Function", "rmse_mean": greens_mean, "rmse_std": 0.0},
            {"Configuration": "MLP (no structure)", "rmse_mean": mlp_mean, "rmse_std": 0.0},
        ]),
    ]).sort_values("rmse_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(all_rows["Configuration"], all_rows["rmse_mean"], xerr=all_rows["rmse_std"],
            color="#3498DB", alpha=0.8, edgecolor="black", capsize=4)
    ax.set_xlabel("RMSE (m)")
    ax.set_title("Comprehensive comparison: all configurations")
    ax.grid(alpha=0.3, linestyle=":", axis="x")
    plt.tight_layout()
    plt.savefig(figures_dir / "comprehensive_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figures to {figures_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run the threshold + top-k sweeps")
    parser.add_argument("--force", action="store_true", help="With --run, rerun even if results exist")
    parser.add_argument("--plot", action="store_true", help="Generate the three ablation figures")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--topks", type=int, nargs="+", default=DEFAULT_TOPKS)
    parser.add_argument("--nri-checkpoint", type=str, default="checkpoints/cfd/nri_edge_probs_spatial_180.pt")
    parser.add_argument("--data-dir", type=str, default="cfd_data/")
    parser.add_argument("--baseline-dir", type=str, default="results/table1/cfd")
    parser.add_argument("--output-root", type=str, default="results/table8")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.run:
        run(args.output_root, args.nri_checkpoint, args.thresholds, args.topks,
            args.data_dir, args.device, args.force)

    combine(args.output_root, args.thresholds, args.topks)

    if args.plot:
        plot(args.output_root, args.baseline_dir)


if __name__ == "__main__":
    main()
