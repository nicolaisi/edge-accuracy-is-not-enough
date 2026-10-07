#!/usr/bin/env python3
"""
Table 4: Two further dynamics-derived structure baselines.

Paper format:
    Method                       | RMSE (m) | Sparsity | Est. J

Rows: MLP (no structure), NRI-Graph (both reused from Table 1/3), plus
Correlation-Graph and Mutual-Information-Graph (from s10_structure_baselines.py's
sweep — the best-RMSE threshold/percentile is selected, not assumed).

Correlation/MI "Est. J" is not available from the sweep JSON directly (it only
stores graph_stats, not the adjacency matrix) — combine mode re-fits just the
winning config's adjacency (cheap: correlation/MI computation, no training)
and computes its Jaccard similarity against the Learned Attention proxy graph
the same way make_table3.py does, for an apples-to-apples "Est. J" column.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from s10_structure_baselines import CorrelationGraphBuilder, MutualInfoGraphBuilder
from table_utils import aggregate_rmse, load_raw
from visualize_structure_comparison import StructureVisualizer

SUPPLEMENT_DATA = "results/table4/supplement_data.pt"
S10_SWEEP_DIR = "results/table4/s10_sweep"
# Separate, smaller supplement file at n_queries=32 — matching the query space
# Table 1/3's attention/NRI models actually use — needed only so the live
# Jaccard re-fit compares apples to apples. The RMSE/Sparsity columns keep
# using SUPPLEMENT_DATA (n_queries=64) as originally computed; untouched.
JACCARD_SUPPLEMENT_DATA = "results/table4/supplement_data_32q.pt"


def run(force: bool):
    if force or not Path(SUPPLEMENT_DATA).exists():
        print(f"Preparing {SUPPLEMENT_DATA} ...")
        subprocess.run([sys.executable, "src/prepare_supplement_data.py",
                         "--out", SUPPLEMENT_DATA], check=True)
    else:
        print(f"{SUPPLEMENT_DATA} already exists, skipping (use --force to regenerate)")

    if force or not Path(JACCARD_SUPPLEMENT_DATA).exists():
        print(f"Preparing {JACCARD_SUPPLEMENT_DATA} (n_queries=32, for Est. J only) ...")
        subprocess.run([sys.executable, "src/prepare_supplement_data.py",
                         "--out", JACCARD_SUPPLEMENT_DATA, "--n_queries", "32"], check=True)
    else:
        print(f"{JACCARD_SUPPLEMENT_DATA} already exists, skipping (use --force to regenerate)")

    sweep_dir = Path(S10_SWEEP_DIR)
    if force or not (sweep_dir / "correlation_results.json").exists() \
            or not (sweep_dir / "mi_results.json").exists():
        print(f"Running s10_structure_baselines.py -> {sweep_dir} ...")
        sweep_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "src/s10_structure_baselines.py",
                         "--data_path", SUPPLEMENT_DATA, "--method", "both",
                         "--output_dir", str(sweep_dir), "--device", "cpu"], check=True)
    else:
        print(f"{sweep_dir} already has correlation/mi results, skipping (use --force to rerun)")


def best_config(results: dict):
    """Pick the key with the lowest rmse_mean (don't assume it's the first)."""
    return min(results.items(), key=lambda kv: kv[1]["rmse_mean"])


def compute_jaccard_for_adjacency(viz: StructureVisualizer, adjacency: np.ndarray, learned_qq: np.ndarray) -> float:
    """Project a (n_sensors, n_queries) adjacency to query-query space and
    compute Jaccard similarity against an already-projected Learned Attention graph."""
    adj_qq = viz._to_query_query(adjacency)
    edges_a = set(zip(*np.where(adj_qq > 0.01)))
    edges_l = set(zip(*np.where(learned_qq > 0.01)))
    union = edges_a | edges_l
    return len(edges_a & edges_l) / len(union) if union else 0.0


def combine(output_root: str, attention_model: str, table3_json: str):
    out = Path(output_root)

    # --- MLP / NRI-Graph rows, reused from Table 1 (CFD) ---
    cfd_raw = load_raw("results/table1/cfd")
    cfd_agg = aggregate_rmse(cfd_raw)

    with open(table3_json) as f:
        table3_jaccard = json.load(f)  # {"j_gn":.., "j_gl":.., "j_nl":..}

    rows = [
        {"Method": "MLP (no structure)", "RMSE (m)": f"{cfd_agg['mlp']['mean']:.2f} ± {cfd_agg['mlp']['std']:.2f}",
         "Sparsity": "100%", "Est. J": "0.00"},
        {"Method": "NRI-Graph", "RMSE (m)": f"{cfd_agg['nri']['mean']:.2f} ± {cfd_agg['nri']['std']:.2f}",
         "Sparsity": "97%", "Est. J": f"{table3_jaccard['j_nl']:.2f}"},
    ]

    # --- Correlation-Graph / MI-Graph rows ---
    with open(Path(S10_SWEEP_DIR) / "correlation_results.json") as f:
        corr_results = json.load(f)
    with open(Path(S10_SWEEP_DIR) / "mi_results.json") as f:
        mi_results = json.load(f)

    best_corr_key, best_corr = best_config(corr_results)
    best_mi_key, best_mi = best_config(mi_results)
    print(f"Best correlation config: {best_corr_key} (rmse_mean={best_corr['rmse_mean']:.2f})")
    print(f"Best MI config: {best_mi_key} (rmse_mean={best_mi['rmse_mean']:.2f})")

    jaccard_supplement_path = Path(JACCARD_SUPPLEMENT_DATA)
    if not jaccard_supplement_path.exists():
        print(f"Missing {jaccard_supplement_path} — run `python make_table4.py --run` first to compute Est. J "
              f"for Correlation-Graph / MI-Graph. Filling those with N/A.")
        for name, best in [("Correlation-Graph (best τ)", best_corr), ("Mutual-Information-Graph (best pct.)", best_mi)]:
            rows.append({
                "Method": name,
                "RMSE (m)": f"{best['rmse_mean']:.2f} ± {best['rmse_std']:.2f}",
                "Sparsity": f"{best['graph_stats']['sparsity'] * 100:.0f}%",
                "Est. J": "N/A",
            })
    else:
        data = torch.load(jaccard_supplement_path, map_location="cpu", weights_only=False)
        time_series = data["time_series_data"].numpy()
        n_sensors = data["sensor_positions"].shape[0]
        n_queries = data["query_positions"].shape[0]

        viz = StructureVisualizer(facility_dims=(50, 30, 3), n_sensors=n_sensors, n_queries=n_queries)
        learned_attention = viz.extract_learned_attention(attention_model)
        learned_qq = viz._to_query_query(learned_attention)

        tau = best_corr["graph_stats"]["threshold"]
        corr_builder = CorrelationGraphBuilder(threshold=tau)
        corr_builder.fit(time_series, n_sensors, n_queries)
        j_corr = compute_jaccard_for_adjacency(viz, corr_builder.get_adjacency_matrix(), learned_qq)

        pct = best_mi["graph_stats"]["percentile"]
        mi_builder = MutualInfoGraphBuilder(percentile=int(pct))
        mi_builder.fit(time_series, n_sensors, n_queries)
        j_mi = compute_jaccard_for_adjacency(viz, mi_builder.get_adjacency_matrix(), learned_qq)

        rows.append({"Method": "Correlation-Graph (best τ)",
                      "RMSE (m)": f"{best_corr['rmse_mean']:.2f} ± {best_corr['rmse_std']:.2f}",
                      "Sparsity": f"{best_corr['graph_stats']['sparsity'] * 100:.0f}%",
                      "Est. J": f"{j_corr:.2f}"})
        rows.append({"Method": "Mutual-Information-Graph (best pct.)",
                      "RMSE (m)": f"{best_mi['rmse_mean']:.2f} ± {best_mi['rmse_std']:.2f}",
                      "Sparsity": f"{best_mi['graph_stats']['sparsity'] * 100:.0f}%",
                      "Est. J": f"{j_mi:.2f}"})

    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table4.csv", index=False)
    with open(out / "table4.tex", "w") as f:
        f.write(table.to_latex(index=False, escape=False,
                                caption="Two additional dynamics-learned structure baselines (CFD domain).",
                                label="tab:additional-baselines"))

    print("=" * 70)
    print("TABLE 4: ADDITIONAL DYNAMICS-DERIVED STRUCTURE BASELINES")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved to {out}/table4.csv, {out}/table4.tex")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                         help="Prepare supplement_data.pt and run the correlation/MI sweep if missing")
    parser.add_argument("--force", action="store_true", help="With --run, regenerate even if outputs exist")
    parser.add_argument("--output-root", type=str, default="results/table4")
    parser.add_argument("--attention-model", type=str, default="results/table1/cfd/model_attention_seed0.pt")
    parser.add_argument("--table3-jaccard-json", type=str,
                         default="results/table3/structure_comparison_cfd_jaccard.json")
    args = parser.parse_args()

    if args.run:
        run(args.force)

    combine(args.output_root, args.attention_model, args.table3_jaccard_json)


if __name__ == "__main__":
    main()
