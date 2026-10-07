#!/usr/bin/env python3
"""
Table 3: Transfer outcomes vs. structural similarity to the task-optimal graph.

Paper format:
    Domain | Structure | RMSE | Degradation | Est. J

Degradation is relative to Learned Attention's RMSE in that domain (the
best-performing, task-optimised baseline). Est. J is the Jaccard similarity
of that structure's edges against Learned Attention's edges (used as a proxy
for the unobservable task-optimal graph), read from the *_jaccard.json files
written by visualize_structure_comparison.py.

--run generates those two JSON files (one per domain) if missing, using the
already-trained NRI/attention checkpoints from Table 1 — no retraining.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from table_utils import aggregate_rmse, load_raw

DOMAINS = ["cfd", "acoustic"]
CHECKPOINT_DIRS = {"cfd": "checkpoints/cfd", "acoustic": "checkpoints/acoustic"}


def run(output_root: str, force: bool):
    for domain in DOMAINS:
        out_prefix = f"{output_root}/structure_comparison_{domain}"
        if not force and Path(f"{out_prefix}_jaccard.json").exists():
            print(f"{domain}: Jaccard JSON already exists, skipping (use --force to rerun)")
            continue
        cmd = [
            sys.executable, "src/visualize_structure_comparison.py",
            "--domain", domain,
            "--nri-path", f"{CHECKPOINT_DIRS[domain]}/nri_edge_probs_spatial_180.pt",
            "--attention-model", f"results/table1/{domain}/model_attention_seed0.pt",
            "--n-sensors", "15",
            "--output", out_prefix,
        ]
        print(f"{domain}: running {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def combine(output_root: str):
    rows = []
    for domain in DOMAINS:
        try:
            raw = load_raw(f"results/table1/{domain}")
        except FileNotFoundError as e:
            print(f"{domain}: {e}")
            continue
        agg = aggregate_rmse(raw)
        if "attention" not in agg:
            print(f"{domain}: no Learned Attention row in results/table1/{domain}, skipping")
            continue
        attn_rmse = agg["attention"]["mean"]

        jaccard_path = Path(output_root) / f"structure_comparison_{domain}_jaccard.json"
        if not jaccard_path.exists():
            print(f"{domain}: missing {jaccard_path} — run `python make_table3.py --run` first")
            continue
        with open(jaccard_path) as f:
            jaccard = json.load(f)

        for arch, structure_name, j_key in [
            ("greens", "Green's Function", "j_gl"),
            ("nri", "NRI Learned Graph", "j_nl"),
        ]:
            if arch not in agg:
                continue
            rmse = agg[arch]["mean"]
            degradation = (rmse - attn_rmse) / attn_rmse * 100
            rows.append({
                "Domain": domain.upper() if domain == "cfd" else domain.capitalize(),
                "Structure": structure_name,
                "RMSE": f"{rmse:.2f} m",
                "Degradation": f"{degradation:.0f}%",
                "Est. J": f"{jaccard[j_key]:.2f}",
            })

    table = pd.DataFrame(rows)

    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table3.csv", index=False)
    with open(out / "table3.tex", "w") as f:
        f.write(table.to_latex(index=False, escape=False,
                                caption="Transfer outcomes vs. estimated structural similarity to the task-optimal graph.",
                                label="tab:transfer-jaccard"))

    print("=" * 70)
    print("TABLE 3: TRANSFER OUTCOMES VS. JACCARD SIMILARITY")
    print("=" * 70)
    print(table.to_string(index=False))
    print(f"\nSaved to {out}/table3.csv, {out}/table3.tex")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                         help="Generate the per-domain Jaccard JSON via visualize_structure_comparison.py")
    parser.add_argument("--force", action="store_true", help="With --run, regenerate even if JSON exists")
    parser.add_argument("--output-root", type=str, default="results/table3")
    args = parser.parse_args()

    if args.run:
        run(args.output_root, args.force)

    combine(args.output_root)


if __name__ == "__main__":
    main()
