"""Shared helpers for the make_table{1,2,3,4,7}.py scripts.

All of these read a run directory's table1_results_raw.csv (one row per
architecture x seed, written by run_localizer_experiment.py) and need the
same mean/std-over-seeds aggregation per architecture before formatting a
paper table.
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

ARCH_ORDER = ["mlp", "greens", "attention", "nri"]
ARCH_NAMES = {
    "mlp": "MLP (no structure)",
    "greens": "Green's Function",
    "attention": "Learned Attention",
    "nri": "NRI-Graph",
}


def load_raw(run_dir: Union[str, Path]) -> pd.DataFrame:
    """Load a run directory's per-(architecture, seed) raw results."""
    path = Path(run_dir) / "table1_results_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No table1_results_raw.csv in {run_dir} — has this run's data been "
            f"migrated, or does it still need to be run?"
        )
    return pd.read_csv(path)


def aggregate_rmse(raw_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Aggregate rmse_mean across seeds, per architecture.

    Returns {arch: {"mean": ..., "std": ...}} for whichever architectures are
    present in raw_df, keyed by the short code (mlp/greens/attention/nri).
    """
    out = {}
    for arch, group in raw_df.groupby("architecture"):
        out[arch] = {
            "mean": float(group["rmse_mean"].mean()),
            "std": float(0.0 if len(group) < 2 else group["rmse_mean"].std()),
        }
    return out


def fmt(mean: float, std: float, decimals: int = 2) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def ordered_archs(present):
    """Return architecture codes from ARCH_ORDER that are present in `present`."""
    return [a for a in ARCH_ORDER if a in present]
