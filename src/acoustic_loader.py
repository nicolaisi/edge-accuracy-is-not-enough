#!/usr/bin/env python3
"""
Acoustic Data Loader — CFD-Pipeline Compatible
================================================

Converts synthetic acoustic scenarios into the *exact same* dict structure
that ``load_cfd_netcdf.py`` produces, so every downstream script works
without modification.

Key adaptations:
  - 2D field → 3D by adding thin dummy z-axis (nz=2, z ∈ [0, 0.1])
  - Intensity → "ppm" scale  (× 1e5 so peak ~550 000 ppm, matching CFD)
  - source_pos (x,y) → leak_location (x, y, 0.0)
  - absorption → ventilation (OOD split variable)
  - source_power → leak_rate

Usage:
    # Standalone test
    python src/acoustic_loader.py --data_dir synthetic_data/

    # From run_localizer_experiment.py
    from acoustic_loader import load_acoustic_scenarios_as_cfd
    scenarios = load_acoustic_scenarios_as_cfd("synthetic_data/")
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional


# ── Scale mappings ──────────────────────────────────────────────────────────

# Acoustic intensity peaks at ~5.5; CFD concentrations peak at ~500 000 ppm.
INTENSITY_TO_PPM = 1e5

# absorption α → ventilation ACH (for OOD split compatibility)
ABSORPTION_TO_ACH = {0.05: 3, 0.20: 6, 0.50: 10}

# source_power → leak_rate (arbitrary but consistent)
POWER_TO_LEAK_RATE = {0.1: 1, 1.0: 30, 3.0: 50, 7.0: 100, 15.0: 150}


def _to_3d(field_2d: np.ndarray) -> np.ndarray:
    """(NX, NY) → (NX, NY, 2) by duplicating along dummy z."""
    return np.stack([field_2d, field_2d], axis=-1)


def _map_nearest(val: float, mapping: dict):
    return mapping[min(mapping, key=lambda k: abs(k - val))]


def _convert_one(raw: Dict, idx: int, pos_lookup: Dict) -> Dict:
    """Convert one raw acoustic scenario dict → CFD-pipeline format."""

    grid_x = raw["grid_x"].astype(np.float64)
    grid_y = raw["grid_y"].astype(np.float64)
    grid_z = np.array([0.0, 0.1], dtype=np.float64)

    conc_3d = _to_3d(raw["field_steady"]).astype(np.float64) * INTENSITY_TO_PPM

    src = raw["source_pos"]
    leak_loc = (float(src[0]), float(src[1]), 0.0)

    absorption = float(raw["absorption"])
    power = float(raw["source_power"])
    ventilation = _map_nearest(absorption, ABSORPTION_TO_ACH)
    leak_rate = _map_nearest(power, POWER_TO_LEAK_RATE)

    pos_key = (round(src[0], 1), round(src[1], 1))
    position_id = pos_lookup.get(pos_key, idx % 12 + 1)

    domain = raw["domain"]
    name = f"ach{ventilation}m{leak_rate}p{position_id}"

    out = {
        "id": idx,
        "name": name,
        "concentration_field": conc_3d,
        "grid": {"x": grid_x, "y": grid_y, "z": grid_z},
        "leak_location": leak_loc,
        "leak_rate": leak_rate,
        "ventilation": ventilation,
        "position_id": position_id,
        "time": float(domain["n_timesteps"] * domain["dt"]),
        "timestep": domain["n_timesteps"] - 1,
        "n_timesteps": domain["n_timesteps"],
    }

    # Optional: trajectory for NRI training
    if "field_trajectory" in raw:
        out["field_trajectory_3d"] = (
            np.stack([raw["field_trajectory"], raw["field_trajectory"]], axis=-1).astype(np.float64)
            * INTENSITY_TO_PPM
        )

    return out


# ── Public API ──────────────────────────────────────────────────────────────

def load_acoustic_scenarios_as_cfd(
    data_dir: str = "synthetic_data",
    n_scenarios: Optional[int] = None,
) -> List[Dict]:
    """
    Drop-in replacement for ``load_cfd_from_netcdf_quick()``.

    Args:
        data_dir:     path to directory containing acoustic .pkl files
        n_scenarios:  max scenarios to load (None = all)

    Returns:
        List[Dict] with same keys as load_cfd_netcdf output
    """
    combined = os.path.join(data_dir, "all_scenarios.pkl")

    if os.path.exists(combined):
        with open(combined, "rb") as f:
            raws = pickle.load(f)
    else:
        raws = []
        for fn in sorted(os.listdir(data_dir)):
            if fn.endswith(".pkl") and fn != "metadata.pkl":
                with open(os.path.join(data_dir, fn), "rb") as f:
                    raws.append(pickle.load(f))

    if not raws:
        raise FileNotFoundError(f"No scenario .pkl files in {data_dir}")

    if n_scenarios is not None:
        raws = raws[:n_scenarios]

    # Build position-id lookup
    uniq = []
    for r in raws:
        p = (round(float(r["source_pos"][0]), 1), round(float(r["source_pos"][1]), 1))
        if p not in uniq:
            uniq.append(p)
    pos_lookup = {p: i + 1 for i, p in enumerate(uniq)}

    scenarios = [_convert_one(r, i, pos_lookup) for i, r in enumerate(raws)]

    # Summary
    s0 = scenarios[0]
    achs = sorted({s["ventilation"] for s in scenarios})
    lrs = sorted({s["leak_rate"] for s in scenarios})
    pids = sorted({s["position_id"] for s in scenarios})
    print(f"\n{'='*70}")
    print(f"ACOUSTIC SCENARIOS (CFD-compatible format)")
    print(f"{'='*70}")
    print(f"  Total:       {len(scenarios)}")
    print(f"  Grid:        {len(s0['grid']['x'])}×{len(s0['grid']['y'])}×{len(s0['grid']['z'])}")
    print(f"  Bounds:      X=[{s0['grid']['x'].min():.1f},{s0['grid']['x'].max():.1f}]  "
          f"Y=[{s0['grid']['y'].min():.1f},{s0['grid']['y'].max():.1f}]  "
          f"Z=[{s0['grid']['z'].min():.1f},{s0['grid']['z'].max():.1f}]")
    c = np.array([s["concentration_field"].max() for s in scenarios])
    print(f"  Conc range:  [{c.min():.0f}, {c.max():.0f}] ppm")
    print(f"  Ventilation: {achs}  (mapped from absorption)")
    print(f"  Leak rates:  {lrs}  (mapped from source power)")
    print(f"  Positions:   {len(pids)} unique")
    print(f"{'='*70}\n")

    return scenarios


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="synthetic_data")
    p.add_argument("--n_scenarios", type=int, default=None)
    args = p.parse_args()

    scenarios = load_acoustic_scenarios_as_cfd(args.data_dir, args.n_scenarios)

    s = scenarios[0]
    needed = ["grid", "concentration_field", "leak_location",
              "ventilation", "leak_rate", "position_id", "name"]
    missing = [k for k in needed if k not in s]
    print(f"{'✓' if not missing else '✗'} Required keys: {needed}")
    if missing:
        print(f"  MISSING: {missing}")
    print(f"  concentration_field.shape = {s['concentration_field'].shape}")
    print(f"  leak_location = {s['leak_location']}")
    print(f"  name = {s['name']}")
