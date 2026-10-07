#!/usr/bin/env python3
"""
Visualise synthetic acoustic scenarios for sanity checking.

Produces a multi-panel figure showing:
  1. Steady-state concentration field with source and sensor positions
  2. Temporal evolution at selected sensors (for NRI input)
  3. Comparison across absorption coefficients (OOD variable)
  4. Source position layout

Usage:
    python src/visualize_acoustic.py                          # default
    python src/visualize_acoustic.py --data_dir my_data/      # custom dir
    python src/visualize_acoustic.py --save diagnostic.pdf    # save to file
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_scenarios(data_dir: str, n_max: int = None):
    combined = os.path.join(data_dir, "all_scenarios.pkl")
    if os.path.exists(combined):
        with open(combined, "rb") as f:
            scenarios = pickle.load(f)
    else:
        scenarios = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".pkl") and fname not in ("all_scenarios.pkl", "metadata.pkl"):
                with open(os.path.join(data_dir, fname), "rb") as f:
                    scenarios.append(pickle.load(f))
    if n_max:
        scenarios = scenarios[:n_max]
    return scenarios


def plot_diagnostic(scenarios, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Synthetic Acoustic Domain — Diagnostic", fontsize=14, fontweight="bold")

    domain = scenarios[0]["domain"]
    obstacles = domain["obstacles"]

    # --- Panel 1: Example steady-state field ---
    ax = axes[0, 0]
    # Pick a mid-power, mid-absorption scenario
    mid_idx = len(scenarios) // 2
    sc = scenarios[mid_idx]
    im = ax.imshow(
        sc["field_steady"].T, origin="lower", aspect="auto",
        extent=[0, domain["length"], 0, domain["width"]],
        cmap="hot",
    )
    ax.plot(*sc["source_pos"], "c*", markersize=15, label="Source")
    for obs in obstacles:
        rect = Rectangle(
            (obs["x_min"], obs["y_min"]),
            obs["x_max"] - obs["x_min"],
            obs["y_max"] - obs["y_min"],
            facecolor="gray", edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
    ax.set_title(f"Steady state (P={sc['source_power']}, α={sc['absorption']})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Intensity")

    # --- Panel 2: All source positions ---
    ax = axes[0, 1]
    positions = np.array([sc["source_pos"] for sc in scenarios])
    unique_pos = np.unique(positions, axis=0)
    ax.scatter(unique_pos[:, 0], unique_pos[:, 1], c="red", s=100, marker="*", zorder=5)
    for i, p in enumerate(unique_pos):
        ax.annotate(f"P{i+1}", (p[0] + 0.3, p[1] + 0.3), fontsize=8)
    for obs in obstacles:
        rect = Rectangle(
            (obs["x_min"], obs["y_min"]),
            obs["x_max"] - obs["x_min"],
            obs["y_max"] - obs["y_min"],
            facecolor="lightgray", edgecolor="black",
        )
        ax.add_patch(rect)
    ax.set_xlim(0, domain["length"])
    ax.set_ylim(0, domain["width"])
    ax.set_title(f"{len(unique_pos)} Source Positions")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Temporal evolution at 3 sensors ---
    ax = axes[0, 2]
    from acoustic_loader import sample_sensor_positions, interpolate_field_at_positions
    sensors = sample_sensor_positions(15, domain["length"], domain["width"])
    sc_traj = scenarios[mid_idx]
    ts = interpolate_field_at_positions(
        sc_traj["field_trajectory"], sensors,
        sc_traj["grid_x"], sc_traj["grid_y"],
    )
    t_axis = np.arange(ts.shape[0]) * domain["dt"]
    for si in [0, 5, 10]:
        ax.plot(t_axis, ts[:, si], label=f"Sensor {si}")
    ax.set_title("Temporal Evolution (NRI input)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity at sensor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Absorption comparison (OOD variable) ---
    ax = axes[1, 0]
    # Find scenarios with same source position & power but different absorption
    ref_pos = scenarios[0]["source_pos"]
    ref_pow = scenarios[0]["source_power"]
    matched = [s for s in scenarios
               if np.allclose(s["source_pos"], ref_pos) and s["source_power"] == ref_pow]
    for sc_match in sorted(matched, key=lambda s: s["absorption"]):
        # Take a horizontal slice through the middle
        ny_mid = sc_match["field_steady"].shape[1] // 2
        profile = sc_match["field_steady"][:, ny_mid]
        x = np.linspace(0, domain["length"], len(profile))
        ax.plot(x, profile, label=f"α = {sc_match['absorption']:.2f}")
    ax.set_title(f"Absorption comparison (source P1, power={ref_pow})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Intensity (y = mid)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 5: Power comparison ---
    ax = axes[1, 1]
    ref_abs = scenarios[0]["absorption"]
    matched_pow = [s for s in scenarios
                   if np.allclose(s["source_pos"], ref_pos) and s["absorption"] == ref_abs]
    for sc_match in sorted(matched_pow, key=lambda s: s["source_power"]):
        ny_mid = sc_match["field_steady"].shape[1] // 2
        profile = sc_match["field_steady"][:, ny_mid]
        x = np.linspace(0, domain["length"], len(profile))
        ax.plot(x, profile, label=f"P = {sc_match['source_power']:.1f}")
    ax.set_title(f"Power comparison (source P1, α={ref_abs})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Intensity (y = mid)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 6: Field statistics summary ---
    ax = axes[1, 2]
    maxvals = [sc["field_steady"].max() for sc in scenarios]
    meanvals = [sc["field_steady"].mean() for sc in scenarios]
    absorptions = [sc["absorption"] for sc in scenarios]
    scatter = ax.scatter(meanvals, maxvals, c=absorptions, cmap="viridis", alpha=0.6, s=20)
    ax.set_xlabel("Mean field intensity")
    ax.set_ylabel("Max field intensity")
    ax.set_title("Field statistics by scenario")
    plt.colorbar(scatter, ax=ax, label="Absorption α")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="synthetic_data")
    parser.add_argument("--save", default=None, help="Save path (e.g. diagnostic.pdf)")
    args = parser.parse_args()

    scenarios = load_scenarios(args.data_dir)
    plot_diagnostic(scenarios, save_path=args.save)
