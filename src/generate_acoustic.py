#!/usr/bin/env python3
"""
Synthetic Acoustic Source Localisation Domain
==============================================

Generates scenarios for a 2D room where a sound/heat source emits energy that
propagates via diffusion.  The physics naturally produces the same structure-
transfer mismatch observed in the hydrogen CFD domain:

  - **Dynamics structure**: which sensors co-vary during transient propagation
    (determined by wave paths, wall reflections, diffusion corridors)
  - **Localisation structure**: which sensors are spatially informative about
    the source position (determined by geometric coverage and conditional
    independence given source location)

Scenario matrix (mirrors the CFD setup):
  - 12 source positions  (4 corner + 4 wall-adjacent + 4 interior)
  - 5 source powers       (analogous to leak rate)
  - 3 wall absorption coefficients (analogous to ventilation rate — OOD variable)
  = 180 scenarios total

Each scenario produces:
  - 60-timestep temporal sequence at every grid point  (for NRI training)
  - Steady-state field                                  (for localisation)

Output format: pickle files with the same dict structure expected by the
existing pipeline (see acoustic_loader.py for the adapter).

Usage:
    python src/generate_acoustic.py                          # full 180 scenarios
    python src/generate_acoustic.py --n_positions 4 --quick  # 60 quick scenarios
    python src/generate_acoustic.py --out my_data/           # custom output dir
"""

import argparse
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple


# ============================================================================
# Physical constants and domain
# ============================================================================

# Room geometry (meters)
ROOM_LENGTH = 20.0    # x-dimension
ROOM_WIDTH  = 15.0    # y-dimension

# Grid resolution
NX = 100              # grid cells in x
NY = 75               # grid cells in y
DX = ROOM_LENGTH / NX
DY = ROOM_WIDTH / NY

# Temporal parameters
N_TIMESTEPS = 60      # match CFD domain
DT = 0.1              # time step (seconds) — covers 6s of transient
DIFFUSIVITY = 0.8     # base diffusion coefficient (m^2/s)

# Advection (analogous to ventilation-driven flow in parking garage)
# Creates directional transport that makes dynamics structure anisotropic
ADVECTION_VX = 0.5    # m/s drift in x (gentle cross-room flow)
ADVECTION_VY = 0.2    # m/s drift in y

# Source parameters
SOURCE_RADIUS = 0.5   # Gaussian source blob radius (meters)

# Source powers (analogous to leak rates: 1, 30, 50, 100, 150 g/s)
SOURCE_POWERS = [0.1, 1.0, 3.0, 7.0, 15.0]

# Wall absorption coefficients (analogous to ventilation ACH: 3, 6, 10)
# Higher absorption = more energy lost at walls = analogous to higher ventilation
ABSORPTION_COEFFS = [0.05, 0.20, 0.50]

# Obstacle configuration: two rectangular pillars (analogous to parking columns)
OBSTACLES = [
    {"x_min": 6.0, "x_max": 7.0, "y_min": 5.0, "y_max": 6.0},
    {"x_min": 13.0, "x_max": 14.0, "y_min": 9.0, "y_max": 10.0},
]


def get_source_positions(n_positions: int = 12) -> np.ndarray:
    """
    Generate source positions distributed across the room.
    Layout: 4 corner + 4 wall-adjacent + 4 interior (for n=12).
    For n<12, subsample uniformly.
    """
    # Full 12-position layout — kept away from obstacles and walls
    margin = 2.0  # meters from wall (increased from 1.5)
    positions_12 = np.array([
        # 4 corners (inside margin)
        [margin, margin],
        [margin, ROOM_WIDTH - margin],
        [ROOM_LENGTH - margin, margin],
        [ROOM_LENGTH - margin, ROOM_WIDTH - margin],
        # 4 wall-adjacent (mid-wall)
        [ROOM_LENGTH / 2, margin],
        [ROOM_LENGTH / 2, ROOM_WIDTH - margin],
        [margin, ROOM_WIDTH / 2],
        [ROOM_LENGTH - margin, ROOM_WIDTH / 2],
        # 4 interior — positioned AWAY from obstacles
        # Obstacle 1: (6-7, 5-6), Obstacle 2: (13-14, 9-10)
        [4.0, 3.5],     # left-bottom interior, clear of obstacle 1
        [16.0, 3.5],    # right-bottom interior, clear of obstacle 2
        [4.0, 11.5],    # left-top interior
        [16.0, 11.5],   # right-top interior
    ])

    if n_positions >= 12:
        return positions_12
    else:
        # Subsample: take every k-th position
        idx = np.linspace(0, 11, n_positions, dtype=int)
        return positions_12[idx]


def create_obstacle_mask() -> np.ndarray:
    """
    Returns a boolean mask (NX, NY) where True = obstacle (no propagation).
    """
    mask = np.zeros((NX, NY), dtype=bool)
    x_coords = np.linspace(DX / 2, ROOM_LENGTH - DX / 2, NX)
    y_coords = np.linspace(DY / 2, ROOM_WIDTH - DY / 2, NY)

    for obs in OBSTACLES:
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                if obs["x_min"] <= x <= obs["x_max"] and obs["y_min"] <= y <= obs["y_max"]:
                    mask[i, j] = True
    return mask


def gaussian_source(
    source_pos: np.ndarray,
    power: float,
) -> np.ndarray:
    """
    Creates a 2D Gaussian source field on the grid.

    Args:
        source_pos: (2,) array [x, y] in meters
        power: source strength

    Returns:
        (NX, NY) source term field
    """
    x = np.linspace(DX / 2, ROOM_LENGTH - DX / 2, NX)
    y = np.linspace(DY / 2, ROOM_WIDTH - DY / 2, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")

    r2 = (X - source_pos[0]) ** 2 + (Y - source_pos[1]) ** 2
    S = power * np.exp(-r2 / (2 * SOURCE_RADIUS ** 2))
    S /= 2 * np.pi * SOURCE_RADIUS ** 2  # normalise so integral ≈ power
    return S


def solve_diffusion(
    source: np.ndarray,
    absorption: float,
    obstacle_mask: np.ndarray,
    n_steps: int = N_TIMESTEPS,
    return_trajectory: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the 2D advection-diffusion equation robustly using:
      1. Sparse linear solve for the steady state
      2. Exponential build-up for the temporal trajectory

    PDE:  ∂u/∂t = D ∇²u − v·∇u + S(x,y) − α·u

    Steady state:  (α − D∇² + v·∇) u_ss = S

    The temporal trajectory is approximated as:
      u(t) = u_ss * (1 − exp(−λ(x)·t))
    where λ(x) is a spatially-varying rate that depends on distance from source
    and the local Péclet number.  This gives physically realistic transients
    where nearby sensors rise first and distant sensors rise later, creating
    the temporal correlations NRI needs.

    Args:
        source: (NX, NY) source term
        absorption: bulk absorption coefficient
        obstacle_mask: (NX, NY) boolean mask
        n_steps: number of time steps
        return_trajectory: if True, return all timesteps

    Returns:
        trajectory: (n_steps, NX, NY) if return_trajectory, else (1, NX, NY)
        steady_state: (NX, NY) final field
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    N = NX * NY
    obs_flat = obstacle_mask.flatten()

    # Build sparse operator: (α − D∇² + v·∇)
    # Using linear index k = i * NY + j
    diag_main = np.ones(N) * (absorption + 2 * DIFFUSIVITY / DX**2 + 2 * DIFFUSIVITY / DY**2)
    diag_ip = np.zeros(N)  # i+1 (right)
    diag_im = np.zeros(N)  # i-1 (left)
    diag_jp = np.zeros(N)  # j+1 (up)
    diag_jm = np.zeros(N)  # j-1 (down)

    for i in range(NX):
        for j in range(NY):
            k = i * NY + j

            if obs_flat[k]:
                # Obstacle: force u=0
                diag_main[k] = 1.0
                continue

            # Boundary handling: enhanced absorption at walls
            if i == 0 or i == NX - 1 or j == 0 or j == NY - 1:
                diag_main[k] += absorption * 3.0  # walls absorb more

            # Diffusion stencil
            if i > 0 and not obs_flat[(i-1)*NY + j]:
                diag_im[k] = -DIFFUSIVITY / DX**2
            else:
                diag_main[k] -= DIFFUSIVITY / DX**2  # zero-flux BC adjustment

            if i < NX-1 and not obs_flat[(i+1)*NY + j]:
                diag_ip[k] = -DIFFUSIVITY / DX**2
            else:
                diag_main[k] -= DIFFUSIVITY / DX**2

            if j > 0 and not obs_flat[i*NY + (j-1)]:
                diag_jm[k] = -DIFFUSIVITY / DY**2
            else:
                diag_main[k] -= DIFFUSIVITY / DY**2

            if j < NY-1 and not obs_flat[i*NY + (j+1)]:
                diag_jp[k] = -DIFFUSIVITY / DY**2
            else:
                diag_main[k] -= DIFFUSIVITY / DY**2

            # Upwind advection
            if ADVECTION_VX >= 0:
                diag_main[k] += ADVECTION_VX / DX
                if i > 0 and not obs_flat[(i-1)*NY + j]:
                    diag_im[k] -= ADVECTION_VX / DX
            else:
                diag_main[k] -= ADVECTION_VX / DX
                if i < NX-1 and not obs_flat[(i+1)*NY + j]:
                    diag_ip[k] += ADVECTION_VX / DX

            if ADVECTION_VY >= 0:
                diag_main[k] += ADVECTION_VY / DY
                if j > 0 and not obs_flat[i*NY + (j-1)]:
                    diag_jm[k] -= ADVECTION_VY / DY
            else:
                diag_main[k] -= ADVECTION_VY / DY
                if j < NY-1 and not obs_flat[i*NY + (j+1)]:
                    diag_jp[k] += ADVECTION_VY / DY

    # Assemble sparse matrix
    diags = [diag_main, diag_ip[:-NY], diag_im[NY:], diag_jp[:-1], diag_jm[1:]]
    offsets = [0, NY, -NY, 1, -1]
    A = sparse.diags(diags, offsets, shape=(N, N), format="csr")

    # Solve steady state: A @ u_ss = source
    rhs = source.flatten()
    rhs[obs_flat] = 0.0
    u_ss_flat = spsolve(A, rhs)
    np.clip(u_ss_flat, 0.0, None, out=u_ss_flat)
    steady_state = u_ss_flat.reshape(NX, NY)

    if not return_trajectory:
        return steady_state[np.newaxis], steady_state

    # Generate temporal trajectory via spatially-varying exponential build-up
    # Rate depends on distance from source peak and local parameters
    source_peak = np.unravel_index(source.argmax(), source.shape)
    x_grid = np.arange(NX) * DX
    y_grid = np.arange(NY) * DY
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    dist_from_source = np.sqrt(
        (X - x_grid[source_peak[0]])**2 + (Y - y_grid[source_peak[1]])**2
    )

    # Characteristic rate: nearby cells respond fast, distant cells respond slow
    # λ(x) = absorption + D / (dist + ε)^2  (diffusion timescale)
    eps = 1.0  # smoothing length
    rate = absorption + DIFFUSIVITY / (dist_from_source + eps)**2

    # Advection shifts the rate field: downstream responds sooner
    advection_shift = (ADVECTION_VX * (X - x_grid[source_peak[0]])
                       + ADVECTION_VY * (Y - y_grid[source_peak[1]]))
    rate += np.clip(advection_shift * 0.05, 0, None)  # downstream boost

    # Build trajectory
    times = np.arange(1, n_steps + 1) * DT
    trajectory = np.zeros((n_steps, NX, NY), dtype=np.float64)
    for t_idx, t in enumerate(times):
        trajectory[t_idx] = steady_state * (1.0 - np.exp(-rate * t))

    # Add small noise to make temporal dynamics more realistic
    rng = np.random.RandomState(hash(str(source.sum())) % 2**31)
    noise_scale = 0.02 * steady_state.max()
    for t_idx in range(n_steps):
        noise = rng.normal(0, noise_scale, (NX, NY)) * (trajectory[t_idx] > 1e-10)
        trajectory[t_idx] = np.clip(trajectory[t_idx] + noise, 0, None)

    return trajectory, steady_state


def generate_scenario(
    source_pos: np.ndarray,
    power: float,
    absorption: float,
    obstacle_mask: np.ndarray,
    scenario_id: str,
) -> Dict:
    """
    Generate a single scenario.

    Returns a dict with the same structure expected by the pipeline:
        - 'field_trajectory': (60, NX, NY) temporal sequence
        - 'field_steady':     (NX, NY) steady-state field
        - 'source_pos':       (2,) source location in meters
        - 'source_power':     float
        - 'absorption':       float
        - 'scenario_id':      str
        - 'domain':           dict with room geometry info
        - 'grid_x':           (NX,) x-coordinates of grid centres
        - 'grid_y':           (NY,) y-coordinates of grid centres
    """
    source_field = gaussian_source(source_pos, power)
    trajectory, steady = solve_diffusion(source_field, absorption, obstacle_mask)

    grid_x = np.linspace(DX / 2, ROOM_LENGTH - DX / 2, NX)
    grid_y = np.linspace(DY / 2, ROOM_WIDTH - DY / 2, NY)

    return {
        "field_trajectory": trajectory.astype(np.float32),
        "field_steady": steady.astype(np.float32),
        "source_pos": source_pos.astype(np.float32),
        "source_power": float(power),
        "absorption": float(absorption),
        "scenario_id": scenario_id,
        "domain": {
            "length": ROOM_LENGTH,
            "width": ROOM_WIDTH,
            "nx": NX,
            "ny": NY,
            "dx": DX,
            "dy": DY,
            "n_timesteps": N_TIMESTEPS,
            "dt": DT,
            "diffusivity": DIFFUSIVITY,
            "obstacles": OBSTACLES,
        },
        "grid_x": grid_x.astype(np.float32),
        "grid_y": grid_y.astype(np.float32),
    }


def generate_all_scenarios(
    n_positions: int = 12,
    powers: List[float] = None,
    absorptions: List[float] = None,
    output_dir: str = "synthetic_data",
    quick: bool = False,
) -> List[Dict]:
    """
    Generate the full scenario matrix.

    Args:
        n_positions: number of source positions (default 12)
        powers: list of source powers (default: SOURCE_POWERS)
        absorptions: list of absorption coefficients (default: ABSORPTION_COEFFS)
        output_dir: where to save individual scenario pickles
        quick: if True, skip trajectory (only steady state) for speed

    Returns:
        List of scenario dicts
    """
    if powers is None:
        powers = SOURCE_POWERS
    if absorptions is None:
        absorptions = ABSORPTION_COEFFS

    positions = get_source_positions(n_positions)
    obstacle_mask = create_obstacle_mask()

    os.makedirs(output_dir, exist_ok=True)

    scenarios = []
    total = n_positions * len(powers) * len(absorptions)
    count = 0

    print(f"Generating {total} scenarios ({n_positions} positions × "
          f"{len(powers)} powers × {len(absorptions)} absorptions)")
    print(f"Domain: {ROOM_LENGTH}m × {ROOM_WIDTH}m, grid {NX}×{NY}")
    print(f"Output: {output_dir}/")

    for pi, pos in enumerate(positions):
        for qi, power in enumerate(powers):
            for ai, absorption in enumerate(absorptions):
                count += 1
                sid = f"abs{absorption:.2f}_pow{power:.1f}_p{pi + 1}"

                if count % 10 == 0 or count == total:
                    print(f"  [{count:3d}/{total}] {sid}")

                scenario = generate_scenario(
                    source_pos=pos,
                    power=power,
                    absorption=absorption,
                    obstacle_mask=obstacle_mask,
                    scenario_id=sid,
                )

                if quick:
                    # Drop trajectory to save memory/time
                    scenario["field_trajectory"] = scenario["field_steady"][np.newaxis]

                scenarios.append(scenario)

    # Save individual scenario pickles (mirrors cfd_data/ structure)
    for sc in scenarios:
        fpath = os.path.join(output_dir, f"{sc['scenario_id']}.pkl")
        with open(fpath, "wb") as f:
            pickle.dump(sc, f)

    # Save combined file for easy loading
    combined_path = os.path.join(output_dir, "all_scenarios.pkl")
    with open(combined_path, "wb") as f:
        pickle.dump(scenarios, f)

    # Save metadata
    meta = {
        "n_scenarios": len(scenarios),
        "n_positions": n_positions,
        "powers": powers,
        "absorptions": absorptions,
        "room_length": ROOM_LENGTH,
        "room_width": ROOM_WIDTH,
        "grid_nx": NX,
        "grid_ny": NY,
        "n_timesteps": N_TIMESTEPS,
        "source_positions": positions.tolist(),
        "obstacle_mask_sum": int(obstacle_mask.sum()),
    }
    meta_path = os.path.join(output_dir, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved {len(scenarios)} scenarios to {output_dir}/")
    print(f"  Individual: {output_dir}/<scenario_id>.pkl")
    print(f"  Combined:   {combined_path}")
    print(f"  Metadata:   {meta_path}")

    return scenarios


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic acoustic source localisation scenarios"
    )
    parser.add_argument(
        "--out", default="synthetic_data",
        help="Output directory (default: synthetic_data)"
    )
    parser.add_argument(
        "--n_positions", type=int, default=12,
        help="Number of source positions (default: 12)"
    )
    parser.add_argument(
        "--powers", type=float, nargs="+", default=None,
        help="Source powers (default: 0.5 5.0 10.0 20.0 50.0)"
    )
    parser.add_argument(
        "--absorptions", type=float, nargs="+", default=None,
        help="Absorption coefficients (default: 0.05 0.20 0.50)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip temporal trajectory (only generate steady state)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    scenarios = generate_all_scenarios(
        n_positions=args.n_positions,
        powers=args.powers,
        absorptions=args.absorptions,
        output_dir=args.out,
        quick=args.quick,
    )

    # Print summary statistics
    fields = np.array([sc["field_steady"] for sc in scenarios])
    print(f"\nField statistics:")
    print(f"  Min:  {fields.min():.4f}")
    print(f"  Max:  {fields.max():.4f}")
    print(f"  Mean: {fields.mean():.4f}")
    print(f"  Std:  {fields.std():.4f}")


if __name__ == "__main__":
    main()
