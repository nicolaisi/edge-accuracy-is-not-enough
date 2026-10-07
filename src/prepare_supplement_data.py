"""
prepare_supplement_data.py

Prepares the preprocessed data file required by:
  - s10_structure_baselines.py  (Table 4: correlation / MI baselines)
  - legacy/s12_joint_training.py (not part of the current paper draft)

Both scripts expect a torch .pt file with keys:
  train_scenarios   : list of scenario dicts (80%)
  val_scenarios     : list of scenario dicts (10%)
  test_scenarios    : list of scenario dicts (10%)
  sensor_positions  : FloatTensor (n_sensors, 3)
  query_positions   : FloatTensor (n_queries,  3)
  time_series_data  : FloatTensor (n_scenarios, n_timesteps, n_sensors + n_queries)

Usage:
  python src/prepare_supplement_data.py --data_dir cfd_data/ --out results/table4/supplement_data.pt
  python src/prepare_supplement_data.py --data_dir cfd_data/ --out results/table4/supplement_data.pt \
      --n_sensors 15 --n_queries 64 --seed 42
"""

import argparse
import numpy as np
import torch
from pathlib import Path

# Reuse the same CFD loader used by run_localizer_experiment.py
from load_cfd_netcdf import load_cfd_from_netcdf_quick

def sample_positions(scenarios, n_positions, seed=42):
    """
    Sample spatial positions from the CFD grid, mirroring the strategy in
    run_localizer_experiment.py: read grid coords directly from the scenario dict
    and subsample uniformly at sensor height (z=0.5 m).
    """
    rng = np.random.default_rng(seed)

    grid = scenarios[0]['grid']   # dict with keys 'x', 'y', 'z' (1-D arrays)
    xs, ys = grid['x'], grid['y']
    z_sensor = 0.5   # fuel-tank height, matching run_localizer_experiment.py

    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    candidates = np.stack([xx.ravel(), yy.ravel(),
                           np.full(xx.size, z_sensor)], axis=1)

    idx = rng.choice(len(candidates), size=n_positions, replace=False)
    return candidates[idx].astype(np.float32)


def extract_time_series(scenarios, sensor_positions, query_positions):
    """
    Extract concentration readings at sensor and query positions for all scenarios.

    s10_structure_baselines.py expects shape (n_scenarios, n_timesteps, n_total).
    Each CFD scenario contributes one snapshot (the final-timestep concentration
    field), so n_timesteps=1 and the 180 scenarios serve as the "time" axis for
    correlation / MI computation.

    Returns FloatTensor of shape (n_scenarios, 1, n_sensors + n_queries).
    """
    from scipy.interpolate import RegularGridInterpolator
    from tqdm import tqdm

    all_positions = np.vstack([sensor_positions, query_positions])  # (N, 3)
    readings_all = []

    print("Extracting concentration readings at sensor/query positions ...")
    for s in tqdm(scenarios):
        grid  = s['grid']
        field = s['concentration_field']   # (nx, ny, nz) — final timestep

        interp = RegularGridInterpolator(
            (grid['x'], grid['y'], grid['z']),
            field, method='linear', bounds_error=False, fill_value=0.0
        )
        readings = interp(all_positions).astype(np.float32)   # (N,)

        # Log-normalise to match train_nri_spatial.py (log1p(x/1000))
        readings = np.log1p(readings / 1000.0)

        readings_all.append(readings)

    # Stack -> (n_scenarios, N), then unsqueeze timestep dim -> (n_scenarios, 1, N)
    arr = np.stack(readings_all, axis=0)[:, np.newaxis, :]
    return torch.tensor(arr, dtype=torch.float32)


def split_scenarios(scenarios, seed=42):
    """80 / 10 / 10 stratified split, mirroring run_localizer_experiment.py."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(scenarios))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    return (
        [scenarios[i] for i in train_idx],
        [scenarios[i] for i in val_idx],
        [scenarios[i] for i in test_idx],
        idx   # keep original order for time_series alignment
    )


def main():
    parser = argparse.ArgumentParser(
        description='Prepare preprocessed data for supplement experiments S9 and S11')
    parser.add_argument('--data_dir',  type=str, default='cfd_data/',
                        help='Directory containing CFD scenario subdirectories')
    parser.add_argument('--out',       type=str, default='results/table4/supplement_data.pt',
                        help='Output .pt file path')
    parser.add_argument('--n_sensors', type=int, default=15,
                        help='Number of sensor positions (default: 15, matching Table 1)')
    parser.add_argument('--n_queries', type=int, default=64,
                        help='Number of query positions (default: 64, matching NRI training)')
    parser.add_argument('--seed',      type=int, default=42,
                        help='Random seed for position sampling and splits')
    parser.add_argument('--n_scenarios', type=int, default=None,
                        help='Limit to first N scenarios (default: all 180)')
    args = parser.parse_args()

    print(f'Loading CFD scenarios from {args.data_dir} ...')
    n_to_load = args.n_scenarios if args.n_scenarios is not None else 180
    scenarios = load_cfd_from_netcdf_quick(args.data_dir, n_to_load)
    print(f'  Loaded {len(scenarios)} scenarios.')

    print(f'Sampling {args.n_sensors} sensor positions and {args.n_queries} query positions ...')
    sensor_positions = sample_positions(scenarios, args.n_sensors, seed=args.seed)
    query_positions  = sample_positions(scenarios, args.n_queries, seed=args.seed + 1)

    # Inject 'readings' and 'source_location' into each scenario dict.
    # s10 / s12 dataset classes expect these keys (see s10_structure_baselines.py:235).
    print('Computing sensor readings for all scenarios ...')
    from scipy.interpolate import RegularGridInterpolator
    from tqdm import tqdm
    for s in tqdm(scenarios):
        grid  = s['grid']
        field = s['concentration_field']
        interp = RegularGridInterpolator(
            (grid['x'], grid['y'], grid['z']),
            field, method='linear', bounds_error=False, fill_value=0.0
        )
        readings = interp(sensor_positions).astype(np.float32)       # (n_sensors,)
        all_readings = interp(
            np.vstack([sensor_positions, query_positions])
        ).astype(np.float32)                                           # (n_sensors+n_queries,)

        s['readings']        = readings
        s['source_location'] = np.array(s['leak_location'], dtype=np.float32)  # (3,)

        # s12_joint_training.py expects 'time_series': (seq_len, n_sensors+n_queries).
        # The quick CFD loader provides only the final-timestep snapshot, so we
        # duplicate it to give seq_len=2. The dynamics loss will be ~0 (predicting
        # an identical next state), consistent with the paper's finding that the
        # ~10,000x loss scale imbalance causes localization to dominate regardless.
        s['time_series'] = np.stack([all_readings, all_readings], axis=0)  # (2, N)

    print('Splitting into train / val / test (80/10/10) ...')
    train_scen, val_scen, test_scen, full_order = split_scenarios(scenarios, seed=args.seed)
    print(f'  train={len(train_scen)}  val={len(val_scen)}  test={len(test_scen)}')

    print('Extracting time-series data (this may take a few minutes) ...')
    # Extract over ALL scenarios in original order, then pass full list to scripts
    # (s10/s12 use train/val/test scenario dicts for the localizer, and
    #  time_series_data for the graph-structure learning phase)
    time_series = extract_time_series(scenarios, sensor_positions, query_positions)
    print(f'  time_series_data shape: {tuple(time_series.shape)}')

    data_dict = {
        'train_scenarios':  train_scen,
        'val_scenarios':    val_scen,
        'test_scenarios':   test_scen,
        'sensor_positions': torch.tensor(sensor_positions, dtype=torch.float32),
        'query_positions':  torch.tensor(query_positions,  dtype=torch.float32),
        'time_series_data': time_series,   # (n_scenarios, T, n_sensors + n_queries)
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_dict, out_path)
    print(f'\nSaved to {out_path}  '
          f'({out_path.stat().st_size / 1e6:.1f} MB)')
    print('\nUsage:')
    print(f'  python src/s10_structure_baselines.py --data_path {out_path} '
          f'--method both --output_dir results/s10_final')
    print(f'  python s12_joint_training.py      --data_path {out_path} '
          f'--output_dir results/s12_final')


if __name__ == '__main__':
    main()
