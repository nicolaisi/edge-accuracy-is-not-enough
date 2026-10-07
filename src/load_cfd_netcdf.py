#!/usr/bin/env python3
"""
Load CFD data from NetCDF files (FDS simulations)

Your data structure:
- cfd_data/
  ├── ach10m100p1/   (ACH=10, leak=100g/s, position=1)
  │   ├── visout00001.nc  (timestep 1)
  │   ├── visout00002.nc  (timestep 2)
  │   └── ...
  ├── ach10m100p2/
  └── ...

NetCDF variables:
- vf_h2: (200, 120, 12) - hydrogen volume fraction
- mesh: (201, 121, 13, 3) - grid vertices
- time: scalar timestamp

Usage:
    from load_cfd_netcdf import load_cfd_from_netcdf
    scenarios = load_cfd_from_netcdf('cfd_data/')
"""

import netCDF4 as nc
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def load_cfd_from_netcdf(
    data_dir: str,
    timestep: int = -1,  # -1 = last (steady state), or specific timestep
    min_scenarios: int = 16,  # Minimum scenarios to load
    max_scenarios: Optional[int] = None
) -> List[Dict]:
    """
    Load CFD scenarios from NetCDF files.
    
    Args:
        data_dir: Root directory containing scenario subdirectories
        timestep: Which timestep to load (-1 for last/steady-state, 0-60 for specific)
        min_scenarios: Minimum number of scenarios to load
        max_scenarios: Maximum number of scenarios (None = all)
    
    Returns:
        List of scenario dictionaries
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all scenario directories
    scenario_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario directories found in {data_dir}")
    
    print(f"Found {len(scenario_dirs)} scenario directories")
    
    # Limit number of scenarios if specified
    if max_scenarios is not None:
        scenario_dirs = scenario_dirs[:max_scenarios]
    
    if len(scenario_dirs) < min_scenarios:
        print(f"Warning: Found only {len(scenario_dirs)} scenarios, need {min_scenarios}")
    
    # Load each scenario
    scenarios = []
    for i, scenario_dir in enumerate(scenario_dirs):
        print(f"Loading scenario {i+1}/{len(scenario_dirs)}: {scenario_dir.name}...")
        
        try:
            scenario = _load_single_scenario(scenario_dir, timestep, scenario_id=i)
            scenarios.append(scenario)
        except Exception as e:
            print(f"  ✗ Error loading {scenario_dir.name}: {e}")
            continue
    
    print(f"\n✓ Successfully loaded {len(scenarios)} scenarios")
    
    if scenarios:
        _print_summary(scenarios)
    
    return scenarios


def _load_single_scenario(scenario_dir: Path, timestep: int, scenario_id: int) -> Dict:
    """Load a single CFD scenario from directory"""
    
    # Find NetCDF files
    nc_files = sorted(scenario_dir.glob('*.nc'))
    
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files in {scenario_dir}")
    
    # Select timestep
    if timestep == -1:
        # Use last timestep (steady state)
        nc_file = nc_files[-1]
    else:
        # Use specific timestep
        if timestep >= len(nc_files):
            raise ValueError(f"Timestep {timestep} not available (only {len(nc_files)} files)")
        nc_file = nc_files[timestep]
    
    # Parse directory name for metadata
    metadata = _parse_scenario_name(scenario_dir.name)
    
    # Load NetCDF file
    with nc.Dataset(nc_file, 'r') as ds:
        # Extract hydrogen concentration (volume fraction)
        vf_h2 = ds.variables['vf_h2'][:]  # (nx, ny, nz)
        
        # Convert volume fraction to ppm
        concentration_ppm = vf_h2 * 1e6
        
        # Extract grid (mesh vertices)
        mesh = ds.variables['mesh'][:]  # (nx+1, ny+1, nz+1, 3)
        
        # Get cell-centered coordinates
        x_coords = (mesh[:-1, 0, 0, 0] + mesh[1:, 0, 0, 0]) / 2
        y_coords = (mesh[0, :-1, 0, 1] + mesh[0, 1:, 0, 1]) / 2
        z_coords = (mesh[0, 0, :-1, 2] + mesh[0, 0, 1:, 2]) / 2
        
        # Find leak location (approximate from peak concentration)
        leak_location = _estimate_leak_location(concentration_ppm, mesh)
        
        # Get time
        time_value = float(ds.variables['time'][()])
    
    scenario = {
        'id': scenario_id,
        'name': scenario_dir.name,
        'concentration_field': concentration_ppm,
        'grid': {
            'x': x_coords,
            'y': y_coords,
            'z': z_coords
        },
        'leak_location': leak_location,
        'leak_rate': metadata['leak_rate'],
        'ventilation': metadata['ach'],
        'position_id': metadata['position'],
        'time': time_value,
        'timestep': timestep if timestep >= 0 else len(nc_files) - 1,
        'n_timesteps': len(nc_files),
    }
    
    return scenario


def _parse_scenario_name(dir_name: str) -> Dict:
    """
    Parse scenario directory name to extract parameters.
    
    Format: ach{ACH}m{leak_rate}p{position}
    Example: ach10m100p1 → ACH=10, leak_rate=100, position=1
    """
    
    match = re.match(r'ach(\d+)m(\d+)p(\d+)', dir_name)
    
    if not match:
        raise ValueError(f"Cannot parse directory name: {dir_name}")
    
    return {
        'ach': int(match.group(1)),           # Air Changes per Hour
        'leak_rate': int(match.group(2)),     # g/s
        'position': int(match.group(3))       # Position ID
    }


def _estimate_leak_location(concentration: np.ndarray, mesh: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate leak location from concentration field.
    
    Finds the cell with maximum concentration.
    """
    
    # Find cell with max concentration
    max_idx = np.unravel_index(concentration.argmax(), concentration.shape)
    i, j, k = max_idx
    
    # Get cell center coordinates from mesh vertices
    leak_x = (mesh[i, 0, 0, 0] + mesh[i+1, 0, 0, 0]) / 2
    leak_y = (mesh[0, j, 0, 1] + mesh[0, j+1, 0, 1]) / 2
    leak_z = (mesh[0, 0, k, 2] + mesh[0, 0, k+1, 2]) / 2
    
    return (float(leak_x), float(leak_y), float(leak_z))


def _print_summary(scenarios: List[Dict]):
    """Print summary of loaded scenarios"""
    
    print("\n" + "="*70)
    print("SCENARIO SUMMARY")
    print("="*70)
    
    s0 = scenarios[0]
    
    print(f"Total scenarios: {len(scenarios)}")
    print(f"\nDomain:")
    print(f"  X: {s0['grid']['x'].min():.2f} - {s0['grid']['x'].max():.2f} m")
    print(f"  Y: {s0['grid']['y'].min():.2f} - {s0['grid']['y'].max():.2f} m")
    print(f"  Z: {s0['grid']['z'].min():.2f} - {s0['grid']['z'].max():.2f} m")
    
    print(f"\nResolution:")
    print(f"  {len(s0['grid']['x'])} × {len(s0['grid']['y'])} × {len(s0['grid']['z'])} cells")
    
    # Extract parameter ranges
    leak_rates = sorted(set(s['leak_rate'] for s in scenarios))
    achs = sorted(set(s['ventilation'] for s in scenarios))
    positions = sorted(set(s['position_id'] for s in scenarios))
    
    print(f"\nParameters:")
    print(f"  Leak rates: {leak_rates} g/s")
    print(f"  Ventilation (ACH): {achs}")
    print(f"  Positions: {len(positions)} unique locations")
    
    print(f"\nConcentration ranges (ppm):")
    for scenario in scenarios[:5]:  # Show first 5
        C = scenario['concentration_field']
        print(f"  {scenario['name']:20s}: [{C.min():8.2f}, {C.max():8.2f}] "
              f"leak at ({scenario['leak_location'][0]:.1f}, "
              f"{scenario['leak_location'][1]:.1f}, "
              f"{scenario['leak_location'][2]:.1f})")
    
    if len(scenarios) > 5:
        print(f"  ... and {len(scenarios)-5} more")
    
    print("="*70)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_cfd_from_netcdf_quick(
    data_dir: str = 'cfd_data/',
    n_scenarios: int = 16
) -> List[Dict]:
    """
    Quick loader with sensible defaults.
    
    Loads last timestep (steady state) from first n_scenarios directories.
    """
    return load_cfd_from_netcdf(
        data_dir=data_dir,
        timestep=-1,  # Last timestep
        min_scenarios=n_scenarios,
        max_scenarios=n_scenarios
    )


def load_cfd_time_series(
    data_dir: str,
    scenario_name: str,
    timesteps: Optional[List[int]] = None
) -> List[Dict]:
    """
    Load multiple timesteps from a single scenario.
    
    Useful for time-dependent analysis or creating video.
    
    Args:
        data_dir: Root data directory
        scenario_name: Subdirectory name (e.g., 'ach10m100p1')
        timesteps: List of timesteps to load, or None for all
    
    Returns:
        List of scenarios, one per timestep
    """
    
    scenario_dir = Path(data_dir) / scenario_name
    nc_files = sorted(scenario_dir.glob('*.nc'))
    
    if timesteps is None:
        timesteps = list(range(len(nc_files)))
    
    scenarios = []
    for t in timesteps:
        scenario = _load_single_scenario(scenario_dir, timestep=t, scenario_id=t)
        scenarios.append(scenario)
    
    return scenarios


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python src/load_cfd_netcdf.py <data_dir> [n_scenarios]")
        print("\nExample:")
        print("  python src/load_cfd_netcdf.py cfd_data/ 16")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    n_scenarios = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    
    print(f"Loading {n_scenarios} scenarios from {data_dir}...")
    
    try:
        scenarios = load_cfd_from_netcdf_quick(data_dir, n_scenarios)
        
        print(f"\n✓ Success! Loaded {len(scenarios)} scenarios")
        print(f"\nTo use in your code:")
        print(f"  from load_cfd_netcdf import load_cfd_from_netcdf_quick")
        print(f"  scenarios = load_cfd_from_netcdf_quick('{data_dir}', {n_scenarios})")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
