#!/usr/bin/env python3
"""
Single-condition localizer experiment runner.

Trains and evaluates the four localizer architectures (MLP, Green's Function,
Learned Attention, NRI-Graph) for one (domain, n_sensors, n_scenarios) condition
and writes table1.csv / table1.tex / table1_results_raw.csv into --output.

This is the shared engine invoked (as a subprocess, with different flags) by
make_table1.py, make_table2.py, make_table5.py, make_table6.py, make_table7.py,
and make_table8.py — it is not itself "Table 1"-specific despite the historical
output filenames.

Author: NTJ
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import os
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

from localizer_architectures import create_localizer


# ============================================================================
# UTILS
# ============================================================================

def subsample_scenarios(scenarios: List[Dict], n_scenarios: Optional[int], seed: int = 0) -> List[Dict]:
    """Deterministically subsample scenarios without replacement."""
    if n_scenarios is None or n_scenarios >= len(scenarios):
        return scenarios
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(scenarios), size=n_scenarios, replace=False)
    return [scenarios[i] for i in idx]


def resolve_nri_edge_path(
    n_scenarios: int,
    nri_edge_dir: Optional[str] = None,
    nri_edge_path: Optional[str] = None,
    domain: str = "cfd",
) -> Path:
    """Pick the NRI checkpoint file. If explicit path is given, use it; otherwise
    infer from m, defaulting the checkpoint directory to checkpoints/{domain}/
    (so callers can't accidentally load the wrong domain's checkpoint by
    forgetting to override it) unless nri_edge_dir is explicitly given."""
    if nri_edge_path is not None:
        return Path(nri_edge_path)
    edge_dir = nri_edge_dir if nri_edge_dir is not None else f"checkpoints/{domain}"
    return Path(edge_dir) / f"nri_edge_probs_spatial_{n_scenarios}.pt"


def create_query_points(n_queries: int, bounds: Tuple[float, float, float]) -> np.ndarray:
    """Create query points as a uniform grid (same logic as train_nri_spatial)."""
    bx, by, bz = bounds
    if bz < 5:
        n_z = 2
        n_xy = int(np.ceil(np.sqrt(n_queries / n_z)))
    else:
        n_side = int(np.ceil(n_queries ** (1/3)))
        n_xy = n_side
        n_z = max(2, n_side // 2)

    x = np.linspace(0, bx, n_xy)
    y = np.linspace(0, by, n_xy)
    z = np.linspace(0, bz, n_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    if len(points) > n_queries:
        idx = np.random.permutation(len(points))[:n_queries]
        points = points[idx]
    elif len(points) < n_queries:
        n_extra = n_queries - len(points)
        extra = np.random.rand(n_extra, 3)
        extra[:, 0] *= bx
        extra[:, 1] *= by
        extra[:, 2] *= bz
        points = np.vstack([points, extra])

    return points.astype(np.float32)


# ============================================================================
# DATASET
# ============================================================================

class LocalizationDataset(Dataset):
    """
    Dataset for training localizers.
    
    Each sample: (sensor_positions, readings, source_location)
    """
    
    def __init__(self, scenarios: List[Dict], n_sensors: int,
                 n_samples_per_scenario: int = 100, 
                 seed: int = 42):
        """
        Args:
            scenarios: List of CFD scenario dictionaries
            n_sensors: Number of sensors
            n_samples_per_scenario: Random sensor placements per scenario
            seed: Random seed
        """
        self.scenarios = scenarios
        self.n_sensors = n_sensors
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Build dataset
        self.samples = []
        self._build_dataset(n_samples_per_scenario)
    
    def _build_dataset(self, n_samples: int):
        """Generate random sensor placements for each scenario"""
        
        for scenario in tqdm(self.scenarios, desc="Building dataset"):
            grid = scenario['grid']
            concentration = scenario['concentration_field']
            source_loc = torch.tensor(scenario['leak_location'], dtype=torch.float32)
            
            # Create interpolator for this scenario
            interpolator = RegularGridInterpolator(
                (grid['x'], grid['y'], grid['z']),
                concentration,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            
            # Generate random sensor positions
            for _ in range(n_samples):
                positions = np.random.rand(self.n_sensors, 3)
                positions[:, 0] *= (grid['x'].max() - grid['x'].min()) + grid['x'].min()
                positions[:, 1] *= (grid['y'].max() - grid['y'].min()) + grid['y'].min()
                positions[:, 2] *= (grid['z'].max() - grid['z'].min()) + grid['z'].min()
                
                # Get sensor readings
                readings = interpolator(positions)
                
                # Store sample
                self.samples.append({
                    'sensor_positions': torch.tensor(positions, dtype=torch.float32),
                    'readings': torch.tensor(readings, dtype=torch.float32),
                    'source_location': source_loc
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return (
            sample['sensor_positions'],
            sample['readings'],
            sample['source_location']
        )


# ============================================================================
# TRAINING
# ============================================================================

def train_localizer(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   n_epochs: int = 100,
                   lr: float = 1e-3,
                   device: str = 'cpu',
                   verbose: bool = True) -> Dict:
    """
    Train a localizer model.
    
    Args:
        model: Localizer module
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        verbose: Print progress
    
    Returns:
        training_history: Dict with train/val losses
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for sensor_pos, readings, source_loc in train_loader:
            sensor_pos = sensor_pos.to(device)
            readings = readings.to(device)
            source_loc = source_loc.to(device)
            
            optimizer.zero_grad()
            
            pred_source = model(readings, sensor_pos)
            loss = ((pred_source - source_loc) ** 2).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_rmse = 0
        
        with torch.no_grad():
            for sensor_pos, readings, source_loc in val_loader:
                sensor_pos = sensor_pos.to(device)
                readings = readings.to(device)
                source_loc = source_loc.to(device)
                
                pred_source = model(readings, sensor_pos)
                loss = ((pred_source - source_loc) ** 2).mean()
                rmse = torch.sqrt(((pred_source - source_loc) ** 2).sum(dim=-1)).mean()
                
                val_loss += loss.item()
                val_rmse += rmse.item()
        
        val_loss /= len(val_loader)
        val_rmse /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"RMSE: {val_rmse:.2f}m")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_localizer(model: nn.Module,
                      test_loader: DataLoader,
                      scenarios: List[Dict],
                      device: str = 'cpu',
                      n_sensors: int = 15,
                      domain_cfg: Optional[Dict] = None) -> Dict:
    """
    Comprehensive evaluation of a localizer.
    
    Computes:
    - Localization RMSE (meters)
    - Detection rate (%)
    - Fitness score (composite metric)
    - Parameter count
    
    Args:
        model: Trained localizer
        test_loader: Test data loader
        scenarios: CFD scenarios (for detection rate)
        device: Device
        domain_cfg: Domain configuration dict (from domain_config.py)
    
    Returns:
        metrics: Dict with all evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    # 1. Localization RMSE
    rmse_errors = []
    
    with torch.no_grad():
        for sensor_pos, readings, source_loc in test_loader:
            sensor_pos = sensor_pos.to(device)
            readings = readings.to(device)
            source_loc = source_loc.to(device)
            
            pred_source = model(readings, sensor_pos)
            
            # Euclidean distance error
            errors = torch.sqrt(((pred_source - source_loc) ** 2).sum(dim=-1))
            rmse_errors.extend(errors.cpu().numpy())
    
    rmse_mean = np.mean(rmse_errors)
    rmse_std = np.std(rmse_errors)
    
    # 2. Detection rate
    # For this we need to evaluate on actual sensor configurations
    # Simplified: use a fixed grid of sensors
    detection_count = 0
    total_scenarios = len(scenarios)

    # Domain-aware sensor grid and detection threshold
    if domain_cfg is None:
        from domain_config import get_domain_config
        domain_cfg = get_domain_config("cfd")

    detection_threshold = domain_cfg["detection_threshold"]
    x_lo, x_hi = domain_cfg["eval_x_range"]
    y_lo, y_hi = domain_cfg["eval_y_range"]
    z_sensor = domain_cfg["eval_z"]

    # Create fixed sensor grid for evaluation
    # Use the requested sensor count for fair scaling comparisons
    n_sensors = int(n_sensors)
    n_side = int(np.ceil(np.sqrt(n_sensors)))
    x_sensors = np.linspace(x_lo, x_hi, n_side)
    y_sensors = np.linspace(y_lo, y_hi, n_side)
    
    sensor_positions = []
    for x in x_sensors[:n_side]:
        for y in y_sensors[:n_side]:
            sensor_positions.append([x, y, z_sensor])
            if len(sensor_positions) >= n_sensors:
                break
        if len(sensor_positions) >= n_sensors:
            break
    
    sensor_positions = np.array(sensor_positions[:n_sensors])
    
    for scenario in scenarios:
        grid = scenario['grid']
        concentration = scenario['concentration_field']
        
        # Interpolate readings
        interpolator = RegularGridInterpolator(
            (grid['x'], grid['y'], grid['z']),
            concentration,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        readings = interpolator(sensor_positions)
        
        if np.max(readings) > detection_threshold:
            detection_count += 1
    
    detection_rate = detection_count / total_scenarios
    
    # 3. Fitness (simplified composite)
    # fitness = 0.4 * MI + 0.3 * detection + 0.3 * coverage
    # For now, use detection rate as proxy
    # (Full fitness requires mutual information and coverage computation)
    fitness = detection_rate  # Simplified
    
    # 4. Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        'rmse_mean': float(rmse_mean),
        'rmse_std': float(rmse_std),
        'detection_rate': float(detection_rate * 100),  # as percentage
        'fitness': float(fitness),
        'n_parameters': int(n_params)
    }
    
    return metrics


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_table1_experiment(scenarios: List[Dict],
                         n_sensors: int = 15,
                         n_samples_per_scenario: int = 100,
                         n_epochs: int = 100,
                         batch_size: int = 64,
                         n_seeds: int = 10,
                         device: str = 'cpu',
                         nri_edge_dir: Optional[str] = None,
                         nri_edge_path: Optional[str] = None,
                         output_dir: str = 'results/table1',
                         specific_arch: Optional[str] = None,
                         domain_cfg: Optional[Dict] = None) -> pd.DataFrame:
    """
    Run complete Table 1 experiment.
    
    Trains and evaluates all 4 localizer architectures with multiple seeds.
    
    Args:
        scenarios: CFD scenarios
        n_sensors: Number of sensors
        n_samples_per_scenario: Training samples per scenario
        n_epochs: Training epochs
        batch_size: Batch size
        n_seeds: Number of random seeds
        device: Device to use
        output_dir: Output directory
    
    Returns:
        results_df: DataFrame with all results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Domain-specific configuration
    if domain_cfg is None:
        from domain_config import get_domain_config
        domain_cfg = get_domain_config("cfd")
    bounds = domain_cfg["bounds"]
    
    all_architectures = ['mlp', 'greens', 'attention', 'nri']
    architecture_names = {
        'mlp': 'MLP (no structure)',
        'greens': 'Green\'s Function',
        'attention': 'Learned Attention',
        'nri': 'NRI-Graph'
    }
    
    # Select architectures to run
    if specific_arch is not None:
        if specific_arch not in all_architectures:
            raise ValueError(f"Invalid architecture: {specific_arch}. Choose from {all_architectures}")
        architectures = [specific_arch]
        print(f"Running specific architecture: {architecture_names[specific_arch]}")
    else:
        architectures = all_architectures
        print("Running all architectures")
    
    results = []
    
    print("="*70)
    print("LOCALIZER EXPERIMENT: ARCHITECTURE COMPARISON")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Domain: {domain_cfg['name']}")
    print(f"  Bounds: {bounds}")
    print(f"  Sensors: {n_sensors}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Samples per scenario: {n_samples_per_scenario}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Device: {device}")
    print()
    
    # For each architecture
    for arch in architectures:
        print(f"\n{'='*70}")
        print(f"Architecture: {architecture_names[arch]}")
        print(f"{'='*70}\n")
        
        arch_results = []
        
        # For each seed
        for seed in range(n_seeds):
            print(f"\n--- Seed {seed+1}/{n_seeds} ---")
            
            # Create dataset
            dataset = LocalizationDataset(
                scenarios, n_sensors, n_samples_per_scenario, seed=seed
            )
            
            # Train/val/test split
            n_total = len(dataset)
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            n_test = n_total - n_train - n_val
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(seed)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            print(f"Dataset: {n_train} train, {n_val} val, {n_test} test")
            
            # Create model
            torch.manual_seed(seed)
            
            #model = create_localizer(arch, n_sensors, hidden_dim=64, n_queries=32)

            if arch == "nri":
                # Load learned NRI graph (edge probabilities between query nodes)
                m_scenarios = len(scenarios)
                ckpt_path = resolve_nri_edge_path(
                    n_scenarios=m_scenarios,
                    nri_edge_dir=nri_edge_dir,
                    nri_edge_path=nri_edge_path,
                    domain=domain_cfg["name"],
                )

                if not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"NRI checkpoint not found for m={m_scenarios}.\n"
                        f"Expected: {ckpt_path}\n"
                        f"Train it first, e.g.:\n"
                        f"  python src/train_nri_spatial.py --domain {domain_cfg['name']} "
                        f"--n_scenarios {m_scenarios}"
                    )

                print(f"Loading NRI structure from: {ckpt_path}")
                nri_obj = torch.load(str(ckpt_path), map_location="cpu")

                # Support either dict checkpoints (recommended) or raw tensors/ndarrays (edge_prob only)
                edge_prob = None
                query_points = None
                if isinstance(nri_obj, dict):
                    edge_prob = nri_obj.get("edge_prob", None)
                    if edge_prob is None:
                        edge_prob = nri_obj.get("edge_probs", None)
                    if edge_prob is None:
                        edge_prob = nri_obj.get("edge_probability", None)
                    query_points = nri_obj.get("query_points", None)
                else:
                    edge_prob = nri_obj  # tensor/ndarray

                # If query_points are missing, regenerate deterministically from the domain bounds
                if query_points is None:
                    rng_state = np.random.get_state()
                    np.random.seed(0)
                    qp_np = create_query_points(n_queries=32, bounds=bounds)
                    np.random.set_state(rng_state)
                    query_points = torch.tensor(qp_np, dtype=torch.float32)

                # Ensure tensors
                if not torch.is_tensor(edge_prob):
                    edge_prob = torch.tensor(edge_prob, dtype=torch.float32)
                if not torch.is_tensor(query_points):
                    query_points = torch.tensor(query_points, dtype=torch.float32)

                model = create_localizer(
                    "nri",
                    n_sensors,
                    hidden_dim=64,
                    n_queries=32,
                    bounds=bounds,
                    edge_prob=edge_prob,
                    query_points=query_points,
                    sparsity_threshold=None
                )
            else:
                model = create_localizer(
                    arch,
                    n_sensors,
                    hidden_dim=64,
                    n_queries=32,
                    bounds=bounds,
                )

            print(f"Model: {model.count_parameters():,} parameters")
            
            # Train
            print("Training...")
            history = train_localizer(
                model, train_loader, val_loader,
                n_epochs=n_epochs, lr=1e-3, device=device, verbose=True
            )
            
            # Evaluate
            print("Evaluating...")
            metrics = evaluate_localizer(model, test_loader, scenarios, device=device,
                                         n_sensors=n_sensors, domain_cfg=domain_cfg)
            
            # Store results
            result = {
                'architecture': arch,
                'architecture_name': architecture_names[arch],
                'seed': seed,
                'n_sensors': n_sensors,
                **metrics,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
            }
            
            arch_results.append(result)
            results.append(result)
            
            print(f"\nResults:")
            print(f"  RMSE: {metrics['rmse_mean']:.2f} ± {metrics['rmse_std']:.2f} m")
            print(f"  Detection: {metrics['detection_rate']:.1f}%")
            print(f"  Fitness: {metrics['fitness']:.3f}")
            print(f"  Params: {metrics['n_parameters']:,}")
            
            # Save model checkpoint
            checkpoint_path = f"{output_dir}/model_{arch}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint_path)
        
        # Print architecture summary
        arch_df = pd.DataFrame(arch_results)
        print(f"\n{'='*70}")
        print(f"Summary for {architecture_names[arch]}:")
        print(f"  RMSE: {arch_df['rmse_mean'].mean():.2f} ± {arch_df['rmse_mean'].std():.2f} m")
        print(f"  Detection: {arch_df['detection_rate'].mean():.1f} ± {arch_df['detection_rate'].std():.1f}%")
        print(f"  Fitness: {arch_df['fitness'].mean():.3f} ± {arch_df['fitness'].std():.3f}")
        print(f"{'='*70}\n")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(f"{output_dir}/table1_results_raw.csv", index=False)
    
    # Generate Table 1
    table1 = generate_table1(results_df)
    print("\n" + "="*70)
    print(f"ARCHITECTURE COMPARISON (n={results_df['n_sensors'].iloc[0]} sensors)")
    print("="*70)
    print(table1.to_string(index=False))
    print()
    
    table1.to_csv(f"{output_dir}/table1.csv", index=False)
    
    # Save as LaTeX
    latex_table = table1.to_latex(index=False, float_format="%.2f")
    with open(f"{output_dir}/table1.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"\nResults saved to {output_dir}/")
    
    return results_df



def generate_table1(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table 1 in paper format."""

    # Aggregate across seeds per architecture
    g = results_df.groupby("architecture_name", as_index=False).agg(
        rmse_mean_mean=("rmse_mean", "mean"),
        rmse_mean_std=("rmse_mean", "std"),
        fitness_mean=("fitness", "mean"),
        fitness_std=("fitness", "std"),
        detection_mean=("detection_rate", "mean"),
        detection_std=("detection_rate", "std"),
        n_parameters_mean=("n_parameters", "mean"),
    )

    # Format like the paper table
    table = pd.DataFrame({
        "Localizer": g["architecture_name"],
        "RMSE (m)": g.apply(lambda r: f"{r.rmse_mean_mean:.2f} ± {0.0 if np.isnan(r.rmse_mean_std) else r.rmse_mean_std:.2f}", axis=1),
        "Fitness": g.apply(lambda r: f"{r.fitness_mean:.3f} ± {0.0 if np.isnan(r.fitness_std) else r.fitness_std:.3f}", axis=1),
        "Det. (%)": g.apply(lambda r: f"{r.detection_mean:.1f} ± {0.0 if np.isnan(r.detection_std) else r.detection_std:.1f}", axis=1),
        "Params": g["n_parameters_mean"].round(0).astype(int).map(lambda x: f"{x:,}")
    })

    # Keep a consistent row order matching the paper
    order = ["MLP (no structure)", "Green's Function", "Learned Attention", "NRI-Graph"]
    table["__order"] = table["Localizer"].apply(lambda x: order.index(x) if x in order else 999)
    table = table.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    return table


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full experiment")

    parser.add_argument(
        "--n_sensors",
        type=int,
        default=15,
        help="Number of sensors to use"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run training on"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing scenario data. Default: cfd_data/ or acoustic_data/, derived from --domain."
    )

    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["mlp", "greens", "attention", "nri"],
        help="Run specific architecture only (default: run all)"
    )

    # Data regime control
    parser.add_argument(
        "--n_scenarios",
        type=int,
        default=None,
        help="Number of scenarios m to use (also selects NRI checkpoint nri_edge_probs_spatial_{m}.pt). "
             "Default: use all scenarios loaded by the loader."
    )
    parser.add_argument(
        "--scenario_seed",
        type=int,
        default=0,
        help="Seed for scenario subsampling (only used if --n_scenarios < available scenarios)."
    )
    parser.add_argument(
        "--scenarios-file",
        type=str,
        default=None,
        help="Path to a pickle file containing pre-loaded scenarios (e.g. an OOD train/test split "
             "for Table 6). If provided, overrides CFD/acoustic loading and --n_scenarios."
    )

    # NRI checkpoint selection
    parser.add_argument(
        "--nri_edge_dir",
        type=str,
        default=None,
        help="Directory containing NRI checkpoints named nri_edge_probs_spatial_{m}.pt. "
             "Default: checkpoints/cfd/ or checkpoints/acoustic/, derived from --domain."
    )
    parser.add_argument(
        "--nri_edge_path",
        type=str,
        default=None,
        help="Explicit path to an NRI checkpoint (overrides --nri_edge_dir)."
    )

    # Exact output directory (optional override)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Exact output directory to store results (overrides --output_root naming)."
    )

    # Output root
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/table1",
        help="Root folder prefix for results (used only if --output is not set). Output dir becomes <output_root>_full_n{n}_m{m} or <output_root>_test_n{n}_m{m}."
    )

    # Domain selection
    parser.add_argument(
        "--domain",
        type=str,
        default="cfd",
        choices=["cfd", "acoustic"],
        help="Domain: 'cfd' (hydrogen dispersion) or 'acoustic' (synthetic). Default: cfd"
    )

    args = parser.parse_args()

    if not args.test and not args.full:
        print("No action specified. Use --test or --full")
        raise SystemExit(0)

    from domain_config import get_domain_config
    domain_cfg = get_domain_config(args.domain)

    # Derive the data directory from --domain unless explicitly overridden,
    # so forgetting --data_dir for an acoustic run doesn't silently try to
    # load acoustic scenarios from cfd_data/.
    data_dir = args.data_dir if args.data_dir is not None else f"{args.domain}_data/"

    def load_scenarios(m_default: int) -> List[Dict]:
        """Load scenarios for this run: from a pickle (--scenarios-file, used for
        OOD splits) if given, otherwise from the domain's CFD/acoustic loader."""
        if args.scenarios_file:
            print(f"\nLoading scenarios from pickle file: {args.scenarios_file}")
            try:
                with open(args.scenarios_file, 'rb') as f:
                    scenarios = pickle.load(f)
                print(f"✓ Loaded {len(scenarios)} scenarios from pickle file")
            except FileNotFoundError:
                print(f"❌ Error: Pickle file not found: {args.scenarios_file}")
                raise SystemExit(1)
            except Exception as e:
                print(f"❌ Error loading pickle file: {e}")
                raise SystemExit(1)
            return scenarios

        m = args.n_scenarios if args.n_scenarios is not None else m_default
        if args.domain == "acoustic":
            from acoustic_loader import load_acoustic_scenarios_as_cfd
            scenarios = load_acoustic_scenarios_as_cfd(data_dir, m)
        else:
            from load_cfd_netcdf import load_cfd_from_netcdf_quick
            scenarios = load_cfd_from_netcdf_quick(data_dir, m)
        return subsample_scenarios(scenarios, m, seed=args.scenario_seed)

    if args.test:
        scenarios = load_scenarios(4)
        output_dir = args.output if args.output is not None else f"{args.output_root}_test_n{args.n_sensors}_m{len(scenarios)}"
        run_table1_experiment(
            scenarios,
            n_sensors=args.n_sensors,
            n_samples_per_scenario=50,
            n_epochs=20,
            n_seeds=2,
            device=args.device,
            nri_edge_dir=args.nri_edge_dir,
            nri_edge_path=args.nri_edge_path,
            output_dir=output_dir,
            specific_arch=args.arch,
            domain_cfg=domain_cfg,
        )
        print(f"✓ Quick test complete! Results in: {output_dir}")

    if args.full:
        scenarios = load_scenarios(180)
        output_dir = args.output if args.output is not None else f"{args.output_root}_full_n{args.n_sensors}_m{len(scenarios)}"
        run_table1_experiment(
            scenarios,
            n_sensors=args.n_sensors,
            n_samples_per_scenario=100,
            n_epochs=100,
            n_seeds=10,
            device=args.device,
            nri_edge_dir=args.nri_edge_dir,
            nri_edge_path=args.nri_edge_path,
            output_dir=output_dir,
            specific_arch=args.arch,
            domain_cfg=domain_cfg,
        )

        print(f"✓ Full experiment complete! Results in: {output_dir}")
