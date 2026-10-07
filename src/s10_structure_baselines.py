"""
Supplement Section S10: Additional Structure Learning Baselines
Experiments for correlation-based and mutual information-based graph structures.

This script implements all experiments needed to replace fabricated S10 content.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, ttest_rel
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, List

# ============================================================================
# Graph Construction Classes
# ============================================================================

class CorrelationGraphBuilder:
    """Build graph structure based on Pearson correlation."""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.correlation_matrix = None
        self.adjacency_matrix = None
        
    def fit(self, time_series_data: np.ndarray, n_sensors: int, n_queries: int):
        """
        Compute correlation-based graph structure.
        
        Args:
            time_series_data: Shape (n_scenarios, n_timesteps, n_sensors + n_queries)
            n_sensors: Number of sensors
            n_queries: Number of query points
        """
        n_scenarios, n_timesteps, n_total = time_series_data.shape
        assert n_total == n_sensors + n_queries
        
        # Reshape for correlation: (n_pairs, n_samples)
        # Each sample is a (scenario, timestep) combination
        data_reshaped = time_series_data.reshape(-1, n_total)  # (n_scenarios * n_timesteps, n_total)
        
        # Compute correlation matrix
        self.correlation_matrix = np.corrcoef(data_reshaped.T)  # (n_total, n_total)
        
        # Extract sensor-query correlations (cross-correlation block)
        sensor_query_corr = np.abs(self.correlation_matrix[:n_sensors, n_sensors:])  # (n_sensors, n_queries)
        
        # Threshold to get adjacency matrix
        self.adjacency_matrix = (sensor_query_corr > self.threshold).astype(float)
        
        return self
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Return adjacency matrix (n_sensors, n_queries)."""
        return self.adjacency_matrix
    
    def get_statistics(self) -> Dict:
        """Compute graph statistics."""
        n_sensors, n_queries = self.adjacency_matrix.shape
        n_edges = self.adjacency_matrix.sum()
        total_possible = n_sensors * n_queries
        avg_degree = n_edges / n_queries  # avg in-degree per query
        sparsity = 1.0 - (n_edges / total_possible)
        
        return {
            'n_edges': int(n_edges),
            'avg_degree': float(avg_degree),
            'sparsity': float(sparsity),
            'threshold': self.threshold
        }


class MutualInfoGraphBuilder:
    """Build graph structure based on mutual information."""
    
    def __init__(self, percentile: int = 70):
        self.percentile = percentile
        self.mi_matrix = None
        self.adjacency_matrix = None
        
    def fit(self, time_series_data: np.ndarray, n_sensors: int, n_queries: int):
        """
        Compute MI-based graph structure.
        
        Args:
            time_series_data: Shape (n_scenarios, n_timesteps, n_sensors + n_queries)
            n_sensors: Number of sensors
            n_queries: Number of query points
        """
        n_scenarios, n_timesteps, n_total = time_series_data.shape
        
        # Reshape for MI computation
        data_reshaped = time_series_data.reshape(-1, n_total)
        n_samples = data_reshaped.shape[0]
        
        # Compute MI between each sensor and each query
        self.mi_matrix = np.zeros((n_sensors, n_queries))
        
        print(f"Computing MI for {n_sensors} sensors × {n_queries} queries...")
        for i in tqdm(range(n_sensors), desc="Sensors"):
            X_sensor = data_reshaped[:, i].reshape(-1, 1)
            for j in range(n_queries):
                y_query = data_reshaped[:, n_sensors + j]
                # MI estimation (this can be slow)
                mi = mutual_info_regression(X_sensor, y_query, random_state=42)
                self.mi_matrix[i, j] = mi[0]
        
        # Threshold at specified percentile
        threshold_value = np.percentile(self.mi_matrix, self.percentile)
        self.adjacency_matrix = (self.mi_matrix > threshold_value).astype(float)
        
        return self
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Return adjacency matrix (n_sensors, n_queries)."""
        return self.adjacency_matrix
    
    def get_statistics(self) -> Dict:
        """Compute graph statistics."""
        n_sensors, n_queries = self.adjacency_matrix.shape
        n_edges = self.adjacency_matrix.sum()
        total_possible = n_sensors * n_queries
        avg_degree = n_edges / n_queries
        sparsity = 1.0 - (n_edges / total_possible)
        
        return {
            'n_edges': int(n_edges),
            'avg_degree': float(avg_degree),
            'sparsity': float(sparsity),
            'percentile': self.percentile
        }


# ============================================================================
# Structure-Constrained Localizer (Same as NRI-Graph)
# ============================================================================

class StructureConstrainedLocalizer(nn.Module):
    """Localizer with pre-specified graph structure (like NRI-Graph)."""
    
    def __init__(self, n_sensors: int, n_queries: int, 
                 adjacency_matrix: np.ndarray, hidden_dim: int = 64):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        
        # Store adjacency as non-trainable parameter
        self.register_buffer('adjacency', torch.FloatTensor(adjacency_matrix))
        
        # Sensor encoder: (position, reading) -> embedding
        self.sensor_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 3 for position + 1 for reading
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output head: aggregated features -> score
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, sensor_readings, sensor_positions, query_positions):
        """
        Args:
            sensor_readings: (batch, n_sensors)
            sensor_positions: (batch, n_sensors, 3)
            query_positions: (batch, n_queries, 3)
        Returns:
            predicted_sources: (batch, 3)
        """
        batch_size = sensor_readings.shape[0]
        
        # Encode sensors
        sensor_features = torch.cat([
            sensor_positions,  # (batch, n_sensors, 3)
            sensor_readings.unsqueeze(-1)  # (batch, n_sensors, 1)
        ], dim=-1)  # (batch, n_sensors, 4)
        
        h_sensors = self.sensor_encoder(sensor_features)  # (batch, n_sensors, hidden_dim)
        
        # Aggregate to queries using adjacency structure
        # adjacency: (n_sensors, n_queries)
        # h_sensors: (batch, n_sensors, hidden_dim)
        
        # Normalize adjacency row-wise (each query's weights sum to 1)
        adjacency_norm = self.adjacency / (self.adjacency.sum(dim=0, keepdim=True) + 1e-8)
        adjacency_norm = adjacency_norm.T  # (n_queries, n_sensors)
        
        # Aggregate: (batch, n_queries, hidden_dim)
        h_queries = torch.einsum('qr,brd->bqd', adjacency_norm, h_sensors)
        
        # Score each query
        scores = self.output_head(h_queries).squeeze(-1)  # (batch, n_queries)
        
        # Soft argmax over queries
        weights = F.softmax(scores, dim=1)  # (batch, n_queries)
        predicted_sources = torch.einsum('bq,bqd->bd', weights, query_positions)
        
        return predicted_sources


# ============================================================================
# Dataset and Training
# ============================================================================

class SourceLocalizationDataset(Dataset):
    """Dataset for source localization."""
    
    def __init__(self, scenarios, sensor_positions, query_positions):
        """
        Args:
            scenarios: List of dicts with 'readings', 'source_location'
            sensor_positions: (n_sensors, 3)
            query_positions: (n_queries, 3)
        """
        self.scenarios = scenarios
        self.sensor_positions = torch.FloatTensor(sensor_positions)
        self.query_positions = torch.FloatTensor(query_positions)
        
    def __len__(self):
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        return {
            'readings': torch.FloatTensor(scenario['readings']),
            'source': torch.FloatTensor(scenario['source_location']),
            'sensor_positions': self.sensor_positions,
            'query_positions': self.query_positions
        }


def train_localizer(model, train_loader, val_loader, device, 
                   n_epochs=500, lr=1e-3, patience=50):
    """Train structure-constrained localizer."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            readings = batch['readings'].to(device)
            sensor_pos = batch['sensor_positions'].to(device)
            query_pos = batch['query_positions'].to(device)
            true_source = batch['source'].to(device)
            
            pred_source = model(readings, sensor_pos, query_pos)
            loss = F.mse_loss(pred_source, true_source)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                readings = batch['readings'].to(device)
                sensor_pos = batch['sensor_positions'].to(device)
                query_pos = batch['query_positions'].to(device)
                true_source = batch['source'].to(device)
                
                pred_source = model(readings, sensor_pos, query_pos)
                loss = F.mse_loss(pred_source, true_source)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return model


def evaluate_localizer(model, test_loader, device):
    """Evaluate localizer and return RMSE in meters."""
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            readings = batch['readings'].to(device)
            sensor_pos = batch['sensor_positions'].to(device)
            query_pos = batch['query_positions'].to(device)
            true_source = batch['source'].to(device)
            
            pred_source = model(readings, sensor_pos, query_pos)
            
            # Euclidean distance in meters
            errors = torch.norm(pred_source - true_source, dim=1)
            all_errors.extend(errors.cpu().numpy())
    
    rmse = np.sqrt(np.mean(np.array(all_errors) ** 2))
    return rmse


# ============================================================================
# Experiment Orchestration
# ============================================================================

def run_single_experiment(adjacency_matrix, data_dict, seed, device):
    """
    Run single training run with given adjacency structure.
    
    Args:
        adjacency_matrix: (n_sensors, n_queries) numpy array
        data_dict: Dict with train/val/test splits
        seed: Random seed
        device: torch device
    
    Returns:
        test_rmse: float
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create datasets
    train_dataset = SourceLocalizationDataset(
        data_dict['train_scenarios'],
        data_dict['sensor_positions'],
        data_dict['query_positions']
    )
    val_dataset = SourceLocalizationDataset(
        data_dict['val_scenarios'],
        data_dict['sensor_positions'],
        data_dict['query_positions']
    )
    test_dataset = SourceLocalizationDataset(
        data_dict['test_scenarios'],
        data_dict['sensor_positions'],
        data_dict['query_positions']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    n_sensors = data_dict['sensor_positions'].shape[0]
    n_queries = data_dict['query_positions'].shape[0]
    model = StructureConstrainedLocalizer(n_sensors, n_queries, adjacency_matrix)
    
    # Train
    model = train_localizer(model, train_loader, val_loader, device)
    
    # Evaluate
    test_rmse = evaluate_localizer(model, test_loader, device)
    
    return test_rmse


def run_correlation_experiments(data_dict, device, output_dir):
    """Run all correlation-based experiments."""
    print("\n" + "="*80)
    print("CORRELATION-BASED GRAPH EXPERIMENTS")
    print("="*80)
    
    thresholds = [0.1, 0.2, 0.3, 0.4]
    n_seeds = 10
    results = {}
    
    # Extract time series for graph construction
    # Assuming data_dict has 'time_series_data' for all scenarios
    time_series = data_dict['time_series_data']  # (n_scenarios, n_timesteps, n_total)
    n_sensors = data_dict['sensor_positions'].shape[0]
    n_queries = data_dict['query_positions'].shape[0]
    
    for tau in thresholds:
        print(f"\n--- Threshold τ = {tau} ---")
        
        # Build graph
        builder = CorrelationGraphBuilder(threshold=tau)
        builder.fit(time_series, n_sensors, n_queries)
        adjacency = builder.get_adjacency_matrix()
        stats = builder.get_statistics()
        
        print(f"Graph statistics: {stats}")
        
        # Train with multiple seeds
        rmse_values = []
        for seed in range(n_seeds):
            print(f"Seed {seed}...", end=" ")
            rmse = run_single_experiment(adjacency, data_dict, seed, device)
            rmse_values.append(rmse)
            print(f"RMSE: {rmse:.2f}m")
        
        # Store results
        results[f'tau_{tau}'] = {
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'rmse_values': rmse_values,
            'graph_stats': stats
        }
        
        print(f"Mean RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f}m")
    
    # Save results
    output_path = Path(output_dir) / 'correlation_results.json'
    
    # Convert numpy types to Python native types for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = {
            'rmse_mean': float(value['rmse_mean']),
            'rmse_std': float(value['rmse_std']),
            'rmse_values': [float(x) for x in value['rmse_values']],
            'graph_stats': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in value['graph_stats'].items()}
        }
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return results


def run_mi_experiments(data_dict, device, output_dir):
    """Run all mutual information-based experiments."""
    print("\n" + "="*80)
    print("MUTUAL INFORMATION-BASED GRAPH EXPERIMENTS")
    print("="*80)
    
    percentiles = [60, 70, 80]
    n_seeds = 10
    results = {}
    
    time_series = data_dict['time_series_data']
    n_sensors = data_dict['sensor_positions'].shape[0]
    n_queries = data_dict['query_positions'].shape[0]
    
    for pct in percentiles:
        print(f"\n--- Percentile = {pct}th ---")
        
        # Build graph (this takes time!)
        builder = MutualInfoGraphBuilder(percentile=pct)
        builder.fit(time_series, n_sensors, n_queries)
        adjacency = builder.get_adjacency_matrix()
        stats = builder.get_statistics()
        
        print(f"Graph statistics: {stats}")
        
        # Train with multiple seeds
        rmse_values = []
        for seed in range(n_seeds):
            print(f"Seed {seed}...", end=" ")
            rmse = run_single_experiment(adjacency, data_dict, seed, device)
            rmse_values.append(rmse)
            print(f"RMSE: {rmse:.2f}m")
        
        # Store results
        results[f'percentile_{pct}'] = {
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'rmse_values': rmse_values,
            'graph_stats': stats
        }
        
        print(f"Mean RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f}m")
    
    # Save results
    output_path = Path(output_dir) / 'mi_results.json'
    
    # Convert numpy types to Python native types for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = {
            'rmse_mean': float(value['rmse_mean']),
            'rmse_std': float(value['rmse_std']),
            'rmse_values': [float(x) for x in value['rmse_values']],
            'graph_stats': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in value['graph_stats'].items()}
        }
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./s10_results',
                       help='Output directory')
    parser.add_argument('--method', type=str, choices=['correlation', 'mi', 'both'],
                       default='both', help='Which experiments to run')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_dict = torch.load(args.data_path, weights_only=False)
    # Expected keys: 'train_scenarios', 'val_scenarios', 'test_scenarios',
    #                'sensor_positions', 'query_positions', 'time_series_data'
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiments
    if args.method in ['correlation', 'both']:
        corr_results = run_correlation_experiments(data_dict, device, args.output_dir)
    
    if args.method in ['mi', 'both']:
        mi_results = run_mi_experiments(data_dict, device, args.output_dir)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
