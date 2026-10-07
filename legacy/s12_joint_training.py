"""
Supplement Section S12: Joint Multi-Task Training
Experiments for training models jointly on dynamics prediction and localization.

This script implements all experiments needed to replace fabricated S12 content.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Tuple, List

# ============================================================================
# Joint Multi-Task Model Architecture
# ============================================================================

class SharedEncoder(nn.Module):
    """Shared encoder for both dynamics and localization tasks."""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        
    def forward(self, time_series):
        """
        Args:
            time_series: (batch, seq_len, n_nodes, input_dim)
        Returns:
            embeddings: (batch, n_nodes, hidden_dim)
        """
        batch_size, seq_len, n_nodes, input_dim = time_series.shape
        
        # Process each node independently
        x = time_series.permute(0, 2, 1, 3)  # (batch, n_nodes, seq_len, input_dim)
        x = x.reshape(batch_size * n_nodes, seq_len, input_dim)
        
        _, h_n = self.rnn(x)  # h_n: (num_layers, batch*n_nodes, hidden_dim)
        embeddings = h_n[-1]  # Take last layer: (batch*n_nodes, hidden_dim)
        embeddings = embeddings.reshape(batch_size, n_nodes, -1)
        
        return embeddings


class EdgePredictor(nn.Module):
    """Predict edge probabilities from node embeddings."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings_i, embeddings_j):
        """
        Args:
            embeddings_i: (batch, n_i, hidden_dim) - e.g., sensors
            embeddings_j: (batch, n_j, hidden_dim) - e.g., queries
        Returns:
            edge_probs: (batch, n_i, n_j)
        """
        batch_size, n_i, hidden_dim = embeddings_i.shape
        _, n_j, _ = embeddings_j.shape
        
        # Expand for pairwise comparison
        emb_i_expanded = embeddings_i.unsqueeze(2).expand(-1, -1, n_j, -1)  # (batch, n_i, n_j, hidden)
        emb_j_expanded = embeddings_j.unsqueeze(1).expand(-1, n_i, -1, -1)  # (batch, n_i, n_j, hidden)
        
        # Concatenate
        edge_features = torch.cat([emb_i_expanded, emb_j_expanded], dim=-1)  # (batch, n_i, n_j, 2*hidden)
        
        # Predict edge probabilities
        edge_probs = self.edge_mlp(edge_features).squeeze(-1)  # (batch, n_i, n_j)
        
        return edge_probs


class DynamicsDecoder(nn.Module):
    """Decode next timestep from node embeddings and edges."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, embeddings, edge_probs):
        """
        Args:
            embeddings: (batch, n_nodes, hidden_dim)
            edge_probs: (batch, n_nodes, n_nodes)
        Returns:
            predictions: (batch, n_nodes, 1) - next timestep predictions
        """
        # Normalize edge probabilities
        edge_weights = edge_probs / (edge_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        # Aggregate information from neighbors
        aggregated = torch.bmm(edge_weights, embeddings)  # (batch, n_nodes, hidden_dim)
        
        # Decode
        predictions = self.decoder(aggregated)  # (batch, n_nodes, 1)
        
        return predictions


class LocalizationDecoder(nn.Module):
    """Decode source location from sensor embeddings and edges."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.sensor_processor = nn.Linear(hidden_dim + 1, hidden_dim)  # +1 for reading
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, sensor_embeddings, edge_probs, sensor_readings, query_positions):
        """
        Args:
            sensor_embeddings: (batch, n_sensors, hidden_dim)
            edge_probs: (batch, n_sensors, n_queries)
            sensor_readings: (batch, n_sensors)
            query_positions: (batch, n_queries, 3)
        Returns:
            predicted_sources: (batch, 3)
        """
        batch_size = sensor_embeddings.shape[0]
        
        # Combine embeddings with readings
        sensor_features = torch.cat([
            sensor_embeddings,
            sensor_readings.unsqueeze(-1)
        ], dim=-1)
        sensor_features = self.sensor_processor(sensor_features)
        
        # Normalize edge probabilities (sensors -> queries)
        edge_weights = edge_probs / (edge_probs.sum(dim=0, keepdim=True) + 1e-8)
        edge_weights = edge_weights.transpose(1, 2)  # (batch, n_queries, n_sensors)
        
        # Aggregate to queries
        query_features = torch.bmm(edge_weights, sensor_features)  # (batch, n_queries, hidden_dim)
        
        # Score queries
        scores = self.output_head(query_features).squeeze(-1)  # (batch, n_queries)
        
        # Soft argmax
        weights = F.softmax(scores, dim=1)  # (batch, n_queries)
        predicted_sources = torch.einsum('bq,bqd->bd', weights, query_positions)
        
        return predicted_sources


class JointModel(nn.Module):
    """Complete joint model for dynamics and localization."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.shared_encoder = SharedEncoder(input_dim=1, hidden_dim=hidden_dim)
        self.edge_predictor = EdgePredictor(hidden_dim=hidden_dim)
        self.dynamics_decoder = DynamicsDecoder(hidden_dim=hidden_dim)
        self.localization_decoder = LocalizationDecoder(hidden_dim=hidden_dim)
        
    def forward(self, time_series, sensor_readings, query_positions, 
                n_sensors: int):
        """
        Args:
            time_series: (batch, seq_len, n_total, 1) - full time series
            sensor_readings: (batch, n_sensors) - final timestep readings
            query_positions: (batch, n_queries, 3)
            n_sensors: number of sensors
        Returns:
            dynamics_pred: (batch, n_total, 1)
            source_pred: (batch, 3)
            edge_probs: (batch, n_sensors, n_queries)
        """
        batch_size, seq_len, n_total, _ = time_series.shape
        n_queries = query_positions.shape[1]
        
        # Encode all nodes
        embeddings = self.shared_encoder(time_series)  # (batch, n_total, hidden_dim)
        
        # Split sensor and query embeddings
        sensor_embeddings = embeddings[:, :n_sensors, :]
        query_embeddings = embeddings[:, n_sensors:, :]
        
        # Predict edges (sensor-query only for localization)
        edge_probs = self.edge_predictor(sensor_embeddings, query_embeddings)
        
        # Dynamics prediction (on all nodes with full graph)
        # For simplicity, use full graph for dynamics
        full_edge_probs = self.edge_predictor(embeddings, embeddings)
        dynamics_pred = self.dynamics_decoder(embeddings, full_edge_probs)
        
        # Localization prediction (sensor-query edges only)
        source_pred = self.localization_decoder(
            sensor_embeddings, edge_probs, sensor_readings, query_positions
        )
        
        return dynamics_pred, source_pred, edge_probs


# ============================================================================
# Joint Loss Function
# ============================================================================

class JointLoss(nn.Module):
    """Combined loss for dynamics and localization."""
    
    def __init__(self, lambda_dyn: float, lambda_loc: float):
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_loc = lambda_loc
        
    def forward(self, dynamics_pred, dynamics_true, source_pred, source_true):
        """
        Args:
            dynamics_pred: (batch, n_nodes, 1)
            dynamics_true: (batch, n_nodes, 1)
            source_pred: (batch, 3)
            source_true: (batch, 3)
        Returns:
            total_loss, dynamics_loss, localization_loss
        """
        dyn_loss = F.mse_loss(dynamics_pred, dynamics_true)
        loc_loss = F.mse_loss(source_pred, source_true)
        
        total_loss = self.lambda_dyn * dyn_loss + self.lambda_loc * loc_loss
        
        return total_loss, dyn_loss, loc_loss


# ============================================================================
# Dataset
# ============================================================================

class JointDataset(Dataset):
    """Dataset for joint training."""
    
    def __init__(self, scenarios, sensor_positions, query_positions):
        """
        Args:
            scenarios: List of dicts with 'time_series', 'readings', 'source_location'
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
        
        # time_series: (seq_len, n_total) -> (seq_len, n_total, 1)
        time_series = torch.FloatTensor(scenario['time_series']).unsqueeze(-1)
        
        # Next timestep for dynamics (predict last from second-to-last)
        X_current = time_series[:-1]  # (seq_len-1, n_total, 1)
        X_next = time_series[-1]  # (n_total, 1)
        
        return {
            'time_series': X_current,
            'X_next': X_next,
            'readings': torch.FloatTensor(scenario['readings']),
            'source': torch.FloatTensor(scenario['source_location']),
            'sensor_positions': self.sensor_positions,
            'query_positions': self.query_positions
        }


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_joint_model(model, train_loader, val_loader, criterion, device,
                     n_epochs=200, lr=1e-3, patience=50, n_sensors=15):
    """Train joint model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'dyn_loss': [], 'loc_loss': []}
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_metrics = {'total': [], 'dyn': [], 'loc': []}
        
        for batch in train_loader:
            time_series = batch['time_series'].to(device)
            X_next = batch['X_next'].to(device)
            readings = batch['readings'].to(device)
            source = batch['source'].to(device)
            query_pos = batch['query_positions'].to(device)
            
            # Forward
            dynamics_pred, source_pred, _ = model(
                time_series, readings, query_pos, n_sensors
            )
            
            # Loss
            total_loss, dyn_loss, loc_loss = criterion(
                dynamics_pred, X_next, source_pred, source
            )
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_metrics['total'].append(total_loss.item())
            train_metrics['dyn'].append(dyn_loss.item())
            train_metrics['loc'].append(loc_loss.item())
        
        # Validation
        model.eval()
        val_metrics = {'total': [], 'dyn': [], 'loc': []}
        
        with torch.no_grad():
            for batch in val_loader:
                time_series = batch['time_series'].to(device)
                X_next = batch['X_next'].to(device)
                readings = batch['readings'].to(device)
                source = batch['source'].to(device)
                query_pos = batch['query_positions'].to(device)
                
                dynamics_pred, source_pred, _ = model(
                    time_series, readings, query_pos, n_sensors
                )
                
                total_loss, dyn_loss, loc_loss = criterion(
                    dynamics_pred, X_next, source_pred, source
                )
                
                val_metrics['total'].append(total_loss.item())
                val_metrics['dyn'].append(dyn_loss.item())
                val_metrics['loc'].append(loc_loss.item())
        
        val_loss = np.mean(val_metrics['total'])
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: "
                  f"Val Loss={val_loss:.4f} "
                  f"(Dyn={np.mean(val_metrics['dyn']):.4f}, "
                  f"Loc={np.mean(val_metrics['loc']):.4f})")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # History
        history['train_loss'].append(np.mean(train_metrics['total']))
        history['val_loss'].append(val_loss)
        history['dyn_loss'].append(np.mean(val_metrics['dyn']))
        history['loc_loss'].append(np.mean(val_metrics['loc']))
    
    return model, history


def evaluate_joint_model(model, test_loader, device, n_sensors=15):
    """
    Evaluate joint model.
    
    Returns:
        loc_rmse: Localization RMSE in meters
        dyn_mae: Dynamics MAE
        jaccard: Jaccard similarity vs optimal (placeholder, need baseline attention)
    """
    model.eval()
    loc_errors = []
    dyn_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            time_series = batch['time_series'].to(device)
            X_next = batch['X_next'].to(device)
            readings = batch['readings'].to(device)
            source = batch['source'].to(device)
            query_pos = batch['query_positions'].to(device)
            
            dynamics_pred, source_pred, edge_probs = model(
                time_series, readings, query_pos, n_sensors
            )
            
            # Localization error (Euclidean distance)
            loc_error = torch.norm(source_pred - source, dim=1)
            loc_errors.extend(loc_error.cpu().numpy())
            
            # Dynamics error (MAE)
            dyn_error = torch.abs(dynamics_pred - X_next).mean(dim=(1, 2))
            dyn_errors.extend(dyn_error.cpu().numpy())
    
    loc_rmse = np.sqrt(np.mean(np.array(loc_errors) ** 2))
    dyn_mae = np.mean(dyn_errors)
    
    return loc_rmse, dyn_mae


# ============================================================================
# Jaccard Similarity Computation
# ============================================================================

def compute_jaccard_similarity(model, baseline_attention, data_loader, 
                               device, n_sensors=15, threshold=0.3):
    """
    Compute Jaccard similarity between learned edges and baseline attention.
    
    Args:
        model: Trained joint model
        baseline_attention: (n_sensors, n_queries) attention from Learned Attention baseline
        data_loader: Data to extract average attention
        device: torch device
        n_sensors: number of sensors
        threshold: threshold for binarizing attention
    
    Returns:
        jaccard: Jaccard similarity
    """
    model.eval()
    all_edge_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            time_series = batch['time_series'].to(device)
            readings = batch['readings'].to(device)
            query_pos = batch['query_positions'].to(device)
            
            _, _, edge_probs = model(time_series, readings, query_pos, n_sensors)
            all_edge_probs.append(edge_probs.cpu().numpy())
    
    # Average edge probabilities across all data
    avg_edge_probs = np.mean(np.concatenate(all_edge_probs, axis=0), axis=0)
    
    # Binarize both
    learned_binary = (avg_edge_probs > threshold).astype(int)
    baseline_binary = (baseline_attention > threshold).astype(int)
    
    # Compute Jaccard
    intersection = np.sum(learned_binary & baseline_binary)
    union = np.sum(learned_binary | baseline_binary)
    
    jaccard = intersection / (union + 1e-8)
    
    return jaccard


# ============================================================================
# Experiment Orchestration
# ============================================================================

def run_single_joint_experiment(lambda_dyn, lambda_loc, data_dict, seed, 
                                device, n_sensors=15):
    """Run single joint training experiment."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create datasets
    train_dataset = JointDataset(
        data_dict['train_scenarios'],
        data_dict['sensor_positions'],
        data_dict['query_positions']
    )
    val_dataset = JointDataset(
        data_dict['val_scenarios'],
        data_dict['sensor_positions'],
        data_dict['query_positions']
    )
    test_dataset = JointDataset(
        data_dict['test_scenarios'],
        data_dict['sensor_positions'],
        data_dict['query_positions']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model and criterion
    model = JointModel(hidden_dim=64)
    criterion = JointLoss(lambda_dyn=lambda_dyn, lambda_loc=lambda_loc)
    
    # Train
    print(f"\nTraining with λ_dyn={lambda_dyn}, λ_loc={lambda_loc}, seed={seed}")
    model, history = train_joint_model(
        model, train_loader, val_loader, criterion, device, n_sensors=n_sensors
    )
    
    # Evaluate
    loc_rmse, dyn_mae = evaluate_joint_model(model, test_loader, device, n_sensors)
    
    # Compute Jaccard (requires baseline attention - load if available)
    jaccard = None
    if 'baseline_attention' in data_dict and data_dict['baseline_attention'] is not None:
        jaccard = compute_jaccard_similarity(
            model, data_dict['baseline_attention'], test_loader, 
            device, n_sensors
        )
    else:
        if seed == 0:  # Only print once
            print("  (Skipping Jaccard similarity - no baseline attention available)")
    
    return {
        'loc_rmse': float(loc_rmse),
        'dyn_mae': float(dyn_mae),
        'jaccard': float(jaccard) if jaccard is not None else None,
        'history': history
    }


def run_all_joint_experiments(data_dict, device, output_dir):
    """Run all joint training configurations."""
    print("\n" + "="*80)
    print("JOINT MULTI-TASK TRAINING EXPERIMENTS")
    print("="*80)
    
    configs = [
        (0.9, 0.1, 'dynamics-focused'),
        (0.5, 0.5, 'balanced'),
        (0.1, 0.9, 'localization-focused')
    ]
    
    n_seeds = 10
    n_sensors = data_dict['sensor_positions'].shape[0]
    results = {}
    
    for lambda_dyn, lambda_loc, name in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {name}")
        print(f"λ_dyn={lambda_dyn}, λ_loc={lambda_loc}")
        print('='*60)
        
        config_results = []
        
        for seed in range(n_seeds):
            result = run_single_joint_experiment(
                lambda_dyn, lambda_loc, data_dict, seed, device, n_sensors
            )
            config_results.append(result)
            
            print(f"Seed {seed}: Loc RMSE={result['loc_rmse']:.2f}m, "
                  f"Dyn MAE={result['dyn_mae']:.4f}")
        
        # Aggregate statistics
        loc_rmses = [r['loc_rmse'] for r in config_results]
        dyn_maes = [r['dyn_mae'] for r in config_results]
        jaccards = [r['jaccard'] for r in config_results if r['jaccard'] is not None]
        
        results[name] = {
            'lambda_dyn': lambda_dyn,
            'lambda_loc': lambda_loc,
            'loc_rmse_mean': float(np.mean(loc_rmses)),
            'loc_rmse_std': float(np.std(loc_rmses)),
            'dyn_mae_mean': float(np.mean(dyn_maes)),
            'dyn_mae_std': float(np.std(dyn_maes)),
            'jaccard_mean': float(np.mean(jaccards)) if jaccards else None,
            'jaccard_std': float(np.std(jaccards)) if jaccards else None,
            'individual_results': config_results
        }
        
        print(f"\n{name} Summary:")
        print(f"  Localization RMSE: {np.mean(loc_rmses):.2f} ± {np.std(loc_rmses):.2f}m")
        print(f"  Dynamics MAE: {np.mean(dyn_maes):.4f} ± {np.std(dyn_maes):.4f}")
        if jaccards:
            print(f"  Jaccard Similarity: {np.mean(jaccards):.2f} ± {np.std(jaccards):.2f}")
    
    # Save results
    output_path = Path(output_dir) / 'joint_training_results.json'
    
    # Convert numpy types to Python native types for JSON serialization
    results_serializable = {}
    for config_name, config_results in results.items():
        results_serializable[config_name] = {
            'lambda_dyn': float(config_results['lambda_dyn']),
            'lambda_loc': float(config_results['lambda_loc']),
            'loc_rmse_mean': float(config_results['loc_rmse_mean']),
            'loc_rmse_std': float(config_results['loc_rmse_std']),
            'dyn_mae_mean': float(config_results['dyn_mae_mean']),
            'dyn_mae_std': float(config_results['dyn_mae_std']),
            'jaccard_mean': float(config_results['jaccard_mean']) if config_results['jaccard_mean'] is not None else None,
            'jaccard_std': float(config_results['jaccard_std']) if config_results['jaccard_std'] is not None else None,
            'individual_results': [
                {k: float(v) if isinstance(v, (np.floating, np.integer, float)) else v 
                 for k, v in res.items() if k != 'history'}
                for res in config_results['individual_results']
            ]
        }
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n\nResults saved to {output_path}")
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./s12_results',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_dict = torch.load(args.data_path, weights_only=False)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run experiments
    results = run_all_joint_experiments(data_dict, device, args.output_dir)
    
    print("\n" + "="*80)
    print("ALL JOINT TRAINING EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
