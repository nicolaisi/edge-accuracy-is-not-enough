#!/usr/bin/env python3
"""
Train NRI on SPATIAL correlations for source localization.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def create_query_points(n_queries, bounds):
    """Create query points as uniform grid"""
    # For flat domains (parking), use more x,y than z
    if bounds[2] < 5:
        n_z = 2
        n_xy = int(np.ceil(np.sqrt(n_queries / n_z)))
    else:
        n_side = int(np.ceil(n_queries ** (1/3)))
        n_xy = n_side
        n_z = max(2, n_side // 2)
    
    x = np.linspace(0, bounds[0], n_xy)
    y = np.linspace(0, bounds[1], n_xy)
    z = np.linspace(0, bounds[2], n_z)
    
    # Meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    
    # Ensure exactly n_queries points
    if len(points) > n_queries:
        indices = np.random.permutation(len(points))[:n_queries]
        points = points[indices]
    elif len(points) < n_queries:
        # Pad with random points
        n_extra = n_queries - len(points)
        extra = np.random.rand(n_extra, 3)
        extra[:, 0] *= bounds[0]
        extra[:, 1] *= bounds[1]
        extra[:, 2] *= bounds[2]
        points = np.vstack([points, extra])
    
    return points.astype(np.float32)


def interpolate_at_points(scenario, query_points):
    """Extract concentration at query points"""
    from scipy.interpolate import RegularGridInterpolator
    
    grid = scenario['grid']
    field = scenario['concentration_field']
    
    interpolator = RegularGridInterpolator(
        (grid['x'], grid['y'], grid['z']),
        field, method='linear', bounds_error=False, fill_value=0.0
    )
    
    return interpolator(query_points).astype(np.float32)


def build_spatial_dataset(scenarios, query_points):
    """Build dataset of spatial concentration patterns"""
    X = []
    print("Building spatial dataset...")
    for scenario in tqdm(scenarios):
        conc = interpolate_at_points(scenario, query_points)
        X.append(conc)
    
    X = np.array(X, dtype=np.float32)
    # Log-normalize to handle wide range
    X = np.log1p(X / 1000.0)
    return X


class SpatialEncoder(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
    
    def forward(self, conc, pos):
        """
        conc: (B, N)
        pos: (N, 3)
        """
        B, N = conc.shape
        pos = pos.unsqueeze(0).expand(B, N, 3)
        x = torch.cat([conc.unsqueeze(-1), pos], dim=-1)
        return self.net(x)


class EdgeNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 2)
        )
    
    def forward(self, z):
        B, N, H = z.shape
        zi = z.unsqueeze(2).expand(B, N, N, H)
        zj = z.unsqueeze(1).expand(B, N, N, H)
        logits = self.net(torch.cat([zi, zj], dim=-1))
        # Mask self-edges
        eye = torch.eye(N, device=z.device).bool()
        logits[:, eye, :] = -1e9
        return logits


class Decoder(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(1, hidden), nn.ReLU())
        self.update = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x, edge_prob):
        B, N = x.shape
        m = self.msg(x.unsqueeze(-1))
        m = m.unsqueeze(1).expand(B, N, N, -1)
        agg = (m * edge_prob.unsqueeze(-1)).sum(dim=2)
        return self.update(agg).squeeze(-1)


def train_spatial_nri(scenarios, query_points, n_epochs=100, 
                     hidden=64, lr=1e-3, lambda_sparse=1e-3, device='cpu'):
    """Train NRI on spatial correlations"""
    
    # Build dataset
    X = build_spatial_dataset(scenarios, query_points)
    X = torch.tensor(X, device=device)
    pos = torch.tensor(query_points, device=device, dtype=torch.float32)
    
    B, N = X.shape
    print(f"\nDataset: {B} scenarios, {N} queries")
    print(f"Concentration range: [{X.min():.3f}, {X.max():.3f}] (log-scaled)")
    
    # Split
    n_train = int(0.8 * B)
    X_train, X_val = X[:n_train], X[n_train:]
    
    # Models
    enc = SpatialEncoder(hidden).to(device)
    edge_net = EdgeNet(hidden).to(device)
    dec = Decoder(hidden).to(device)
    
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(edge_net.parameters()) + list(dec.parameters()),
        lr=lr, weight_decay=1e-5
    )
    
    best_val, best_edges = float('inf'), None
    
    print("\nTraining...")
    for ep in range(n_epochs):
        enc.train(); edge_net.train(); dec.train()
        
        opt.zero_grad()
        z = enc(X_train, pos)
        logits = edge_net(z)
        edge_prob = F.softmax(logits, dim=-1)[..., 1]
        
        X_pred = dec(X_train, edge_prob)
        loss_rec = F.mse_loss(X_pred, X_train)
        loss_sp = edge_prob.mean()
        loss = loss_rec + lambda_sparse * loss_sp
        
        loss.backward()
        opt.step()
        
        if (ep+1) % 10 == 0 or ep == 0:
            with torch.no_grad():
                enc.eval(); edge_net.eval(); dec.eval()
                z_v = enc(X_val, pos)
                edge_v = F.softmax(edge_net(z_v), dim=-1)[..., 1]
                pred_v = dec(X_val, edge_v)
                val_loss = F.mse_loss(pred_v, X_val)
                
                if val_loss < best_val:
                    best_val = val_loss
                    best_edges = edge_prob.mean(0)
                
                deg = edge_prob.mean(0).sum(-1).mean().item()
                print(f"Epoch {ep+1:3d}/{n_epochs} | "
                      f"Train: {loss_rec:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"Sparsity: {loss_sp:.4f} | "
                      f"Deg: {deg:.1f}")
    
    return best_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='cfd_data/')
    parser.add_argument('--domain', default='cfd', choices=['cfd', 'acoustic'],
                        help="Domain: 'cfd' or 'acoustic'. Default: cfd")
    parser.add_argument('--n_scenarios', type=int, default=16)
    parser.add_argument('--scenarios-file', type=str, default=None,
                        help="Path to a pickle of pre-selected scenarios (e.g. an OOD train split "
                             "saved by make_table6.py) to train on, overriding --data_dir/--n_scenarios. "
                             "Needed whenever the downstream task uses a *filtered* subset rather than "
                             "'the first N scenarios' (--n_scenarios alone assumes the latter).")
    parser.add_argument('--n_queries', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lambda_sparse', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default=None,
                        help='Output checkpoint path. Default: <output_dir>/nri_edge_probs_spatial_<n>.pt')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for checkpoint (default: checkpoints/cfd/ or checkpoints/acoustic/, derived from --domain).')
    parser.add_argument('--analysis_topk', type=int, default=None,
                        help='If set, analyze sparsity by keeping top-k outgoing edges per node (excludes self-edges).')
    parser.add_argument('--analysis_threshold', type=float, default=0.2,
                        help='Edge probability threshold for analysis if --analysis_topk is not set.')
    args = parser.parse_args()
    
    print("="*70)
    print("SPATIAL NRI TRAINING")
    print("="*70)

    # Domain-aware loading
    from domain_config import get_domain_config
    domain_cfg = get_domain_config(args.domain)
    bounds = domain_cfg["bounds"]

    print(f"\nDomain: {domain_cfg['name']}")
    print(f"Bounds: {bounds}")

    if args.scenarios_file:
        import pickle
        print(f"Loading pre-selected scenarios from: {args.scenarios_file}")
        with open(args.scenarios_file, 'rb') as f:
            scenarios = pickle.load(f)
        args.n_scenarios = len(scenarios)
        print(f"✓ Loaded {len(scenarios)} scenarios")
    else:
        print(f"Loading {args.n_scenarios} scenarios from {args.data_dir}...")
        if args.domain == "acoustic":
            from acoustic_loader import load_acoustic_scenarios_as_cfd
            scenarios = load_acoustic_scenarios_as_cfd(args.data_dir, args.n_scenarios)
        else:
            from load_cfd_netcdf import load_cfd_from_netcdf_quick
            scenarios = load_cfd_from_netcdf_quick(args.data_dir, args.n_scenarios)
    
    # Create query points (same logic as localizer)
    print(f"\nCreating {args.n_queries} query points...")
    query_points = create_query_points(args.n_queries, bounds)
    
    print(f"Query points: {query_points.shape}")
    print(f"Domain: [{query_points[:, 0].min():.1f}, {query_points[:, 0].max():.1f}] × "
          f"[{query_points[:, 1].min():.1f}, {query_points[:, 1].max():.1f}] × "
          f"[{query_points[:, 2].min():.1f}, {query_points[:, 2].max():.1f}]")
    
    # Train NRI
    edge_prob = train_spatial_nri(
        scenarios, query_points, args.epochs,
        args.hidden, lr=1e-3, lambda_sparse=args.lambda_sparse,
        device=args.device
    )
    
    # Analyze structure
    print("\n" + "="*70)
    print("LEARNED STRUCTURE ANALYSIS")
    print("="*70)
    
    ep_np = edge_prob.detach().cpu().numpy()
    N = ep_np.shape[0]

    def _topk_mask(ep: np.ndarray, k: int) -> np.ndarray:
        """Boolean mask (N,N) keeping top-k outgoing edges per node (excluding self)."""
        mask = np.zeros((N, N), dtype=bool)
        for i in range(N):
            row = ep[i].copy()
            row[i] = -np.inf  # exclude self-edge
            # argpartition is O(N) and order within top-k doesn't matter for masking
            topj = np.argpartition(row, -k)[-k:]
            mask[i, topj] = True
        return mask

    # Distance matrix between query points (meters)
    distances = np.sqrt(((query_points[:, None, :] - query_points[None, :, :]) ** 2).sum(axis=-1))

    if args.analysis_topk is not None:
        k = int(args.analysis_topk)
        if k <= 0 or k >= N:
            raise ValueError(f"--analysis_topk must be in [1, {N-1}] (got {k})")

        active = _topk_mask(ep_np, k)
        n_edges = int(active.sum())
        n_poss = N * (N - 1)
        avg_degree = n_edges / N  # exactly k by construction

        print(f"\nEdge statistics (topk={k}):")
        print(f"  Active edges: {n_edges}/{n_poss} ({100*n_edges/n_poss:.1f}%)")
        print(f"  Avg degree: {avg_degree:.1f}")
        print(f"  Edge weight range: [{ep_np.min():.4f}, {ep_np.max():.4f}]")
        print(f"  Mean edge probability: {ep_np.mean():.6f}")

        edge_dists = distances[active]
        non_edge_dists = distances[~active & ~np.eye(N, dtype=bool)]

    else:
        threshold = float(args.analysis_threshold)
        active = (ep_np > threshold)
        np.fill_diagonal(active, False)

        n_edges = int(active.sum())
        n_poss = N * (N - 1)
        avg_degree = n_edges / N

        print(f"\nEdge statistics (threshold={threshold}):")
        print(f"  Active edges: {n_edges}/{n_poss} ({100*n_edges/n_poss:.1f}%)")
        print(f"  Avg degree: {avg_degree:.1f}")
        print(f"  Edge weight range: [{ep_np.min():.4f}, {ep_np.max():.4f}]")
        print(f"  Mean edge probability: {ep_np.mean():.6f}")

        edge_dists = distances[active]
        non_edge_dists = distances[~active & ~np.eye(N, dtype=bool)]

    # Distance analysis (report only if we have at least 1 active edge)
    if edge_dists.size > 0:
        print(f"\nDistance analysis:")
        print(f"  Edges connect: {edge_dists.mean():.2f} ± {edge_dists.std():.2f} m")
        print(f"  Non-edges: {non_edge_dists.mean():.2f} ± {non_edge_dists.std():.2f} m")
        if edge_dists.mean() < non_edge_dists.mean():
            print("  ✓ Edges prefer nearby points (local structure)")
        else:
            print("  ⚠ Edges connect distant points (long-range)")
    else:
        print("\nDistance analysis: (no active edges under current analysis setting)")
    # Save checkpoint with domain-aware default path
    if args.out is not None:
        out_path = Path(args.out)
    else:
        if args.output_dir is not None:
            out_dir = Path(args.output_dir)
        else:
            out_dir = Path("checkpoints") / args.domain
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"nri_edge_probs_spatial_{len(scenarios)}.pt"

    torch.save({
        'edge_prob': edge_prob,
        'query_points': torch.tensor(query_points)
    }, str(out_path))
    
    print(f"\n✓ Saved spatial NRI to: {out_path}")
    print(f"\nNext steps:")
    print(f"  1. Run: python src/run_localizer_experiment.py --test --domain {args.domain} "
          f"--data_dir {args.data_dir}")
    print(f"  2. If improved: python src/run_localizer_experiment.py --full --domain {args.domain} "
          f"--data_dir {args.data_dir}")


if __name__ == '__main__':
    main()
