#!/usr/bin/env python3
"""
Localizer Architectures for Table 1 Comparison

Implements the 4 localizer architectures:
1. MLP (no structure) - baseline
2. Green's Function - physics-based attention
3. Learned Attention - transformer-style
4. NRI-Graph - data-driven structure (requires NRI)

Author: NTJ
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# 1. MLP LOCALIZER (Baseline - No Structure)
# ============================================================================

class MLPLocalizer(nn.Module):
    """
    Baseline MLP that concatenates all sensor readings and positions.
    
    Represents H_full with O(n²) sample complexity.
    No explicit relational structure.
    """
    
    def __init__(self, n_sensors: int, hidden_dim: int = 64, 
                 n_queries: int = 32, bounds: Tuple = (50, 30, 3)):
        super().__init__()
        
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        self.bounds = torch.tensor(bounds, dtype=torch.float32)
        
        # Input: readings (n) + positions (3n) = 4n
        input_dim = 4 * n_sensors
        
        # MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Query points (learnable grid of potential source locations)
        self.register_buffer('query_points', self._init_query_points(n_queries, bounds))
        
        # Output head: predict scores over query points
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_queries)
        )
        
        # Scaling for numerical stability
        self.register_buffer('length_scale', torch.tensor(10.0))
        self.register_buffer('concentration_scale', torch.tensor(1000.0))
    
    def _init_query_points(self, n_queries: int, bounds: Tuple) -> torch.Tensor:
        """Initialize query points as uniform grid"""
        # Create grid - for flat domains (parking), use more x,y than z
        if bounds[2] < 5:  # Flat domain
            n_z = 2
            n_xy = int(np.ceil(np.sqrt(n_queries / n_z)))
        else:
            n_side = int(np.ceil(n_queries ** (1/3)))
            n_xy = n_side
            n_z = max(2, n_side // 2)
        
        x = torch.linspace(0, bounds[0], n_xy)
        y = torch.linspace(0, bounds[1], n_xy)
        z = torch.linspace(0, bounds[2], n_z)
        
        # Meshgrid
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # Ensure exactly n_queries points
        if len(points) > n_queries:
            indices = torch.randperm(len(points))[:n_queries]
            points = points[indices]
        elif len(points) < n_queries:
            # Pad with additional random points
            n_extra = n_queries - len(points)
            extra_points = torch.rand(n_extra, 3)
            extra_points[:, 0] *= bounds[0]
            extra_points[:, 1] *= bounds[1]
            extra_points[:, 2] *= bounds[2]
            points = torch.cat([points, extra_points], dim=0)
        
        return points
    
    def forward(self, readings: torch.Tensor, sensor_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            readings: (batch, n_sensors) or (n_sensors,)
            sensor_positions: (batch, n_sensors, 3) or (n_sensors, 3)
        
        Returns:
            predicted_source: (batch, 3) or (3,)
        """
        # Handle unbatched input
        if readings.dim() == 1:
            readings = readings.unsqueeze(0)
            sensor_positions = sensor_positions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = readings.shape[0]
        device = readings.device
        
        # Normalize inputs
        readings_norm = readings / self.concentration_scale
        positions_flat = sensor_positions.reshape(batch_size, -1) / self.length_scale
        
        # Concatenate: [r1, ..., rn, p1x, p1y, p1z, ..., pnx, pny, pnz]
        x = torch.cat([readings_norm, positions_flat], dim=-1)
        
        # Encode
        features = self.encoder(x)
        
        # Predict scores over query points
        scores = self.output_head(features)  # (batch, n_queries)
        
        # Soft argmax: weighted average of query points
        weights = F.softmax(scores, dim=-1)  # (batch, n_queries)
        query_points = self.query_points.to(device)  # (n_queries, 3)
        
        predicted_source = torch.einsum('bq,qd->bd', weights, query_points)
        
        if squeeze_output:
            predicted_source = predicted_source.squeeze(0)
        
        return predicted_source
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 2. GREEN'S FUNCTION LOCALIZER (Physics-Based Attention)
# ============================================================================

class GreensFunctionLocalizer(nn.Module):
    """
    Localizer with physics-based attention structure.
    
    Attention weights: α_ji ∝ exp(-||p_i - q_j|| / λ)
    where λ is the diffusion length scale.
    
    Represents H_G with sparse physics prior.
    """
    
    def __init__(self, n_sensors: int, hidden_dim: int = 64,
                 n_queries: int = 32, bounds: Tuple = (50, 30, 3),
                 lambda_diffusion: float = 5.0):
        super().__init__()
        
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        self.lambda_diffusion = lambda_diffusion
        
        # Query points
        self.register_buffer('query_points', self._init_query_points(n_queries, bounds))
        
        # Sensor encoder: process (position, reading) for each sensor
        self.sensor_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # (x, y, z, reading)
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message aggregation
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output head: predict score for each query
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Scaling
        self.register_buffer('length_scale', torch.tensor(10.0))
        self.register_buffer('concentration_scale', torch.tensor(1000.0))
    
    def _init_query_points(self, n_queries: int, bounds: Tuple) -> torch.Tensor:
        """Initialize query points as uniform grid"""
        # Create grid - for flat domains (parking), use more x,y than z
        if bounds[2] < 5:  # Flat domain
            n_z = 2
            n_xy = int(np.ceil(np.sqrt(n_queries / n_z)))
        else:
            n_side = int(np.ceil(n_queries ** (1/3)))
            n_xy = n_side
            n_z = max(2, n_side // 2)
        
        x = torch.linspace(0, bounds[0], n_xy)
        y = torch.linspace(0, bounds[1], n_xy)
        z = torch.linspace(0, bounds[2], n_z)
        
        # Meshgrid
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # Ensure exactly n_queries points
        if len(points) > n_queries:
            indices = torch.randperm(len(points))[:n_queries]
            points = points[indices]
        elif len(points) < n_queries:
            # Pad with additional random points
            n_extra = n_queries - len(points)
            extra_points = torch.rand(n_extra, 3)
            extra_points[:, 0] *= bounds[0]
            extra_points[:, 1] *= bounds[1]
            extra_points[:, 2] *= bounds[2]
            points = torch.cat([points, extra_points], dim=0)
        
        return points
    
    def _compute_physics_attention(self, sensor_positions: torch.Tensor,
                                   query_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-based attention weights.
        
        Args:
            sensor_positions: (batch, n_sensors, 3)
            query_positions: (n_queries, 3)
        
        Returns:
            attention: (batch, n_queries, n_sensors)
        """
        batch_size = sensor_positions.shape[0]
        device = sensor_positions.device
        
        # Expand dimensions for broadcasting
        # sensors: (batch, n_sensors, 1, 3)
        # queries: (1, 1, n_queries, 3)
        sensors = sensor_positions.unsqueeze(2)
        queries = query_positions.unsqueeze(0).unsqueeze(0).to(device)
        
        # Compute distances: (batch, n_sensors, n_queries)
        distances = torch.norm(sensors - queries, dim=-1).permute(0, 2, 1)
        
        # Green's function: exp(-distance / lambda)
        attention_logits = -distances / self.lambda_diffusion
        
        # Softmax over sensors for each query
        attention = F.softmax(attention_logits, dim=-1)  # (batch, n_queries, n_sensors)
        
        return attention
    
    def forward(self, readings: torch.Tensor, sensor_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            readings: (batch, n_sensors) or (n_sensors,)
            sensor_positions: (batch, n_sensors, 3) or (n_sensors, 3)
        
        Returns:
            predicted_source: (batch, 3) or (3,)
        """
        if readings.dim() == 1:
            readings = readings.unsqueeze(0)
            sensor_positions = sensor_positions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = readings.shape[0]
        device = readings.device
        
        # Normalize
        readings_norm = readings / self.concentration_scale
        positions_norm = sensor_positions / self.length_scale
        
        # Encode each sensor: (reading, position)
        sensor_input = torch.cat([positions_norm, readings_norm.unsqueeze(-1)], dim=-1)
        sensor_features = self.sensor_encoder(sensor_input)  # (batch, n_sensors, hidden_dim)
        
        # Compute physics-based attention
        query_points = self.query_points.to(device)
        attention = self._compute_physics_attention(sensor_positions, query_points)
        # attention: (batch, n_queries, n_sensors)
        
        # Aggregate features using attention
        # (batch, n_queries, n_sensors) @ (batch, n_sensors, hidden_dim)
        # = (batch, n_queries, hidden_dim)
        aggregated = torch.bmm(attention, sensor_features)
        
        # Process aggregated features
        messages = self.message_net(aggregated)  # (batch, n_queries, hidden_dim)
        
        # Predict score for each query
        scores = self.output_head(messages).squeeze(-1)  # (batch, n_queries)
        
        # Soft argmax
        weights = F.softmax(scores, dim=-1)
        predicted_source = torch.einsum('bq,qd->bd', weights, query_points)
        
        if squeeze_output:
            predicted_source = predicted_source.squeeze(0)
        
        return predicted_source
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 3. LEARNED ATTENTION LOCALIZER (Transformer-Style)
# ============================================================================

class LearnedAttentionLocalizer(nn.Module):
    """
    Transformer-style localizer with learned attention.
    
    Attention: α_ji ∝ exp(q_j^T k_i / √d)
    where queries and keys are learned projections.
    
    Represents H_full (no structural constraint).
    """
    
    def __init__(self, n_sensors: int, hidden_dim: int = 64,
                 n_queries: int = 32, bounds: Tuple = (50, 30, 3),
                 n_heads: int = 4):
        super().__init__()
        
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        # Query points
        self.register_buffer('query_points', self._init_query_points(n_queries, bounds))
        
        # Sensor encoder
        self.sensor_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # (x, y, z, reading)
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Query embedding (for each potential source location)
        self.query_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # (x, y, z)
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head attention components
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # Query projection
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # Key projection
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # Value projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)  # Output projection
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Scaling
        self.register_buffer('length_scale', torch.tensor(10.0))
        self.register_buffer('concentration_scale', torch.tensor(1000.0))
    
    def _init_query_points(self, n_queries: int, bounds: Tuple) -> torch.Tensor:
        """Initialize query points as uniform grid"""
        # Create grid - for flat domains (parking), use more x,y than z
        if bounds[2] < 5:  # Flat domain
            n_z = 2
            n_xy = int(np.ceil(np.sqrt(n_queries / n_z)))
        else:
            n_side = int(np.ceil(n_queries ** (1/3)))
            n_xy = n_side
            n_z = max(2, n_side // 2)
        
        x = torch.linspace(0, bounds[0], n_xy)
        y = torch.linspace(0, bounds[1], n_xy)
        z = torch.linspace(0, bounds[2], n_z)
        
        # Meshgrid
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # Ensure exactly n_queries points
        if len(points) > n_queries:
            indices = torch.randperm(len(points))[:n_queries]
            points = points[indices]
        elif len(points) < n_queries:
            # Pad with additional random points
            n_extra = n_queries - len(points)
            extra_points = torch.rand(n_extra, 3)
            extra_points[:, 0] *= bounds[0]
            extra_points[:, 1] *= bounds[1]
            extra_points[:, 2] *= bounds[2]
            points = torch.cat([points, extra_points], dim=0)
        
        return points
    
    def _multi_head_attention(self, queries: torch.Tensor, keys: torch.Tensor,
                              values: torch.Tensor) -> torch.Tensor:
        """
        Multi-head attention.
        
        Args:
            queries: (batch, n_queries, hidden_dim)
            keys: (batch, n_sensors, hidden_dim)
            values: (batch, n_sensors, hidden_dim)
        
        Returns:
            output: (batch, n_queries, hidden_dim)
        """
        batch_size = queries.shape[0]
        
        # Project
        Q = self.W_q(queries)  # (batch, n_queries, hidden_dim)
        K = self.W_k(keys)     # (batch, n_sensors, hidden_dim)
        V = self.W_v(values)   # (batch, n_sensors, hidden_dim)
        
        # Reshape for multi-head attention
        # (batch, n_queries, n_heads, head_dim) -> (batch, n_heads, n_queries, head_dim)
        Q = Q.view(batch_size, self.n_queries, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.n_sensors, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.n_sensors, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        # (batch, n_heads, n_queries, head_dim) @ (batch, n_heads, head_dim, n_sensors)
        # = (batch, n_heads, n_queries, n_sensors)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        # (batch, n_heads, n_queries, n_sensors) @ (batch, n_heads, n_sensors, head_dim)
        # = (batch, n_heads, n_queries, head_dim)
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        # (batch, n_queries, n_heads, head_dim) -> (batch, n_queries, hidden_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, self.n_queries, self.hidden_dim)
        
        # Output projection
        out = self.W_o(out)
        
        return out
    
    def forward(self, readings: torch.Tensor, sensor_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            readings: (batch, n_sensors) or (n_sensors,)
            sensor_positions: (batch, n_sensors, 3) or (n_sensors, 3)
        
        Returns:
            predicted_source: (batch, 3) or (3,)
        """
        if readings.dim() == 1:
            readings = readings.unsqueeze(0)
            sensor_positions = sensor_positions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = readings.shape[0]
        device = readings.device
        
        # Normalize
        readings_norm = readings / self.concentration_scale
        positions_norm = sensor_positions / self.length_scale
        
        # Encode sensors (keys and values)
        sensor_input = torch.cat([positions_norm, readings_norm.unsqueeze(-1)], dim=-1)
        sensor_features = self.sensor_encoder(sensor_input)  # (batch, n_sensors, hidden_dim)
        
        # Encode query points (queries)
        query_points = self.query_points.to(device)
        query_points_norm = query_points / self.length_scale
        query_points_batch = query_points_norm.unsqueeze(0).expand(batch_size, -1, -1)
        query_features = self.query_encoder(query_points_batch)  # (batch, n_queries, hidden_dim)
        
        # Multi-head attention
        attended = self._multi_head_attention(query_features, sensor_features, sensor_features)
        # (batch, n_queries, hidden_dim)
        
        # Predict score for each query
        scores = self.output_head(attended).squeeze(-1)  # (batch, n_queries)
        
        # Soft argmax
        weights = F.softmax(scores, dim=-1)
        predicted_source = torch.einsum('bq,qd->bd', weights, query_points)
        
        if squeeze_output:
            predicted_source = predicted_source.squeeze(0)
        
        return predicted_source
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 4. NRI-GRAPH LOCALIZER (Data-Driven Structure) - Placeholder
# ============================================================================

class NRIGraphLocalizer(nn.Module):
    """
    Localizer with NRI-learned graph structure.
    
    Attention constrained to learned graph: α_ji = p(e_ij) if e_ij ∈ G
    
    This requires:
    1. Pre-trained NRI model to learn graph structure from CFD dynamics
    2. Graph adjacency matrix or edge probabilities
    
    For now, this is a placeholder. Implementation requires:
    - NRI training on CFD time series
    - Edge probability extraction
    - Graph-constrained attention
    """
    def __init__(self, n_sensors: int, hidden_dim: int = 64,
             n_queries: int = 32, bounds: Tuple = (50, 30, 3),
             edge_prob: Optional[torch.Tensor] = None,
             sparsity_threshold: float = 0.3):

        super().__init__()
        
        self.n_sensors = n_sensors
        self.n_queries = n_queries
        self.hidden_dim = hidden_dim
        self.sparsity_threshold = sparsity_threshold
        
        # Query points
        self.register_buffer('query_points', self._init_query_points(n_queries, bounds))
        
        # NRI edge probabilities over query nodes: (Q, Q)
        if edge_prob is None:
            print("WARNING: No NRI edge_prob provided. Using random fallback.")
            Q = n_queries
            edge_prob = torch.zeros(Q, Q)
            for i in range(Q):
                nbrs = torch.randperm(Q)[:5]
                edge_prob[i, nbrs] = 1.0
        else:
            assert edge_prob.shape == (n_queries, n_queries), \
                f"edge_prob must be (Q,Q) = ({n_queries},{n_queries}), got {tuple(edge_prob.shape)}"

        self.register_buffer("edge_prob", edge_prob)
        
        # Sensor encoder
        self.sensor_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Scaling
        self.register_buffer('length_scale', torch.tensor(10.0))
        self.register_buffer('concentration_scale', torch.tensor(1000.0))
    
    def _init_query_points(self, n_queries: int, bounds: Tuple) -> torch.Tensor:
        """Initialize query points as uniform grid"""
        # Create grid - for flat domains (parking), use more x,y than z
        if bounds[2] < 5:  # Flat domain
            n_z = 2
            n_xy = int(np.ceil(np.sqrt(n_queries / n_z)))
        else:
            n_side = int(np.ceil(n_queries ** (1/3)))
            n_xy = n_side
            n_z = max(2, n_side // 2)
        
        x = torch.linspace(0, bounds[0], n_xy)
        y = torch.linspace(0, bounds[1], n_xy)
        z = torch.linspace(0, bounds[2], n_z)
        
        # Meshgrid
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # Ensure exactly n_queries points
        if len(points) > n_queries:
            indices = torch.randperm(len(points))[:n_queries]
            points = points[indices]
        elif len(points) < n_queries:
            # Pad with additional random points
            n_extra = n_queries - len(points)
            extra_points = torch.rand(n_extra, 3)
            extra_points[:, 0] *= bounds[0]
            extra_points[:, 1] *= bounds[1]
            extra_points[:, 2] *= bounds[2]
            points = torch.cat([points, extra_points], dim=0)
        
        return points
    
    def _init_distance_graph(self, n_queries: int, n_sensors: int, 
                            bounds: Tuple, k: int = 5) -> torch.Tensor:
        """
        Create fallback distance-based sparse graph.
        For each query, connect to k nearest potential sensor locations.
        """
        # This is a simplified placeholder
        # In reality, NRI would learn this from CFD dynamics
        
        # For now, create a sparse random graph with average degree k
        graph = torch.zeros(n_queries, n_sensors)
        for j in range(n_queries):
            # Connect each query to k random sensors
            sensor_indices = torch.randperm(n_sensors)[:k]
            graph[j, sensor_indices] = 1.0
        
        return graph
    
    def forward(self, readings: torch.Tensor, sensor_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            readings: (batch, n_sensors) or (n_sensors,)
            sensor_positions: (batch, n_sensors, 3) or (n_sensors, 3)
        
        Returns:
            predicted_source: (batch, 3) or (3,)
        """
        if readings.dim() == 1:
            readings = readings.unsqueeze(0)
            sensor_positions = sensor_positions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = readings.shape[0]
        device = readings.device
        
        # Normalize
        readings_norm = readings / self.concentration_scale
        positions_norm = sensor_positions / self.length_scale
        
        # Encode sensors
        sensor_input = torch.cat([positions_norm, readings_norm.unsqueeze(-1)], dim=-1)
        sensor_features = self.sensor_encoder(sensor_input)  # (batch, n_sensors, hidden_dim)
        
        # Graph-constrained attention from learned edge probabilities
        # edge_prob: (Q, Q) over query nodes
        edge_prob = self.edge_prob.to(device)  # (Q, Q)
        Q = self.n_queries
        S = self.n_sensors

        # Map each sensor position to nearest query node index
        # sensor_positions: (B, S, 3), query_points: (Q, 3)
        qp = self.query_points.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,Q,3)
        sp = sensor_positions.unsqueeze(2)                            # (B,S,1,3)
        d2 = ((sp - qp) ** 2).sum(dim=-1)                             # (B,S,Q)
        nearest = d2.argmin(dim=-1)                                   # (B,S) in [0..Q-1]

        # Build attention: attention[b, q, s] = edge_prob[node(sensor_s), q]
        rows = edge_prob[nearest]                                     # (B,S,Q)
        attention = rows.permute(0, 2, 1).contiguous()                 # (B,Q,S)

        # Optional thresholding (soft mask)
        if self.sparsity_threshold is not None:
            attention = attention * (attention >= self.sparsity_threshold).float()

        # Normalize over sensors for each query
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Aggregate features
        aggregated = torch.bmm(attention, sensor_features)  # (batch, n_queries, hidden_dim)
        
        # Process
        messages = self.message_net(aggregated)
        
        # Predict scores
        scores = self.output_head(messages).squeeze(-1)  # (batch, n_queries)
        
        # Soft argmax
        weights = F.softmax(scores, dim=-1)
        query_points = self.query_points.to(device)
        predicted_source = torch.einsum('bq,qd->bd', weights, query_points)
        
        if squeeze_output:
            predicted_source = predicted_source.squeeze(0)
        
        return predicted_source
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def update_graph(self, new_edge_prob: torch.Tensor):
        self.edge_prob = new_edge_prob.to(self.edge_prob.device)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_localizer(architecture: str, n_sensors: int, hidden_dim: int = 64,
                     n_queries: int = 32, bounds: Tuple = (50, 30, 3),
                     **kwargs) -> nn.Module:
    """
    Factory function to create localizer by name.
    
    Args:
        architecture: One of ['mlp', 'greens', 'attention', 'nri']
        n_sensors: Number of sensors
        hidden_dim: Hidden dimension
        n_queries: Number of query points
        bounds: Domain bounds (x, y, z)
        **kwargs: Additional architecture-specific arguments
    
    Returns:
        localizer: Instantiated localizer module
    """
    architecture = architecture.lower()
    
    if architecture == 'mlp':
        return MLPLocalizer(n_sensors, hidden_dim, n_queries, bounds)
    
    elif architecture in ['greens', 'green', 'greens_function']:
        lambda_diffusion = kwargs.get('lambda_diffusion', 5.0)
        return GreensFunctionLocalizer(n_sensors, hidden_dim, n_queries, bounds, lambda_diffusion)
    
    elif architecture in ['attention', 'learned_attention', 'transformer']:
        n_heads = kwargs.get('n_heads', 4)
        return LearnedAttentionLocalizer(n_sensors, hidden_dim, n_queries, bounds, n_heads)
    
    elif architecture in ['nri', 'nri_graph', 'graph']:
        edge_prob = kwargs.get('edge_prob', None)
        sparsity_threshold = kwargs.get('sparsity_threshold', 0.3)
        return NRIGraphLocalizer(n_sensors, hidden_dim, n_queries, bounds,
                                edge_prob=edge_prob,
                                sparsity_threshold=sparsity_threshold)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def print_architecture_summary():
    """Print summary of all architectures"""
    print("\n" + "="*70)
    print("LOCALIZER ARCHITECTURES FOR TABLE 1")
    print("="*70)
    
    n_sensors = 15
    
    architectures = {
        'MLP': create_localizer('mlp', n_sensors),
        'Greens Function': create_localizer('greens', n_sensors),
        'Learned Attention': create_localizer('attention', n_sensors),
        'NRI-Graph': create_localizer('nri', n_sensors)
    }
    
    print(f"\nConfiguration: {n_sensors} sensors, 64 queries\n")
    
    for name, model in architectures.items():
        n_params = model.count_parameters()
        print(f"{name:20s}: {n_params:,} parameters")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("Localizer architectures module loaded.")
    print_architecture_summary()
    
    # Test forward pass
    print("\nTesting forward pass...")
    n_sensors = 15
    batch_size = 4
    
    model = create_localizer('mlp', n_sensors)
    readings = torch.randn(batch_size, n_sensors)
    positions = torch.rand(batch_size, n_sensors, 3) * torch.tensor([50, 30, 3])
    
    output = model(readings, positions)
    print(f"Input shape: readings {readings.shape}, positions {positions.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: x∈[{output[:, 0].min():.1f}, {output[:, 0].max():.1f}], " +
          f"y∈[{output[:, 1].min():.1f}, {output[:, 1].max():.1f}], " +
          f"z∈[{output[:, 2].min():.1f}, {output[:, 2].max():.1f}]")
    
    print("\n✓ All architectures loaded successfully!")
