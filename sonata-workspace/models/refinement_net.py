"""
Refinement Network for Scene Completion

Densifies coarse diffusion output by predicting per-point offsets.
Takes (N, 3) coarse points, outputs (N * up_factor, 3) refined dense points.
Uses point-based MLPs (no MinkowskiEngine).
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree


class RefinementNetwork(nn.Module):
    """
    Point-based refinement: predicts up_factor offsets per input point.
    Output: input_points + offsets -> denser point cloud.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: list = [64, 128, 256, 128],
        up_factor: int = 6,
    ):
        """
        Args:
            in_channels: Input feature dim (3 for xyz)
            hidden_dims: MLP hidden dimensions
            up_factor: Number of output points per input point
        """
        super().__init__()
        self.up_factor = up_factor
        self.out_dim = up_factor * 3  # 3 offsets per output point

        layers = []
        dims = [in_channels] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
            ])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], self.out_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (N, 3) coarse point coordinates

        Returns:
            (N * up_factor, 3) refined point coordinates
        """
        feat = self.backbone(points)  # (N, hidden)
        offsets = self.head(feat)  # (N, up_factor * 3)
        offsets = offsets.view(-1, self.up_factor, 3)  # (N, up_factor, 3)
        refined = points.unsqueeze(1) + offsets  # (N, up_factor, 3)
        return refined.reshape(-1, 3)  # (N * up_factor, 3)


class RefinementNetworkWithContext(nn.Module):
    """
    Refinement with local context (k-NN features) for better detail.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: list = [64, 128, 256, 256, 128],
        up_factor: int = 6,
        num_neighbors: int = 16,
    ):
        super().__init__()
        self.up_factor = up_factor
        self.num_neighbors = num_neighbors

        # Local context: point + relative positions to neighbors
        context_dim = in_channels + num_neighbors * 3  # self + neighbor deltas
        dims = [context_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
            ])
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], up_factor * 3)

    def _get_neighbor_context(self, points: torch.Tensor) -> torch.Tensor:
        """Add k-NN relative positions as context."""
        pts_np = points.detach().cpu().numpy()
        tree = cKDTree(pts_np)
        k = min(self.num_neighbors + 1, len(pts_np))
        _, indices = tree.query(pts_np, k=k)
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
        indices = indices[:, 1:]  # Exclude self

        neighbors = points[indices]  # (N, k, 3)
        deltas = neighbors - points.unsqueeze(1)  # (N, k, 3)
        context = deltas.reshape(points.shape[0], -1)  # (N, k*3)
        return torch.cat([points, context], dim=-1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        context = self._get_neighbor_context(points)
        feat = self.backbone(context)
        offsets = self.head(feat).view(-1, self.up_factor, 3)
        refined = points.unsqueeze(1) + offsets
        return refined.reshape(-1, 3)


def chamfer_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduce: str = "mean",
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Symmetric Chamfer distance between two point clouds.
    pred: (N, 3), target: (M, 3)
    Uses chunking to limit memory for large point clouds.
    """
    n_pred, n_target = pred.shape[0], target.shape[0]

    def min_dist_single_dir(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """For each point in a, min distance to b."""
        mins = []
        for i in range(0, a.shape[0], chunk_size):
            chunk = a[i : i + chunk_size]  # (C, 3)
            diff = chunk.unsqueeze(1) - b.unsqueeze(0)  # (C, M, 3)
            dist = (diff ** 2).sum(-1)  # (C, M)
            mins.append(dist.min(1)[0])
        return torch.cat(mins, dim=0)

    min_p2t = min_dist_single_dir(pred, target)
    min_t2p = min_dist_single_dir(target, pred)

    if reduce == "mean":
        return (min_p2t.mean() + min_t2p.mean()) / 2
    return min_p2t, min_t2p
