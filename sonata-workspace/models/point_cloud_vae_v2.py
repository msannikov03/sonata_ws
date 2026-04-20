"""
PointCloudVAEv2: Improved Gaussian VAE for point cloud scene completion.

Changes over v1:
  - Encoder: max+mean pooling concatenation (512-d) instead of max-only (256-d)
  - Decoder: Two-stage FoldingNet instead of single linear projection
  - Default output: 8000 points (was 2048)

Trained with Chamfer reconstruction + KL divergence.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.refinement_net import chamfer_distance


# ---------------------------------------------------------------------------
# Folding decoder
# ---------------------------------------------------------------------------

class FoldingDecoder(nn.Module):
    """Two-stage folding decoder that warps a learned 2-D grid into 3-D.

    Stage 1 (coarse): concatenate latent + 2-D grid  -> 3-D coordinates.
    Stage 2 (refine): concatenate latent + coarse 3-D -> refined 3-D coordinates.
    """

    def __init__(self, latent_dim: int, num_points: int, hidden_dim: int = 512):
        super().__init__()
        self.num_points = num_points

        grid = self._create_grid(num_points)
        self.register_buffer("grid", grid)

        # Fold 1: (latent_dim + 2) -> 3
        self.fold1 = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        # Fold 2: (latent_dim + 3) -> 3
        self.fold2 = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _create_grid(num_points: int) -> torch.Tensor:
        """Uniform 2-D grid in [-1, 1]^2, trimmed to exactly *num_points*."""
        side = int(math.ceil(math.sqrt(num_points)))
        u = torch.linspace(-1.0, 1.0, side)
        v = torch.linspace(-1.0, 1.0, side)
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)
        return grid.reshape(-1, 2)[:num_points]           # (num_points, 2)

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) or (latent_dim,)
        Returns:
            points: (B, num_points, 3) or (num_points, 3)
        """
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        B = z.size(0)

        # Tile latent for every grid point: (B, N, D)
        z_tiled = z.unsqueeze(1).expand(-1, self.num_points, -1)
        # Grid: (N, 2) -> (B, N, 2)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)

        # Stage 1 – coarse fold
        inp1 = torch.cat([z_tiled, grid], dim=-1)          # (B, N, D+2)
        coarse = self.fold1(inp1)                           # (B, N, 3)

        # Stage 2 – refinement fold
        inp2 = torch.cat([z_tiled, coarse], dim=-1)         # (B, N, D+3)
        fine = self.fold2(inp2)                              # (B, N, 3)

        if single:
            fine = fine.squeeze(0)
        return fine


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class PointCloudVAEv2(nn.Module):
    """Gaussian VAE with PointNet encoder (max+mean pool) and FoldingNet decoder."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_decoded_points: int = 8000,
        encoder_widths: Optional[List[int]] = None,
        decoder_hidden: int = 512,
    ):
        super().__init__()
        if encoder_widths is None:
            encoder_widths = [64, 128, 256]

        self.latent_dim = latent_dim
        self.num_decoded_points = num_decoded_points

        # --- Encoder (shared MLP) ---
        enc_layers: List[nn.Module] = []
        d_in = 3
        for w in encoder_widths:
            enc_layers.extend([nn.Linear(d_in, w), nn.LayerNorm(w), nn.GELU()])
            d_in = w
        self.point_encoder = nn.Sequential(*enc_layers)

        # max + mean pooling => 2 * encoder_widths[-1]
        d_pool = 2 * encoder_widths[-1]
        self.fc_mu = nn.Linear(d_pool, latent_dim)
        self.fc_logvar = nn.Linear(d_pool, latent_dim)

        # --- Decoder (FoldingNet) ---
        self.decoder = FoldingDecoder(
            latent_dim=latent_dim,
            num_points=num_decoded_points,
            hidden_dim=decoder_hidden,
        )

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3)
        Returns:
            mu, logvar: each (latent_dim,)
        """
        if points.dim() != 2 or points.size(-1) != 3:
            raise ValueError(f"Expected (N, 3) points, got {tuple(points.shape)}")
        if points.size(0) == 0:
            z = points.new_zeros(self.latent_dim)
            return z, z.new_full((self.latent_dim,), -30.0)

        feat = self.point_encoder(points)                   # (N, C)
        max_pool = feat.max(dim=0).values                   # (C,)
        mean_pool = feat.mean(dim=0)                        # (C,)
        pooled = torch.cat([max_pool, mean_pool], dim=-1)   # (2C,)
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def encode_batched(
        self,
        points: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a flat concatenated batch of point clouds. Returns (B, latent_dim)."""
        mus, logvars = [], []
        for b in range(batch_size):
            mask = batch == b
            mu, lv = self.encode(points[mask])
            mus.append(mu)
            logvars.append(lv)
        return torch.stack(mus, 0), torch.stack(logvars, 0)

    # ------------------------------------------------------------------
    # Reparameterize
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (latent_dim,) or (B, latent_dim)
        Returns:
            (num_decoded_points, 3) or (B, num_decoded_points, 3)
        """
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3) single cloud
        Returns:
            recon: (K, 3), mu, logvar: (latent_dim,)
        """
        mu, logvar = self.encode(points)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def forward_batched(
        self,
        points: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N_tot, 3), batch: (N_tot,) indices in [0, B)
        Returns:
            recon: (B, K, 3), mu, logvar: (B, latent_dim)
        """
        mu, logvar = self.encode_batched(points, batch, batch_size)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Loss utilities (same API as v1)
# ---------------------------------------------------------------------------

def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)), averaged over batch and latent dims."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_reconstruction_chamfer(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        recon:  (B, K, 3) or (K, 3)
        target: (B, M, 3) or (M, 3)
        mu, logvar: (B, latent_dim) or (latent_dim,)
    Returns:
        total, chamfer_term, kl_term
    """
    if recon.dim() == 2:
        recon = recon.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    if logvar.dim() == 1:
        logvar = logvar.unsqueeze(0)

    chamfers = []
    for b in range(recon.size(0)):
        chamfers.append(chamfer_distance(recon[b], target[b]))
    chamfer_term = torch.stack(chamfers).mean()
    kl_term = kl_divergence(mu, logvar)
    total = chamfer_term + beta_kl * kl_term
    return total, chamfer_term, kl_term
