"""
Point cloud VAE: PointNet-style encoder -> Gaussian latent -> MLP decoder -> fixed K points.
Trained with Chamfer reconstruction + KL.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.refinement_net import chamfer_distance


class PointCloudVAE(nn.Module):
    """
    Encoder: shared MLP on (N, 3) -> per-point features, global max pool -> mu, logvar.
    Decoder: z -> MLP -> (K, 3) point set.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_decoded_points: int = 2048,
        encoder_widths: Optional[List[int]] = None,
        decoder_widths: Optional[List[int]] = None,
    ):
        super().__init__()
        if encoder_widths is None:
            encoder_widths = [64, 128, 256]
        if decoder_widths is None:
            decoder_widths = [512, 512, 512]

        self.latent_dim = latent_dim
        self.num_decoded_points = num_decoded_points

        enc_layers: List[nn.Module] = []
        d_in = 3
        for w in encoder_widths:
            enc_layers.extend([nn.Linear(d_in, w), nn.LayerNorm(w), nn.GELU()])
            d_in = w
        self.point_encoder = nn.Sequential(*enc_layers)
        d_enc = encoder_widths[-1]
        self.fc_mu = nn.Linear(d_enc, latent_dim)
        self.fc_logvar = nn.Linear(d_enc, latent_dim)

        dec_layers: List[nn.Module] = []
        d = latent_dim
        for w in decoder_widths:
            dec_layers.extend([nn.Linear(d, w), nn.LayerNorm(w), nn.GELU()])
            d = w
        self.decoder_backbone = nn.Sequential(*dec_layers)
        self.decoder_out = nn.Linear(d, num_decoded_points * 3)

    def encode(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3)
        Returns:
            mu, logvar each (latent_dim,)
        """
        if points.dim() != 2 or points.size(-1) != 3:
            raise ValueError(f"Expected (N, 3) points, got {tuple(points.shape)}")
        if points.size(0) == 0:
            z = points.new_zeros(self.latent_dim)
            return z, z.new_full((self.latent_dim,), -30.0)

        feat = self.point_encoder(points)
        pooled = feat.max(dim=0).values
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
            pb = points[mask]
            mu, lv = self.encode(pb)
            mus.append(mu)
            logvars.append(lv)
        return torch.stack(mus, dim=0), torch.stack(logvars, dim=0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (latent_dim,) or (B, latent_dim)
        Returns:
            (K, 3) or (B, K, 3)
        """
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        h = self.decoder_backbone(z)
        out = self.decoder_out(h).view(z.size(0), self.num_decoded_points, 3)
        if single:
            out = out.squeeze(0)
        return out

    def forward(
        self, points: torch.Tensor, return_kl_inputs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3) single cloud
        Returns:
            recon: (K, 3), mu, logvar (latent_dim,)
        """
        mu, logvar = self.encode(points)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        if return_kl_inputs:
            return recon, mu, logvar
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


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)), averaged over batch and latent dims."""
    # mu, logvar: (B, D)
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
        recon: (B, K, 3) or (K, 3) when B=1
        target: (B, M, 3) or (M, 3)
        mu, logvar: (B, latent_dim)
    Returns:
        total, chamfer_term, kl_term
    """
    if recon.dim() == 2:
        recon = recon.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    chamfers = []
    for b in range(recon.size(0)):
        chamfers.append(chamfer_distance(recon[b], target[b]))
    chamfer_term = torch.stack(chamfers).mean()
    kl_term = kl_divergence(mu, logvar)
    total = chamfer_term + beta_kl * kl_term
    return total, chamfer_term, kl_term
