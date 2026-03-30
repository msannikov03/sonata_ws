"""
Point cloud VQ-VAE with Residual Vector Quantization (RVQ).

Instead of a single codebook vector per scene (which gives only num_codes
representable latent states), RVQ uses multiple quantization levels that
progressively refine the residual.  With L levels of C codes each, the
effective codebook size is C^L (e.g. 1024^8 ≈ 10^24).

Architecture:
  Encoder: PointNet-style shared MLP + global max-pool → z_e (D,)
  Quantizer: L-level residual VQ → z_q = sum of L codebook entries (D,)
  Decoder: MLP → K output points (K*3,)

Loss = Chamfer(recon, target) + per-level codebook + commitment losses.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.refinement_net import chamfer_distance


# ---------------------------------------------------------------------------
# Residual Vector Quantization
# ---------------------------------------------------------------------------


class ResidualVQ(nn.Module):
    """Multi-level residual vector quantization.

    Each level quantizes the residual left by previous levels.  The final
    quantized representation is the sum of all per-level codes.  This keeps
    the latent shape as (D,) or (B, D) — fully compatible with downstream
    modules that expect a single latent vector.
    """

    def __init__(
        self,
        num_quantizers: int = 8,
        num_codes: int = 1024,
        dim: int = 256,
        beta_commit: float = 0.25,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_codes = num_codes
        self.dim = dim
        self.beta_commit = beta_commit
        self.codebooks = nn.ModuleList(
            [nn.Embedding(num_codes, dim) for _ in range(num_quantizers)]
        )

    def _nearest_indices(
        self, z: torch.Tensor, codebook_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Squared-L2 nearest-neighbor lookup.  z: (B, D)."""
        dists = (
            z.pow(2).sum(-1, keepdim=True)
            - 2.0 * z @ codebook_weight.t()
            + codebook_weight.pow(2).sum(-1).unsqueeze(0)
        )
        return dists.argmin(dim=-1)  # (B,)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: (D,) or (B, D) continuous encoder output.
        Returns:
            z_q: quantized latent (same shape), with straight-through gradients
                 flowing back to z_e.
            vq_loss: scalar combining codebook + commitment losses across all
                     quantization levels.
        """
        single = z_e.dim() == 1
        if single:
            z_e = z_e.unsqueeze(0)

        residual = z_e
        z_q_sum = torch.zeros_like(z_e)
        total_cb = z_e.new_tensor(0.0)
        total_commit = z_e.new_tensor(0.0)

        for codebook in self.codebooks:
            indices = self._nearest_indices(residual.detach(), codebook.weight)
            z_q_l = codebook(indices)  # (B, D) — has codebook gradients

            total_cb = total_cb + F.mse_loss(z_q_l, residual.detach())
            total_commit = total_commit + F.mse_loss(residual, z_q_l.detach())

            # Straight-through: forward uses codebook value, backward goes to residual.
            quantized_st = residual + (z_q_l - residual).detach()
            z_q_sum = z_q_sum + quantized_st
            residual = residual - quantized_st.detach()

        vq_loss = total_cb + self.beta_commit * total_commit

        if single:
            z_q_sum = z_q_sum.squeeze(0)
        return z_q_sum, vq_loss

    @torch.no_grad()
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Inference-time sequential residual quantization (no gradients)."""
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)

        residual = z
        z_q_sum = torch.zeros_like(z)
        for codebook in self.codebooks:
            indices = self._nearest_indices(residual, codebook.weight)
            z_q_l = codebook(indices)
            z_q_sum = z_q_sum + z_q_l
            residual = residual - z_q_l

        if single:
            z_q_sum = z_q_sum.squeeze(0)
        return z_q_sum

    @torch.no_grad()
    def compute_usage(self, z_e: torch.Tensor) -> dict:
        """Per-level codebook utilisation stats for monitoring collapse."""
        single = z_e.dim() == 1
        if single:
            z_e = z_e.unsqueeze(0)
        residual = z_e
        usage = {}
        for i, codebook in enumerate(self.codebooks):
            indices = self._nearest_indices(residual, codebook.weight)
            unique = indices.unique().numel()
            usage[f"level_{i}_unique_codes"] = unique
            usage[f"level_{i}_utilisation"] = unique / self.num_codes
            z_q_l = codebook(indices)
            residual = residual - z_q_l
        return usage


# ---------------------------------------------------------------------------
# VQ-VAE with Residual VQ
# ---------------------------------------------------------------------------


class PointCloudVQVAE(nn.Module):
    """
    Point cloud VQ-VAE with Residual Vector Quantization.

    Encoder outputs continuous z_e (D).  ResidualVQ maps it to z_q (D)
    via L levels of codebook lookup on successive residuals.
    Decoder maps z_q → (K, 3).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_codes: int = 1024,
        num_quantizers: int = 8,
        num_decoded_points: int = 2048,
        beta_commit: float = 0.25,
        encoder_widths: Optional[List[int]] = None,
        decoder_widths: Optional[List[int]] = None,
    ):
        super().__init__()

        if encoder_widths is None:
            encoder_widths = [64, 128, 256]
        if decoder_widths is None:
            decoder_widths = [512, 512, 512]

        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.num_quantizers = num_quantizers
        self.num_decoded_points = num_decoded_points

        # --- Encoder (PointNet-style) ---
        enc_layers: List[nn.Module] = []
        d_in = 3
        for w in encoder_widths:
            enc_layers.extend([nn.Linear(d_in, w), nn.LayerNorm(w), nn.GELU()])
            d_in = w
        self.point_encoder = nn.Sequential(*enc_layers)
        self.encoder_out = nn.Linear(encoder_widths[-1], latent_dim)

        # --- Residual VQ ---
        self.residual_vq = ResidualVQ(
            num_quantizers=num_quantizers,
            num_codes=num_codes,
            dim=latent_dim,
            beta_commit=beta_commit,
        )

        # --- Decoder (MLP) ---
        dec_layers: List[nn.Module] = []
        d = latent_dim
        for w in decoder_widths:
            dec_layers.extend([nn.Linear(d, w), nn.LayerNorm(w), nn.GELU()])
            d = w
        self.decoder_backbone = nn.Sequential(*dec_layers)
        self.decoder_out = nn.Linear(d, num_decoded_points * 3)

    # ---- encoding --------------------------------------------------------

    def encode_continuous(self, points: torch.Tensor) -> torch.Tensor:
        """(N, 3) → z_e (latent_dim,)."""
        if points.dim() != 2 or points.size(-1) != 3:
            raise ValueError(f"Expected (N, 3) points, got {tuple(points.shape)}")
        if points.size(0) == 0:
            return points.new_zeros(self.latent_dim)
        feat = self.point_encoder(points)
        pooled = feat.max(dim=0).values
        return self.encoder_out(pooled)

    # ---- quantization (inference-time, no grad) --------------------------

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Inference-time residual quantization. (B,D) or (D,) → same shape."""
        return self.residual_vq.quantize(z)

    # ---- decoding --------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(D,) or (B, D) → (K, 3) or (B, K, 3)."""
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        h = self.decoder_backbone(z)
        out = self.decoder_out(h).view(z.size(0), self.num_decoded_points, 3)
        if single:
            out = out.squeeze(0)
        return out

    # ---- training forward ------------------------------------------------

    def forward(
        self, points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3)
        Returns:
            recon: (K, 3)
            z_e: (D,) continuous encoder output
            z_q: (D,) quantized (sum of residual codes), straight-through
            vq_loss: scalar VQ loss (codebook + commitment across all levels)
        """
        z_e = self.encode_continuous(points)
        z_q, vq_loss = self.residual_vq(z_e)
        recon = self.decode(z_q)
        return recon, z_e, z_q, vq_loss

    # ---- batched encoding for latent diffusion ---------------------------

    def encode_batched(
        self,
        points: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Returns (B, latent_dim) quantized latents for diffusion targets."""
        zs = []
        for b in range(batch_size):
            mask = batch == b
            z_e = self.encode_continuous(points[mask])
            z_q = self.residual_vq.quantize(z_e.unsqueeze(0)).squeeze(0)
            zs.append(z_q)
        return torch.stack(zs, dim=0)
