"""
Point cloud VQ-VAE: PointNet-style encoder -> vector quantization codebook -> MLP decoder
-> fixed K output points.

Loss = Chamfer reconstruction + codebook loss + commitment loss.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.refinement_net import chamfer_distance


class PointCloudVQVAE(nn.Module):
    """
    Global (single-vector) VQ-VAE for point clouds.

    Encoder outputs a latent vector z_e (D).
    Quantizer maps z_e to nearest codebook entry z_q (D) via straight-through.
    Decoder maps z_q (D) -> (K, 3).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_codes: int = 1024,
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
        self.num_codes = num_codes
        self.num_decoded_points = num_decoded_points

        enc_layers: List[nn.Module] = []
        d_in = 3
        for w in encoder_widths:
            enc_layers.extend([nn.Linear(d_in, w), nn.LayerNorm(w), nn.GELU()])
            d_in = w
        self.point_encoder = nn.Sequential(*enc_layers)
        d_enc = encoder_widths[-1]

        # Encoder output (continuous embedding) for quantization.
        self.encoder_out = nn.Linear(d_enc, latent_dim)

        # Codebook embeddings: (num_codes, latent_dim)
        self.codebook = nn.Embedding(num_codes, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

        dec_layers: List[nn.Module] = []
        d = latent_dim
        for w in decoder_widths:
            dec_layers.extend([nn.Linear(d, w), nn.LayerNorm(w), nn.GELU()])
            d = w
        self.decoder_backbone = nn.Sequential(*dec_layers)
        self.decoder_out = nn.Linear(d, num_decoded_points * 3)

    def encode_continuous(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (N, 3)
        Returns:
            z_e: (latent_dim,)
        """
        if points.dim() != 2 or points.size(-1) != 3:
            raise ValueError(f"Expected (N, 3) points, got {tuple(points.shape)}")
        if points.size(0) == 0:
            return points.new_zeros(self.latent_dim)

        feat = self.point_encoder(points)  # (N, d_enc)
        pooled = feat.max(dim=0).values  # (d_enc,)
        return self.encoder_out(pooled)  # (D,)

    @torch.no_grad()
    def quantize(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Quantize latent vectors to nearest codebook entries (no grad).
        Used at inference time.
        """
        single = z_e.dim() == 1
        if single:
            z_e = z_e.unsqueeze(0)
        code = self.codebook.weight
        dists = (z_e.unsqueeze(1) - code.unsqueeze(0)).pow(2).sum(dim=-1)
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)
        if single:
            z_q = z_q.squeeze(0)
        return z_q

    def quantize_with_grad(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Quantize with gradient flowing to codebook embeddings.
        Finds nearest index without grad, then does lookup with grad.
        """
        single = z_e.dim() == 1
        if single:
            z_e = z_e.unsqueeze(0)
        with torch.no_grad():
            code = self.codebook.weight
            dists = (z_e.unsqueeze(1) - code.unsqueeze(0)).pow(2).sum(dim=-1)
            indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)  # grad flows to codebook
        if single:
            z_q = z_q.squeeze(0)
        return z_q

    def forward(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3)
        Returns:
            recon: (K, 3)
            z_e: (D,) continuous embedding
            z_q_st: (D,) quantized embedding with straight-through gradients
            z_q: (D,) quantized embedding with codebook gradients
        """
        z_e = self.encode_continuous(points)  # (D,)
        z_q = self.quantize_with_grad(z_e)  # (D,) grad to codebook

        # Straight-through estimator: forward uses z_q, backward uses z_e gradients.
        z_q_st = z_e + (z_q - z_e).detach()

        recon = self.decode(z_q_st)
        return recon, z_e, z_q_st, z_q

    def encode_batched(
        self,
        points: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Encode a flat concatenated batch, returning quantized latents z_q.

        Args:
            points: (N_tot, 3)
            batch: (N_tot,) integer batch indices in [0, B)
        Returns:
            z_q: (B, latent_dim)
        """
        zs = []
        for b in range(batch_size):
            mask = batch == b
            pb = points[mask]
            z_e = self.encode_continuous(pb)
            z_q = self.quantize(z_e)
            zs.append(z_q)
        return torch.stack(zs, dim=0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (D,) or (B, D)
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


def vq_vae_reconstruction_chamfer(
    recon: torch.Tensor,
    target: torch.Tensor,
    z_e: torch.Tensor,
    z_q_st: torch.Tensor,
    beta_commit: float = 0.25,
    beta_codebook: float = 1.0,
    z_q: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruction + VQ losses.

    Args:
        recon: (K, 3) reconstructed points
        target: (M, 3) target points
        z_e: (D,) encoder output embedding
        z_q_st: (D,) quantized latent (straight-through tensor)
        z_q: (D,) quantized embedding with codebook gradients
    Returns:
        total, recon_term, codebook_term, commit_term
    """
    recon_term = chamfer_distance(recon, target)

    # Use z_q with grad if provided, otherwise fall back (broken but backward compat)
    if z_q is None:
        z_q = z_q_st.detach()

    # Codebook loss: move codebook toward encoder output (grad to codebook only)
    codebook_term = F.mse_loss(z_q, z_e.detach())

    # Commitment loss: encourage encoder to stay close to codebook (grad to encoder only)
    commit_term = F.mse_loss(z_e, z_q.detach())

    total = recon_term + beta_codebook * codebook_term + beta_commit * commit_term
    return total, recon_term, codebook_term, commit_term

