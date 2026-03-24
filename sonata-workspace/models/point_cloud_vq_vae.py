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

        self.encoder_out = nn.Linear(d_enc, latent_dim)

        # Codebook embeddings: (num_codes, latent_dim).
        # Default nn.Embedding init is N(0,1) which gives a reasonable spread;
        # the old uniform(±1/num_codes) was far too small and caused collapse.
        self.codebook = nn.Embedding(num_codes, latent_dim)

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

    def quantize(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Quantize latent vectors to nearest codebook entries.

        NOTE: no @torch.no_grad() — the codebook lookup (nn.Embedding forward)
        must retain gradients so the codebook loss can update the entries.
        The argmin index selection is naturally non-differentiable, which is fine.

        Args:
            z_e: (B, D) or (D,)
        Returns:
            z_q: same shape, each row is a codebook embedding.
        """
        single = z_e.dim() == 1
        if single:
            z_e = z_e.unsqueeze(0)

        code = self.codebook.weight  # (C, D)
        # Squared L2 distances — expanded form avoids materialising (B,C,D).
        dists = (
            z_e.pow(2).sum(-1, keepdim=True)
            - 2 * z_e @ code.t()
            + code.pow(2).sum(-1, keepdim=True).t()
        )  # (B, C)
        indices = dists.argmin(dim=-1)  # (B,) — not differentiable, that's OK
        z_q = self.codebook(indices)  # (B, D) — differentiable w.r.t. codebook.weight

        if single:
            z_q = z_q.squeeze(0)
        return z_q

    def forward(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (N, 3)
        Returns:
            recon: (K, 3)
            z_e: (D,) continuous embedding (has encoder gradients)
            z_q: (D,) quantized embedding (has codebook gradients)
        """
        z_e = self.encode_continuous(points)  # (D,)
        z_q = self.quantize(z_e)  # (D,) — has codebook gradients

        # Straight-through estimator: forward pass uses z_q values,
        # backward pass routes gradients to z_e (the encoder).
        z_q_st = z_e + (z_q - z_e).detach()

        recon = self.decode(z_q_st)
        return recon, z_e, z_q

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
    z_q: torch.Tensor,
    beta_commit: float = 0.25,
    beta_codebook: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruction + VQ losses.

    Args:
        recon: (K, 3) reconstructed points
        target: (M, 3) target points
        z_e: (D,) encoder output embedding (has encoder gradients)
        z_q: (D,) quantized embedding (has codebook gradients via nn.Embedding)
    Returns:
        total, recon_term, codebook_term, commit_term
    """
    recon_term = chamfer_distance(recon, target)

    # Codebook loss: move codebook entries toward encoder outputs.
    # z_q carries gradients to self.codebook.weight; stop-grad on z_e.
    codebook_term = F.mse_loss(z_q, z_e.detach())

    # Commitment loss: encourage encoder output to stay close to codebook.
    # Stop-grad on z_q so gradient flows only to encoder.
    commit_term = F.mse_loss(z_e, z_q.detach())

    total = recon_term + beta_codebook * codebook_term + beta_commit * commit_term
    return total, recon_term, codebook_term, commit_term
