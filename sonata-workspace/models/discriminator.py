"""
Multi-scale PointNet critic for WGAN-GP adversarial training of the VAE.

Why WGAN-GP for LiDAR point cloud completion:
  - Wasserstein distance gives a stable, meaningful gradient signal even when
    the generator distribution is far from real — unlike vanilla GAN which
    saturates early and gives near-zero gradients.
  - Gradient penalty (GP) enforces the 1-Lipschitz constraint without weight
    clipping, preserving model capacity.
  - No mode collapse: the critic never "saturates" since it outputs unbounded
    real values (no sigmoid).

Why multi-scale PointNet architecture:
  LiDAR scenes have structure at two distinct granularities:
    Scale 1 (local, 3→64→128): curvature, surface smoothness, point density
    Scale 2 (global, 128→256→512): scene extent, region coverage, layout
  Max-pooling at each scale extracts the most salient feature across all points,
  then both are concatenated so the critic sees the full picture. A single-scale
  discriminator captures only one of these, leading to partial feedback.

No BatchNorm anywhere: BatchNorm normalises across the batch, which corrupts the
gradient penalty computation (the penalty requires per-sample Lipschitz
enforcement on interpolated inputs). LayerNorm normalises per sample → safe.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiScalePointDiscriminator(nn.Module):
    """
    Two-scale PointNet WGAN critic for point cloud realism scoring.

    Forward pass:
      xyz (B, N, 3)
        → MLP-1: (B, N, 128) → max-pool → g1 (B, 128)   [local geometry]
        → MLP-2: (B, N, 512) → max-pool → g2 (B, 512)   [global structure]
        → cat([g1, g2]) (B, 640) → head → score (B, 1)  [unbounded critic]
    """

    def __init__(self, in_dim: int = 3):
        super().__init__()

        # Scale 1 — local geometry features
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Scale 2 — global scene structure features
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Critic head — no sigmoid, outputs unbounded Wasserstein score
        self.head = nn.Sequential(
            nn.Linear(128 + 512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Args:
            xyz: (B, N, 3) point cloud batch
            return_features: if True, also return the multi-scale global
              features [g1, g2]. Used for feature-matching loss in the
              generator step (Salimans et al. 2016; Larsen et al. 2016).
        Returns:
            score: (B, 1) — unbounded Wasserstein critic score.
            (if return_features) features: list of [g1 (B,128), g2 (B,512)].
        """
        x1 = self.mlp1(xyz)               # (B, N, 128)
        g1 = x1.max(dim=1)[0]             # (B, 128)
        x2 = self.mlp2(x1)                # (B, N, 512)
        g2 = x2.max(dim=1)[0]             # (B, 512)
        g = torch.cat([g1, g2], dim=-1)   # (B, 640)
        score = self.head(g)              # (B, 1)
        if return_features:
            return score, [g1, g2]
        return score


def feature_matching_loss(
    real_feats: list,
    fake_feats: list,
) -> torch.Tensor:
    """
    L1 feature-matching loss between discriminator activations on real vs.
    fake samples (Salimans et al. 2016).

    The generator is encouraged to produce samples whose internal D-features
    match the statistics of real samples, providing a much smoother gradient
    signal than the adversarial loss alone. Standard in VAE-GAN literature.

    Real features are detached so this loss only updates the generator.
    """
    loss = 0.0
    for rf, ff in zip(real_feats, fake_feats):
        loss = loss + torch.nn.functional.l1_loss(ff, rf.detach())
    return loss


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty: penalises the critic's gradient norm on random
    interpolations between real and fake samples.

    The penalty enforces ||∇D(x̂)||₂ ≈ 1 everywhere along the straight line
    between real and fake distributions (Gulrajani et al. 2017).

    Args:
        real: (B, N, 3) real point clouds
        fake: (B, N, 3) generated point clouds (should be detached)
    Returns:
        scalar gradient penalty term
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    d_interp = discriminator(interpolated)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]                          # (B, N, 3)

    grads = grads.reshape(B, -1)  # (B, N*3)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
