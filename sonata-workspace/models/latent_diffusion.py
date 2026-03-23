"""
Latent diffusion on PointCloudVAE latent z, conditioned on pooled Sonata features.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_module import DiffusionScheduler, SinusoidalTimeEmbedding
from models.point_cloud_vae import PointCloudVAE
from models.sonata_encoder import ConditionalFeatureExtractor


class LatentMLPDenoiser(nn.Module):
    """Predicts epsilon in latent space."""

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        time_embed_dim: int = 128,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        in_dim = latent_dim + time_embed_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        z_t: (B, latent_dim), t: (B,) long, cond: (B, cond_dim)
        """
        t_emb = self.time_embedding(t.float())
        x = torch.cat([z_t, t_emb, cond], dim=-1)
        return self.net(x)


def pool_per_point_to_batch(
    feat: torch.Tensor, batch_idx: torch.Tensor, batch_size: int
) -> torch.Tensor:
    """Mean-pool (N, C) -> (B, C)."""
    out = []
    for b in range(batch_size):
        m = batch_idx == b
        if m.any():
            out.append(feat[m].mean(dim=0))
        else:
            out.append(feat.new_zeros(feat.size(-1)))
    return torch.stack(out, dim=0)


class SceneCompletionLatentDiffusion(nn.Module):
    """
    Diffusion on VAE latent z_0 = encode(complete cloud).
    Condition: mean-pooled Sonata conditional features from partial scan.
    """

    def __init__(
        self,
        vae: PointCloudVAE,
        condition_extractor: ConditionalFeatureExtractor,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        denoising_steps: int = 50,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.vae = vae
        self.condition_extractor = condition_extractor
        self.latent_dim = vae.latent_dim
        self.cond_dim = condition_extractor.out_dim
        self.denoising_steps = denoising_steps

        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps, schedule=schedule
        )
        self.denoiser = LatentMLPDenoiser(
            self.latent_dim,
            self.cond_dim,
            time_embed_dim=128,
            hidden_dim=hidden_dim,
        )

    def _move_scheduler(self, device: torch.device) -> None:
        self.scheduler.betas = self.scheduler.betas.to(device)
        self.scheduler.alphas = self.scheduler.alphas.to(device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        self.scheduler.alphas_cumprod_prev = self.scheduler.alphas_cumprod_prev.to(
            device
        )
        self.scheduler.sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod.to(
            device
        )
        self.scheduler.sqrt_one_minus_alphas_cumprod = (
            self.scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        )
        self.scheduler.sqrt_recip_alphas = self.scheduler.sqrt_recip_alphas.to(device)
        self.scheduler.sqrt_recipm1_alphas = self.scheduler.sqrt_recipm1_alphas.to(
            device
        )
        self.scheduler.posterior_variance = self.scheduler.posterior_variance.to(
            device
        )
        self.scheduler.posterior_log_variance_clipped = (
            self.scheduler.posterior_log_variance_clipped.to(device)
        )

    def encode_condition(self, partial_scan: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused, _ = self.condition_extractor(partial_scan)
        batch = partial_scan["batch"]
        bsz = int(batch.max().item()) + 1
        return pool_per_point_to_batch(fused, batch, bsz)

    def forward_training(
        self,
        partial_scan: Dict[str, torch.Tensor],
        complete_coord: torch.Tensor,
        complete_batch: torch.Tensor,
        freeze_vae: bool = True,
        use_posterior_sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Train denoiser with epsilon prediction loss in latent space.
        z_0 is encoder mean mu (or reparameterized sample if use_posterior_sample).
        """
        device = complete_coord.device
        self._move_scheduler(device)

        bsz = int(complete_batch.max().item()) + 1
        cond = self.encode_condition(partial_scan)

        ctx = torch.no_grad() if freeze_vae else torch.enable_grad()
        with ctx:
            enc_out = self.vae.encode_batched(
                complete_coord, complete_batch, bsz
            )

        # Gaussian VAE returns (mu, logvar). VQ-VAE returns z_q directly.
        if isinstance(enc_out, tuple) and len(enc_out) == 2:
            mu, logvar = enc_out
            if use_posterior_sample and not freeze_vae and hasattr(
                self.vae, "reparameterize"
            ):
                z0 = self.vae.reparameterize(mu, logvar)
            else:
                z0 = mu
        else:
            # VQ-VAE (or other latent models) return z0 directly.
            z0 = enc_out

        t = torch.randint(
            0, self.scheduler.num_timesteps, (bsz,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(z0)
        sa = self.scheduler.sqrt_alphas_cumprod.to(device)[t].unsqueeze(-1)
        som = self.scheduler.sqrt_one_minus_alphas_cumprod.to(device)[t].unsqueeze(-1)
        z_t = sa * z0 + som * noise

        pred_noise = self.denoiser(z_t, t, cond)
        loss = F.mse_loss(pred_noise, noise)
        return {"loss": loss, "pred_noise": pred_noise}

    @torch.no_grad()
    def p_sample_step(
        self,
        z: torch.Tensor,
        t_int: int,
        cond: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Single reverse DDPM step; z (B, D), cond (B, cond_dim)."""
        device = z.device
        self._move_scheduler(device)
        B = z.size(0)
        t = torch.full((B,), t_int, device=device, dtype=torch.long)

        pred_noise = self.denoiser(z, t, cond)

        alpha_bar = self.scheduler.alphas_cumprod[t_int]
        alpha_bar_prev = self.scheduler.alphas_cumprod_prev[t_int]
        beta_t = self.scheduler.betas[t_int]
        alpha_t = self.scheduler.alphas[t_int]

        sqrt_ac = self.scheduler.sqrt_alphas_cumprod[t_int]
        sqrt_om = self.scheduler.sqrt_one_minus_alphas_cumprod[t_int]

        pred_z0 = (z - sqrt_om * pred_noise) / sqrt_ac
        if clip_denoised:
            pred_z0 = pred_z0.clamp(-10.0, 10.0)

        if t_int == 0:
            return pred_z0

        posterior_mean = (
            torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar) * pred_z0
            + torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * z
        )
        posterior_var = self.scheduler.posterior_variance[t_int]
        noise = torch.randn_like(z)
        return posterior_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def complete_scene(
        self,
        partial_scan: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        sampling: str = "ddim",
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Returns decoded point cloud (K, 3) for batch_size 1, else (B, K, 3).
        """
        if num_steps is None:
            num_steps = self.denoising_steps

        device = next(self.denoiser.parameters()).device
        self._move_scheduler(device)

        cond = self.encode_condition(partial_scan)
        B = cond.size(0)
        z = torch.randn(B, self.latent_dim, device=device)

        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=device,
        )
        # If integer linspace skips values, `p_sample_step()` (t -> t-1) is no longer correct.
        # Use DDIM update when steps are not adjacent.
        ts_list = timesteps.tolist()
        ts_unique = sorted(set(ts_list), reverse=True)
        if 0 not in ts_unique:
            ts_unique.append(0)

        sampling = sampling.lower()
        use_ddpm = sampling == "ddpm" and all(
            ts_unique[i] - 1 == ts_unique[i + 1]
            for i in range(len(ts_unique) - 1)
        )

        if use_ddpm:
            for t_int in ts_unique:
                z = self.p_sample_step(z, int(t_int), cond)
        else:
            # DDIM-style latent sampling (epsilon-prediction parameterization).
            # eta=0 gives deterministic sampling; higher eta adds stochasticity.
            for i in range(len(ts_unique) - 1):
                t = ts_unique[i]
                t_prev = ts_unique[i + 1]

                t_tensor = torch.full(
                    (B,), t, device=device, dtype=torch.long
                )
                eps = self.denoiser(z, t_tensor, cond)

                a_bar_t = self.scheduler.alphas_cumprod[t]
                a_bar_prev = self.scheduler.alphas_cumprod[t_prev]

                # x0 prediction from epsilon
                sqrt_a_bar_t = torch.sqrt(a_bar_t)
                sqrt_one_minus_a_bar_t = torch.sqrt(1.0 - a_bar_t)
                x0_pred = (z - sqrt_one_minus_a_bar_t * eps) / sqrt_a_bar_t

                # DDIM update
                sqrt_a_bar_prev = torch.sqrt(a_bar_prev)
                if eta == 0.0:
                    z = (
                        sqrt_a_bar_prev * x0_pred
                        + torch.sqrt(1.0 - a_bar_prev) * eps
                    )
                else:
                    sigma_t = (
                        eta
                        * torch.sqrt((1.0 - a_bar_prev) / (1.0 - a_bar_t))
                        * torch.sqrt(1.0 - a_bar_t / a_bar_prev)
                    )
                    noise = torch.randn_like(z)
                    z = (
                        sqrt_a_bar_prev * x0_pred
                        + torch.sqrt(torch.clamp(1.0 - a_bar_prev - sigma_t**2, min=0.0)) * eps
                        + sigma_t * noise
                    )

        # If this is a VQ-VAE, map the final continuous latent back to the codebook
        # before decoding (helps avoid drifting off-manifold).
        if hasattr(self.vae, "quantize"):
            z = self.vae.quantize(z)
        pts = self.vae.decode(z)
        if pts.dim() == 3 and pts.size(0) == 1:
            pts = pts.squeeze(0)
        return pts
