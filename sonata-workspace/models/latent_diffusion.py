"""
Latent diffusion for point-cloud scene completion.

Key design decisions (fixes over the initial MLP-based implementation):

1. **Latent normalisation** – VAE latents are scaled to ~unit variance before
   the forward diffusion process.  The cosine / linear noise schedule assumes
   unit-scale data; without this, the signal-to-noise ratio is wrong and
   inference from pure noise fails (Rombach et al. 2022, §3.3).

2. **Perceiver-style condition pooler** – replaces mean-pooling with learned
   cross-attention queries, compressing variable-length per-point Sonata
   features into a fixed set of condition tokens that retain spatial info.

3. **Token-based transformer denoiser** – the latent vector is split into
   tokens, processed through self-attention + cross-attention to condition
   tokens + MLP blocks with AdaLN time conditioning, then recombined.
   Much more expressive than the old 3-layer MLP (DiT-style architecture).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion_module import DiffusionScheduler, SinusoidalTimeEmbedding
from models.sonata_encoder import ConditionalFeatureExtractor


# ---------------------------------------------------------------------------
# Latent normaliser
# ---------------------------------------------------------------------------


class LatentNormalizer(nn.Module):
    """Running-EMA normalisation that brings VAE latents to ~unit variance.

    The first batch initialises the statistics directly; subsequent batches
    refine them via exponential moving average.  During inference the stored
    running_mean / running_var are used to denormalize the sampled latent.
    """

    def __init__(self, dim: int, momentum: float = 0.01):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("num_batches", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """EMA update from a batch of latents z (B, D)."""
        if z.size(0) < 2:
            return
        batch_mean = z.mean(dim=0)
        batch_var = z.var(dim=0, unbiased=False).clamp(min=1e-8)
        if self.num_batches == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
        else:
            m = self.momentum
            self.running_mean.lerp_(batch_mean, m)
            self.running_var.lerp_(batch_var, m)
        self.num_batches += 1

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.running_mean) / (self.running_var.sqrt() + 1e-8)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * (self.running_var.sqrt() + 1e-8) + self.running_mean


# ---------------------------------------------------------------------------
# Condition pooler (Perceiver-style)
# ---------------------------------------------------------------------------


class ConditionPooler(nn.Module):
    """Compress variable-length per-point Sonata features into a fixed set of
    condition tokens via learned cross-attention queries (Perceiver-style)."""

    def __init__(
        self,
        feat_dim: int,
        num_tokens: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.query = nn.Parameter(torch.randn(1, num_tokens, feat_dim) * 0.02)
        self.norm_q = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(feat_dim)
        self.ff = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim),
        )

    def forward(
        self,
        feat: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """(N_total, C) + batch indices → (B, num_tokens, C)."""
        parts: List[torch.Tensor] = []
        for b in range(batch_size):
            mask = batch_idx == b
            kv = feat[mask].unsqueeze(0)  # (1, Nb, C)
            q = self.norm_q(self.query)  # (1, T, C)
            h, _ = self.attn(q, kv, kv)  # (1, T, C)
            h = h + self.query  # residual around attention
            h = h + self.ff(self.norm_ff(h))  # FF residual
            parts.append(h.squeeze(0))  # (T, C)
        return torch.stack(parts)  # (B, T, C)


# ---------------------------------------------------------------------------
# Denoiser building blocks (DiT-style)
# ---------------------------------------------------------------------------


class AdaLN(nn.Module):
    """Adaptive Layer Normalisation conditioned on a time embedding vector.
    Initialised so that scale=1, shift=0 (i.e. identity)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D), c: (B, cond_dim). Returns (B, L, D)."""
        s, b = self.proj(c).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + s) + b


class DenoiserBlock(nn.Module):
    """Self-attn → cross-attn (to condition tokens) → MLP, each gated by AdaLN."""

    def __init__(
        self,
        dim: int,
        cond_kv_dim: int,
        time_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.adaln_sa = AdaLN(dim, time_dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.adaln_ca = AdaLN(dim, time_dim)
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True,
            kdim=cond_kv_dim,
            vdim=cond_kv_dim,
        )

        self.adaln_ff = AdaLN(dim, time_dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B,L,D), cond: (B,K,C_cond), t_emb: (B,t_dim)."""
        h = self.adaln_sa(x, t_emb)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        h = self.adaln_ca(x, t_emb)
        h, _ = self.cross_attn(h, cond, cond)
        x = x + h

        x = x + self.ff(self.adaln_ff(x, t_emb))
        return x


# ---------------------------------------------------------------------------
# Token-based latent denoiser
# ---------------------------------------------------------------------------


class LatentDenoiser(nn.Module):
    """Splits z into tokens, processes with transformer blocks that
    cross-attend to condition tokens, then recombines to predict epsilon."""

    def __init__(
        self,
        latent_dim: int = 256,
        cond_dim: int = 256,
        hidden_dim: int = 1024,
        time_embed_dim: int = 256,
        num_blocks: int = 8,
        num_heads: int = 4,
        num_latent_tokens: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_latent_tokens = num_latent_tokens
        assert hidden_dim % num_latent_tokens == 0
        self.token_dim = hidden_dim // num_latent_tokens

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.GELU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_latent_tokens, self.token_dim) * 0.02,
        )

        self.blocks = nn.ModuleList(
            [
                DenoiserBlock(
                    self.token_dim,
                    cond_dim,
                    time_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(self.token_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t:  (B, latent_dim) noisy latent
        t:    (B,) integer timesteps
        cond: (B, K, cond_dim) condition tokens from ConditionPooler
        Returns: (B, latent_dim) predicted noise epsilon.
        """
        B = z_t.size(0)

        t_emb = self.time_proj(self.time_embed(t.float()))  # (B, time_embed_dim)

        h = self.input_proj(z_t)  # (B, hidden_dim)
        tokens = h.view(B, self.num_latent_tokens, self.token_dim) + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens, cond, t_emb)

        tokens = self.final_norm(tokens)
        h = tokens.reshape(B, self.hidden_dim)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Scene completion module
# ---------------------------------------------------------------------------


class SceneCompletionLatentDiffusion(nn.Module):
    """
    Full latent diffusion pipeline for scene completion.

    Fixes:
    - LatentNormalizer: z scaled to unit variance before diffusion.
    - ConditionPooler: Perceiver cross-attention preserves spatial info.
    - LatentDenoiser: token transformer (not a 3-layer MLP).
    """

    def __init__(
        self,
        vae: nn.Module,
        condition_extractor: ConditionalFeatureExtractor,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        denoising_steps: int = 50,
        hidden_dim: int = 1024,
        num_denoiser_blocks: int = 8,
        num_latent_tokens: int = 8,
        num_cond_tokens: int = 32,
        num_heads: int = 4,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.vae = vae
        self.condition_extractor = condition_extractor
        self.latent_dim = vae.latent_dim
        self.cond_dim = condition_extractor.out_dim
        self.denoising_steps = denoising_steps

        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps, schedule=schedule,
        )

        self.normalizer = LatentNormalizer(self.latent_dim)

        self.cond_pooler = ConditionPooler(
            self.cond_dim,
            num_tokens=num_cond_tokens,
        )

        self.denoiser = LatentDenoiser(
            latent_dim=self.latent_dim,
            cond_dim=self.cond_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_blocks=num_denoiser_blocks,
            num_heads=num_heads,
            num_latent_tokens=num_latent_tokens,
        )

    # --- helpers ----------------------------------------------------------

    def _move_scheduler(self, device: torch.device) -> None:
        for attr in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas",
            "sqrt_recipm1_alphas",
            "posterior_variance",
            "posterior_log_variance_clipped",
        ):
            t = getattr(self.scheduler, attr)
            if t.device != device:
                setattr(self.scheduler, attr, t.to(device))

    def encode_condition(
        self,
        partial_scan: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Returns (B, num_cond_tokens, cond_dim)."""
        fused, _ = self.condition_extractor(partial_scan)
        batch = partial_scan["batch"]
        return self.cond_pooler(fused, batch, batch_size)

    # --- training ---------------------------------------------------------

    def forward_training(
        self,
        partial_scan: Dict[str, torch.Tensor],
        complete_coord: torch.Tensor,
        complete_batch: torch.Tensor,
        freeze_vae: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Epsilon-prediction loss in normalised latent space."""
        device = complete_coord.device
        self._move_scheduler(device)
        bsz = int(complete_batch.max().item()) + 1

        cond = self.encode_condition(partial_scan, bsz)

        ctx = torch.no_grad() if freeze_vae else torch.enable_grad()
        with ctx:
            enc_out = self.vae.encode_batched(
                complete_coord, complete_batch, bsz,
            )

        if isinstance(enc_out, tuple) and len(enc_out) == 2:
            mu, _logvar = enc_out
            z0 = mu
        else:
            z0 = enc_out

        if self.training:
            self.normalizer.update(z0)
        z0 = self.normalizer.normalize(z0)

        t = torch.randint(
            0, self.scheduler.num_timesteps, (bsz,),
            device=device, dtype=torch.long,
        )
        noise = torch.randn_like(z0)
        sa = self.scheduler.sqrt_alphas_cumprod[t].unsqueeze(-1)
        som = self.scheduler.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        z_t = sa * z0 + som * noise

        pred = self.denoiser(z_t, t, cond)
        loss = F.mse_loss(pred, noise)
        return {"loss": loss}

    # --- inference --------------------------------------------------------

    @torch.no_grad()
    def complete_scene(
        self,
        partial_scan: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        sampling: str = "ddim",
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Denoise from pure noise → denormalize → decode with VAE → point cloud."""
        if num_steps is None:
            num_steps = self.denoising_steps

        device = next(self.denoiser.parameters()).device
        self._move_scheduler(device)

        batch = partial_scan["batch"]
        bsz = int(batch.max().item()) + 1
        cond = self.encode_condition(partial_scan, bsz)
        B = cond.size(0)

        z = torch.randn(B, self.latent_dim, device=device)

        # Build monotone-decreasing timestep list ending at 0.
        ts = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_steps,
            dtype=torch.long, device=device,
        )
        ts_list = sorted(set(int(v) for v in ts.tolist()), reverse=True)
        if 0 not in ts_list:
            ts_list.append(0)

        # DDIM sampling (deterministic when eta=0).
        for i in range(len(ts_list) - 1):
            t_cur = ts_list[i]
            t_prev = ts_list[i + 1]

            t_vec = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps = self.denoiser(z, t_vec, cond)

            ab = self.scheduler.alphas_cumprod[t_cur]
            ab_prev = self.scheduler.alphas_cumprod[t_prev]

            x0_pred = (z - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)

            if eta == 0.0:
                z = (
                    torch.sqrt(ab_prev) * x0_pred
                    + torch.sqrt(1.0 - ab_prev) * eps
                )
            else:
                sigma = (
                    eta
                    * torch.sqrt((1.0 - ab_prev) / (1.0 - ab))
                    * torch.sqrt(1.0 - ab / ab_prev)
                )
                z = (
                    torch.sqrt(ab_prev) * x0_pred
                    + torch.sqrt((1.0 - ab_prev - sigma ** 2).clamp(min=0)) * eps
                    + sigma * torch.randn_like(z)
                )

        # Denormalize back to VAE latent scale.
        z = self.normalizer.denormalize(z)

        # VQ-VAE: snap to nearest codebook entry before decoding.
        if hasattr(self.vae, "quantize"):
            z = self.vae.quantize(z)

        pts = self.vae.decode(z)
        if pts.dim() == 3 and pts.size(0) == 1:
            pts = pts.squeeze(0)
        return pts
