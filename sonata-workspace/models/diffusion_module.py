"""
Diffusion Module for Semantic Scene Completion

Implements the diffusion process for point cloud completion,
following LiDiff's point-wise local approach with Sonata encoder conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


class DiffusionScheduler:
    """
    Diffusion noise scheduler supporting multiple schedules.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = "cosine",  # "linear", "cosine", "sigmoid"
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Initialize diffusion scheduler.
        
        Args:
            num_timesteps: Number of diffusion steps
            schedule: Type of noise schedule
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        
        # Generate beta schedule
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            self.betas = self._cosine_schedule(num_timesteps)
        elif schedule == "sigmoid":
            self.betas = self._sigmoid_schedule(
                num_timesteps, beta_start, beta_end
            )
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Compute alpha schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        
        # Precompute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas = torch.sqrt(1.0 / self.alphas - 1)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / 
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
    
    def _cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * np.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _sigmoid_schedule(
        self, 
        timesteps: int, 
        start: float = -3, 
        end: float = 3
    ) -> torch.Tensor:
        """Sigmoid schedule."""
        betas = torch.linspace(start, end, timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
        return betas
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        
        Args:
            x_start: Clean data (x_0)
            t: Timestep
            noise: Optional pre-generated noise
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = \
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return (
            sqrt_alphas_cumprod_t * x_start + 
            sqrt_one_minus_alphas_cumprod_t * noise
        )
    
    def p_sample_step(
        self,
        model: nn.Module,
        x_t_features: torch.Tensor,
        x_t_coords: torch.Tensor,
        t: int,
        condition: Dict[str, torch.Tensor],
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Single step of reverse diffusion: p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model
            x_t_features: (N, 3) Noisy features at timestep t
            x_t_coords: (N, 3) Point coordinates
            t: Current timestep
            condition: Conditional information
            clip_denoised: Clip denoised output
            
        Returns:
            (N, 3) Denoised features at timestep t-1
        """
        batch_size = 1  # Assume single batch
        t_tensor = torch.full((batch_size,), t, device=x_t_features.device)
        
        # Predict noise
        pred_noise = model(x_t_features, x_t_coords, t_tensor, condition)
        
        # Compute x_0 prediction
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]
        
        # x_0 = (x_t - sqrt(1-alpha_t) * pred_noise) / sqrt(alpha_t)
        pred_x0 = (
            x_t_features - self.sqrt_one_minus_alphas_cumprod[t] * pred_noise
        ) / self.sqrt_alphas_cumprod[t]
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Compute x_{t-1}
        if t > 0:
            noise = torch.randn_like(x_t_features)
            posterior_variance_t = self.posterior_variance[t]
            
            # x_{t-1} = posterior_mean + sqrt(posterior_variance) * noise
            posterior_mean = (
                self.sqrt_alphas_cumprod[t - 1] * pred_x0 +
                torch.sqrt(1 - alpha_t_prev - posterior_variance_t) * pred_noise
            )
            
            x_t_prev = posterior_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x_t_prev = pred_x0
        
        return x_t_prev


class SonataTransformerBlock(nn.Module):
    """
    Sonata-style transformer block for point cloud processing.
    
    Inspired by Point Transformer V3's grouped vector attention mechanism.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_groups: int = 4,
        num_neighbors: int = 16,
    ):
        """
        Initialize transformer block.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            num_groups: Number of groups for grouped vector attention
            num_neighbors: Number of neighbors for local attention
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_neighbors = num_neighbors
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert dim % num_groups == 0, "dim must be divisible by num_groups"
        
        # Grouped vector attention components
        self.group_dim = dim // num_groups
        
        # Query, Key, Value projections for each group
        self.q_proj = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_dim, bias=False)
            for _ in range(num_groups)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_dim, bias=False)
            for _ in range(num_groups)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_dim, bias=False)
            for _ in range(num_groups)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        neighbors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with grouped vector attention.
        
        Args:
            features: (N, dim) point features
            coords: (N, 3) point coordinates
            neighbors: (N, num_neighbors) neighbor indices (optional)
            
        Returns:
            (N, dim) transformed features
        """
        residual = features
        features = self.norm1(features)
        
        # Split features into groups
        group_features = torch.chunk(features, self.num_groups, dim=-1)
        
        # Process each group independently
        group_outputs = []
        for i, group_feat in enumerate(group_features):
            # Compute queries, keys, values
            q = self.q_proj[i](group_feat)  # (N, group_dim)
            k = self.k_proj[i](group_feat)  # (N, group_dim)
            v = self.v_proj[i](group_feat)  # (N, group_dim)
            
            # Find neighbors if not provided
            if neighbors is None:
                neighbors = self._find_neighbors(coords)
            
            # Local attention within neighbors
            attn_output = self._local_attention(q, k, v, coords, neighbors)
            group_outputs.append(attn_output)
        
        # Concatenate group outputs
        out = torch.cat(group_outputs, dim=-1)
        out = self.out_proj(out)
        out = out + residual
        
        # FFN
        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out = out + residual
        
        return out
    
    def _find_neighbors(self, coords: torch.Tensor) -> torch.Tensor:
        """Find k-nearest neighbors for each point."""
        coords_np = coords.detach().cpu().numpy()
        tree = cKDTree(coords_np)
        _, indices = tree.query(coords_np, k=min(self.num_neighbors + 1, len(coords)))
        # Remove self (first neighbor)
        if indices.ndim == 2:
            indices = indices[:, 1:]
        return torch.from_numpy(indices).to(coords.device)
    
    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        neighbors: torch.Tensor
    ) -> torch.Tensor:
        """Compute local attention within neighbors."""
        N = q.shape[0]
        device = q.device
        
        # Gather neighbor features
        neighbor_k = k[neighbors]  # (N, num_neighbors, group_dim)
        neighbor_v = v[neighbors]  # (N, num_neighbors, group_dim)
        
        # Compute attention scores
        q_expanded = q.unsqueeze(1)  # (N, 1, group_dim)
        scores = torch.sum(q_expanded * neighbor_k, dim=-1) / np.sqrt(self.group_dim)
        attn_weights = F.softmax(scores, dim=-1)  # (N, num_neighbors)
        
        # Apply attention
        out = torch.sum(attn_weights.unsqueeze(-1) * neighbor_v, dim=1)  # (N, group_dim)
        
        return out


class PointwiseDiffusionBlock(nn.Module):
    """
    Point-wise diffusion block using Sonata-style transformer.
    
    Processes each point's local neighborhood with transformer attention,
    replacing sparse convolutions with Sonata's grouped vector attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int = 128,
        condition_dim: int = 256,
        num_neighbors: int = 16,
        num_heads: int = 8,
        num_groups: int = 4,
    ):
        """
        Initialize point-wise diffusion block.
        
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            condition_dim: Conditional feature dimension
            num_neighbors: Number of neighbors for local processing
            num_heads: Number of attention heads
            num_groups: Number of groups for grouped vector attention
        """
        super().__init__()
        
        self.num_neighbors = num_neighbors
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, in_channels)
        )
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, in_channels)
        
        # Sonata-style transformer block
        self.transformer = SonataTransformerBlock(
            dim=in_channels,
            num_heads=num_heads,
            num_groups=num_groups,
            num_neighbors=num_neighbors
        )
        
        # Output projection
        if in_channels != out_channels:
            self.out_proj = nn.Linear(in_channels, out_channels)
        else:
            self.out_proj = nn.Identity()
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        time_embed: torch.Tensor,
        condition: torch.Tensor,
        neighbors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: (N, in_channels) point features
            coords: (N, 3) point coordinates
            time_embed: (batch_size, time_embed_dim) time step embedding
            condition: (N, condition_dim) conditional features from encoder
            neighbors: (N, num_neighbors) neighbor indices (optional)
            
        Returns:
            (N, out_channels) processed features
        """
        # Apply time embedding (broadcast to all points)
        if time_embed.dim() == 1:
            time_embed = time_embed.unsqueeze(0)
        time_feat = self.time_mlp(time_embed)  # (batch_size, in_channels)
        # Assume single batch for now, expand to match features
        if features.shape[0] > time_feat.shape[0]:
            # Repeat time embedding for all points
            time_feat = time_feat.repeat(features.shape[0] // time_feat.shape[0] + 1, 1)[:features.shape[0]]
        x_feat = features + time_feat
        
        # Apply condition
        cond_feat = self.condition_proj(condition)
        x_feat = x_feat + cond_feat
        
        # Transformer processing
        x_transformed = self.transformer(x_feat, coords, neighbors)
        
        # Output projection
        out = self.out_proj(x_transformed)
        
        return out


class DenoisingNetwork(nn.Module):
    """
    U-Net style denoising network using Sonata transformer blocks.
    
    Predicts noise at each diffusion step, conditioned on:
    - Partial input scan
    - Sonata encoder features
    - Timestep
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # xyz coordinates
        condition_dim: int = 256,
        hidden_dims: list = [64, 128, 256, 512],
        time_embed_dim: int = 128,
        num_neighbors: int = 16,
    ):
        """
        Initialize denoising network.
        
        Args:
            in_channels: Input point feature dimension
            condition_dim: Conditional feature dimension from Sonata
            hidden_dims: Hidden dimensions for U-Net levels
            time_embed_dim: Time embedding dimension
            num_neighbors: Neighbors for local processing
        """
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        self.hidden_dims = hidden_dims
        
        # Sinusoidal time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dims[0])
        
        # Encoder blocks (downsampling)
        self.encoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            # Diffusion block
            self.encoder_blocks.append(
                PointwiseDiffusionBlock(
                    hidden_dims[i], hidden_dims[i],
                    time_embed_dim, condition_dim, num_neighbors
                )
            )
        
        # Bottleneck
        self.bottleneck = PointwiseDiffusionBlock(
            hidden_dims[-1], hidden_dims[-1],
            time_embed_dim, condition_dim, num_neighbors
        )
        
        # Decoder blocks (upsampling)
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            # Diffusion block with skip connection
            self.decoder_blocks.append(
                PointwiseDiffusionBlock(
                    hidden_dims[i - 1] + hidden_dims[i], hidden_dims[i - 1],
                    time_embed_dim, condition_dim, num_neighbors
                )
            )
        
        # Output projection (predict noise)
        self.output_proj = nn.Linear(hidden_dims[0], in_channels)
    
    def _downsample_points(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        target_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Downsample points using farthest point sampling."""
        N = features.shape[0]
        if N <= target_num:
            return features, coords
        
        # Simple uniform sampling (can be replaced with FPS)
        indices = torch.linspace(0, N - 1, target_num, dtype=torch.long, device=features.device)
        return features[indices], coords[indices]
    
    def _upsample_points(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        target_coords: torch.Tensor
    ) -> torch.Tensor:
        """Upsample features to target coordinates using nearest neighbor interpolation."""
        # Find nearest neighbors
        coords_np = coords.detach().cpu().numpy()
        target_np = target_coords.detach().cpu().numpy()
        tree = cKDTree(coords_np)
        _, indices = tree.query(target_np, k=1)
        indices = torch.from_numpy(indices).long().to(features.device)
        return features[indices]
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        timestep: torch.Tensor,
        condition: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict noise for denoising step.
        
        Args:
            features: (N, in_channels) noisy point features
            coords: (N, 3) point coordinates
            timestep: Current timestep (batch_size,)
            condition: Conditional features from Sonata encoder
            
        Returns:
            (N, in_channels) predicted noise
        """
        # Time embedding
        t_embed = self.time_embedding(timestep)
        
        # Get conditional features
        cond_feat = condition['features']
        
        # Input projection
        x = self.input_proj(features)
        x_coords = coords
        
        # Encoder path
        skip_features = []
        skip_coords = []
        
        for i, enc_block in enumerate(self.encoder_blocks):
            # Process with transformer
            x = enc_block(x, x_coords, t_embed, cond_feat)
            
            # Save skip connection
            skip_features.append(x)
            skip_coords.append(x_coords)
            
            # Downsample for next level
            if i < len(self.encoder_blocks) - 1:
                target_num = x.shape[0] // 2
                x, x_coords = self._downsample_points(x, x_coords, target_num)
        
        # Bottleneck
        x = self.bottleneck(x, x_coords, t_embed, cond_feat)
        
        # Decoder path
        for i, dec_block in enumerate(self.decoder_blocks):
            # Upsample to match skip connection
            skip_feat = skip_features[-(i+1)]
            skip_coord = skip_coords[-(i+1)]
            
            x = self._upsample_points(x, x_coords, skip_coord)
            x_coords = skip_coord
            
            # Concatenate skip connection
            x = torch.cat([x, skip_feat], dim=-1)
            
            # Process with transformer
            x = dec_block(x, x_coords, t_embed, cond_feat)
        
        # Output projection
        noise_pred = self.output_proj(x)
        
        return noise_pred


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings.
        
        Args:
            timesteps: (batch_size,) timestep values
            
        Returns:
            (batch_size, embed_dim) embeddings
        """
        device = timesteps.device
        half_dim = self.embed_dim // 2
        
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([
            torch.sin(embeddings), torch.cos(embeddings)
        ], dim=-1)
        
        return embeddings


class SceneCompletionDiffusion(nn.Module):
    """
    Complete diffusion model for semantic scene completion.
    
    Combines:
    - Sonata encoder for conditional features
    - Diffusion scheduler for noise schedule
    - Denoising network for reverse process
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        condition_extractor: nn.Module,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
        denoising_steps: int = 50,
    ):
        """
        Initialize complete diffusion model.
        
        Args:
            encoder: Sonata encoder
            condition_extractor: Conditional feature extractor
            num_timesteps: Total diffusion steps
            schedule: Noise schedule type
            denoising_steps: Steps for inference
        """
        super().__init__()
        
        self.encoder = encoder
        self.condition_extractor = condition_extractor
        self.denoising_steps = denoising_steps
        
        # Diffusion scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            schedule=schedule
        )
        
        # Denoising network
        self.denoiser = DenoisingNetwork(
            in_channels=3,
            condition_dim=condition_extractor.out_dim,
            hidden_dims=[64, 128, 256, 512],
            time_embed_dim=128
        )
    
    def forward(
        self,
        partial_scan: Dict[str, torch.Tensor],
        complete_scan: torch.Tensor,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            partial_scan: Incomplete input scan
            complete_scan: Ground truth complete scene (N, 3)
            return_loss: Whether to compute loss
            
        Returns:
            Dictionary with predictions and losses
        """
        # Extract conditional features from partial scan
        cond_features, _ = self.condition_extractor(partial_scan)
        
        # Get coordinates from partial scan
        coords = partial_scan['coord']
        
        # Sample random timesteps
        batch_size = 1  # Assume single batch
        t = torch.randint(
            0, self.scheduler.num_timesteps, 
            (batch_size,), device=complete_scan.device
        )
        
        # Add noise to complete scan
        noise = torch.randn_like(complete_scan)
        noisy_scan = self.scheduler.q_sample(complete_scan, t, noise)
        
        # Predict noise
        pred_noise = self.denoiser(
            noisy_scan, coords, t, {'features': cond_features}
        )
        
        # Compute loss
        if return_loss:
            loss = F.mse_loss(pred_noise, noise)
            return {'loss': loss, 'pred_noise': pred_noise}
        
        return {'pred_noise': pred_noise}
    
    @torch.no_grad()
    def complete_scene(
        self,
        partial_scan: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Complete scene from partial scan (inference).
        
        Args:
            partial_scan: Incomplete input scan
            num_steps: Number of denoising steps (default: self.denoising_steps)
            
        Returns:
            Completed scene point cloud (N, 3)
        """
        if num_steps is None:
            num_steps = self.denoising_steps
        
        # Extract conditional features
        cond_features, _ = self.condition_extractor(partial_scan)
        
        # Get coordinates from partial scan
        coords = partial_scan['coord']
        
        # Start from pure noise
        x_t = torch.randn_like(coords)
        
        # Denoise step by step
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_steps, 
            dtype=torch.long, device=x_t.device
        )
        
        for t in timesteps:
            x_t = self.scheduler.p_sample_step(
                self.denoiser,
                x_t,
                coords,
                t.item(),
                {'features': cond_features}
            )
        
        return x_t


if __name__ == "__main__":
    print("Testing Diffusion Module...")
    
    # Test scheduler
    scheduler = DiffusionScheduler(num_timesteps=1000, schedule="cosine")
    print(f"Beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
    
    # Test diffusion block
    block = PointwiseDiffusionBlock(64, 128, 128, 256).cuda()
    print(f"\nDiffusion block created: {block}")
