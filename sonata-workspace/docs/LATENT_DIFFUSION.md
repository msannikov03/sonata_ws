# Point-cloud VAE + latent diffusion (v2)

This track keeps **geometry in point space** (no dense voxel SSC grid).
A **PointNet-style VAE** (Gaussian or VQ-VAE) compresses the complete scene
to a latent **z**, and a **token-based transformer denoiser** runs
**DDPM / DDIM** in **z**-space, conditioned on **Perceiver-pooled Sonata
features** from the partial scan.

## Key changes from v1

| Issue (v1) | Fix (v2) |
|---|---|
| No latent normalisation → noise schedule at wrong SNR → inference from pure noise fails | **LatentNormalizer**: running-EMA mean/var; z scaled to ~N(0,1) before diffusion |
| Mean-pooled conditioning → all spatial info lost | **ConditionPooler**: Perceiver cross-attention compresses per-point Sonata features into 32 learned tokens |
| 3-layer MLP denoiser (~720K params) | **LatentDenoiser**: 8-block DiT-style transformer with self-attn + cross-attn + AdaLN (~7M params) |
| VQ-VAE codebook never updated (`@torch.no_grad` on `quantize()`) | Removed decorator; codebook loss now flows gradients correctly |
| VQ-VAE codebook init too small (`uniform(±0.001)`) | Default `nn.Embedding` init (~N(0,1)) |

## Pipeline

1. **VAE** (`models/point_cloud_vae.py` or `models/point_cloud_vq_vae.py`):
   encoder over complete xyz → global max-pool → latent **z**;
   decoder maps **z → K points**. Loss: Chamfer + KL (Gaussian) or Chamfer + VQ losses.

2. **Latent diffusion** (`models/latent_diffusion.py`):
   target **z₀ = normalize(encode(complete))**. Train **ε-prediction** MSE.
   Condition: **Sonata** on partial points → per-point features →
   **ConditionPooler** → (B, 32, 256) tokens → cross-attended by denoiser.

3. **Dataset** (`data/semantickitti.py`): `use_point_cloud=True` skips
   voxel merging; only random subsamples xyz for partial and complete.

## Training order

```bash
cd sonata-workspace
conda activate sonata_lidiff   # or your env

# 1) Latent autoencoder (pick one)
# (A) Gaussian VAE:
python training/train_point_vae.py \
  --data_path /path/to/SemanticKITTI/dataset \
  --output_dir checkpoints/point_vae

# (B) VQ-VAE:
python training/train_point_vq_vae.py \
  --data_path /path/to/SemanticKITTI/dataset \
  --output_dir checkpoints/point_vq_vae

# 2) Latent diffusion (frozen VAE by default)
python training/train_diffusion_latent.py \
  --vae_ckpt checkpoints/point_vae/best_point_vae.pth \
  --data_path /path/to/SemanticKITTI/dataset \
  --freeze_encoder

# If using VQ-VAE, swap --vae_ckpt:
#   --vae_ckpt checkpoints/point_vq_vae/best_point_vq_vae.pth
```

Optional YAML config (still pass `--vae_ckpt` on CLI):

```bash
python training/train_diffusion_latent.py \
  --config configs/latent_diffusion.yaml \
  --vae_ckpt checkpoints/point_vae/best_point_vae.pth
```

Resume from a latent diffusion checkpoint (VAE weights are inside it):

```bash
python training/train_diffusion_latent.py \
  --resume checkpoints/latent_diffusion/checkpoint_epoch_10.pth
```

## Denoiser architecture args

| Flag | Default | Description |
|---|---|---|
| `--hidden_dim` | 1024 | Denoiser internal width (split across latent tokens) |
| `--num_denoiser_blocks` | 8 | Transformer blocks (self-attn + cross-attn + MLP each) |
| `--num_latent_tokens` | 8 | z is split into this many tokens for self-attention |
| `--num_cond_tokens` | 32 | Perceiver queries compressing per-point Sonata features |
| `--num_heads` | 4 | Attention heads per block |
| `--time_embed_dim` | 256 | Sinusoidal time embedding + MLP dimension |

## Inference

```bash
python inference_latent.py \
  --input /path/to/scan.bin \
  --checkpoint checkpoints/latent_diffusion/best_latent_diffusion.pth \
  --output out.ply \
  --denoising_steps 50 \
  --sampling ddim
```

Architecture hyperparameters are saved in the checkpoint and restored
automatically at inference time.

## Outputs

- VAE checkpoint: `best_point_vae.pth` (or `best_point_vq_vae.pth`).
- Latent diffusion: `best_latent_diffusion.pth` (full state dict + architecture
  hyperparams + LatentNormalizer running stats).

## Notes

- **K** decoded points is fixed by `num_decoded_points` (VAE); completion
  is **not** the same as the full-resolution coordinate diffusion in
  `SceneCompletionDiffusion`.
- The theoretical CD floor is bounded by the VAE's reconstruction quality
  (encode GT → decode). A perfect latent diffusion model approaches but
  cannot beat this floor.
- For standard **voxel SSC mIoU**, you would still need to rasterize the
  K points (or train a separate voxel head).
