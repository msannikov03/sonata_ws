# Point-cloud VAE + latent diffusion

This track keeps **geometry in point space** (no dense voxel SSC grid): a **PointNet-style VAE** compresses the complete scene to a Gaussian latent **z**, and a small **MLP denoiser** runs **DDPM** in **z**-space, conditioned on **mean-pooled Sonata features** from the partial scan.

## Pipeline

1. **VAE** (`models/point_cloud_vae.py`): encoder over partial/complete xyz → global max-pool → `mu`, `logvar`; decoder maps **z → K points**. Loss: Chamfer(recon, target) + β·KL.
2. **Latent diffusion** (`models/latent_diffusion.py`): target **z₀ = encode(complete)** (encoder mean **μ** while VAE is frozen). Train **ε-prediction** MSE in latent space. Condition: **Sonata** on partial points → fused per-point features → **mean pool per batch item**.
3. **Dataset** (`data/semantickitti.py`): `use_point_cloud=True` skips voxel merging; only random **subsamples** xyz for partial and complete. Complete-side **semantic labels are zeros** (GT maps are xyz-only; labels are not carried after subsampling).

## Training order

```bash
cd sonata-workspace
conda activate sonata_lidiff   # or your env

# 1) VAE on complete clouds
# (A) Gaussian VAE:
python training/train_point_vae.py \
  --data_path /path/to/SemanticKITTI/dataset \
  --output_dir checkpoints/point_vae \
  --log_dir logs/point_vae

# (B) VQ-VAE:
python training/train_point_vq_vae.py \
  --data_path /path/to/SemanticKITTI/dataset \
  --output_dir checkpoints/point_vq_vae \
  --log_dir logs/point_vq_vae

# 2) Latent diffusion (frozen VAE by default)
python training/train_diffusion_latent.py \
  --vae_ckpt checkpoints/point_vae/best_point_vae.pth \
  --data_path /path/to/SemanticKITTI/dataset \
  --freeze_encoder \
  --output_dir checkpoints/latent_diffusion \
  --log_dir logs/latent_diffusion

# If using VQ-VAE, swap --vae_ckpt:
#   --vae_ckpt checkpoints/point_vq_vae/best_point_vq_vae.pth
```

Optional YAML (still pass `--vae_ckpt` on the CLI unless you embed it in a custom YAML):

```bash
python training/train_diffusion_latent.py \
  --config configs/latent_diffusion.yaml \
  --vae_ckpt checkpoints/point_vae/best_point_vae.pth
```

Resume without a separate VAE path (weights come from the latent checkpoint):

```bash
python training/train_diffusion_latent.py --resume checkpoints/latent_diffusion/checkpoint_epoch_10.pth
```

Joint fine-tune (unfreezes VAE; optional posterior sampling for z₀):

```bash
python training/train_diffusion_latent.py \
  --vae_ckpt checkpoints/point_vae/best_point_vae.pth \
  --train_vae \
  --use_posterior_sample \
  ...
```

## Inference

```bash
python inference_latent.py \
  --input /path/to/scan.bin \
  --checkpoint checkpoints/latent_diffusion/best_latent_diffusion.pth \
  --output out.ply \
  --denoising_steps 50
```

Sonata grid for conditioning: `grid_coord = floor(coord / voxel_size_sonata)` (default **0.05**), same as training (`--voxel_size_sonata`).

## Outputs

- VAE checkpoint: `best_point_vae.pth` (includes `latent_dim`, `num_decoded_points` for rebuilding).
- Latent diffusion: `best_latent_diffusion.pth` (full `SceneCompletionLatentDiffusion` state dict; also stores `num_timesteps`, `schedule` when saved from `train_diffusion_latent.py`).

## Notes

- **K** decoded points is fixed by `num_decoded_points` (VAE); completion is **not** the same as the full-resolution coordinate diffusion in `SceneCompletionDiffusion`.
- For standard **voxel SSC mIoU**, you would still need to rasterize the K points (or train a separate voxel head).
