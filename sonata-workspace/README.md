# CloudDiffusion (Sonata-LiDiff + VoxFormerDepthPro)

Code for the SMC 2026 **CloudDiffusion** paper (with an RA-L companion): diffusion-based
3D semantic scene completion on SemanticKITTI, built on a frozen **Sonata** encoder and a
**coordinate-space diffusion** teacher, plus a multi-token Gaussian VAE and a drift-reduced
ground-truth ("boost v2") build.

Unified workspace for:

- **Sonata-LiDiff** – diffusion-based 3D semantic scene completion on SemanticKITTI.
- **VoxFormerDepthPro** – RGB → Depth Pro → LiDAR-style point clouds for Sonata.

It supports training Sonata on real LiDAR (SemanticKITTI velodyne), on pseudo-LiDAR from
RGB (Depth Pro), and comparing both pipelines end-to-end.

## Headline results (what this repo reproduces)

| Component | Number | Entry point | Canonical artifact |
|---|---|---|---|
| Multi-token Gaussian VAE | **CD 0.120** | `training/train_point_vae.py` (train) → `eval_vae_multitoken.py` (eval) | `models/point_cloud_vae.py` (`PointCloudVAE`) |
| Coordinate-space diffusion teacher | — | `training/train_diffusion.py` → `checkpoints/diffusion_v2gt/best_model.pth` | `models/diffusion_module.py` (`SceneCompletionDiffusion`) |
| GT v2 / v1 Chamfer | **CD 0.024 / 0.396** | `evaluate_ral_metrics.py` (via `run_ral_metrics_all.sh`) | `eval_v2gt_fullval/.../all_metrics.json`, `eval_v1gt_fullval/.../all_metrics.json` |
| Drift-reduced GT build (boost v2) | — | `data/map_from_scans_boost_v2.py` + `scripts/run_boost_v2_gt.sh` | `data/MAP_FROM_SCANS_BOOST_V2_RUNBOOK.md` |
| Inference latency | **209 ms** | `run_latency_benchmark.py` | `results/apr17_morning/latency_benchmark.json` |
| RA-L stats (Wilcoxon) | — | `verify_ral_numbers.py`, `wilcoxon_test.py` | root `eval_*.json` summaries |

> **Naming trap:** the final multi-token Gaussian VAE is the **un-suffixed**
> `models/point_cloud_vae.py`. The file named `models/point_cloud_vae_v2.py`
> (`PointCloudVAEv2`) is the *older* global-pool + FoldingNet design — do not confuse them.

## Reproduce the VAE result end-to-end

1. **Install** (from this `sonata-workspace/` dir): `pip install -e .` (needs `requirements.txt`, now present).
2. **Build the GT** first — the dataset loads `ground_truth/<seq>/<id>.npz`. Run the boost-v2
   pipeline per `data/MAP_FROM_SCANS_BOOST_V2_RUNBOOK.md` (writes into `ground_truth/`).
3. **Train the VAE** (final config; the README §6.1 command, *not* the older
   `run_vae_training.sh` which encodes a smaller/stale config):
   ```bash
   python training/train_point_vae.py \
     --data_path <DATASET_ROOT>/sonata_depth_pro \
     --output_dir checkpoints/point_vae_multitoken \
     --latent_dim 1024 --num_latent_tokens 16 \
     --num_decoded_points 4096 --num_dec_blocks 3 --beta_kl 1e-4
   ```
4. **Eval**: `python eval_vae_multitoken.py` (expects the checkpoint under
   `checkpoints/point_vae_multitoken/`).

Replace `<DATASET_ROOT>` with your SemanticKITTI / `sonata_depth_pro` root. Many scripts in
this repo still hardcode personal absolute paths (`/home/anywherevla/...`, `~/Simon_ws/...`);
prefer the argparse flags shown above and adjust paths to your machine.

---

## 1. Environment

Create and activate your `sonata_lidiff` environment (if not already):

```bash
conda activate sonata_lidiff
cd sonata-workspace
pip install -e .
```

Depth Pro is installed separately (see `VoxFormerDepthPro/README.md`), and its checkpoints are under `~/Simon_ws/ml-depth-pro/checkpoints/`.

---

## 2. Datasets

All KITTI / SemanticKITTI archives are assumed to be downloaded to:

```text
~/Simon_ws/dataset/
  data_odometry_color.zip
  data_odometry_velodyne.zip
  data_odometry_labels.zip
  data_odometry_calib.zip
  data_odometry_poses.zip
  data_odometry_voxels.zip
```

They have been extracted (once) into:

```text
~/Simon_ws/dataset/SemanticKITTI/dataset/
  sequences/XX/
    image_2/      # RGB
    velodyne/     # LiDAR
    labels/       # LiDAR labels
    voxels/       # voxel labels + masks
    calib.txt
    poses.txt
  poses/XX.txt    # original KITTI odometry poses
```

> Default paths in the code point to `~/Simon_ws/dataset/SemanticKITTI/dataset` and `~/Simon_ws/dataset/VoxFormerDepthPro_out`, so you rarely need to type them.

For more dataset detail, see `VoxFormerDepthPro/DATASET_AND_RUN.md`.

---

## 3. One‑command pipelines

From the repo root:

```bash
cd sonata-workspace
conda activate sonata_lidiff
```

### 3.1 LiDAR → Sonata (baseline, original velodyne)

Runs:

1. Generate ground truth maps from **LiDAR** (`map_from_scans.py`, GPU voxelization via PyTorch).
2. Train **Sonata-LiDiff diffusion** on this LiDAR dataset.
3. Train **refinement network** on the same data.

Command:

```bash
python scripts/run_lidar_to_sonata.py
```

Outputs:

- Checkpoints: `checkpoints/diffusion_lidar`, `checkpoints/refinement_lidar`
- Logs: `logs/diffusion_lidar`, `logs/refinement_lidar`
- GT maps: `~/Simon_ws/dataset/SemanticKITTI/dataset/ground_truth/XX/*.npz`

### 3.2 RGB → Depth Pro → Sonata (VoxFormerDepthPro + Sonata)

Runs:

1. **VoxFormerDepthPro**:
   - Preprocess voxel labels.
   - Depth Pro on RGB (`image_2/`) → depth maps (`.npy`).
   - Depth → LiDAR-style point clouds (`.bin`).
   - Assign voxel labels to Depth Pro point clouds.
2. Build a **separate dataset root** for Depth Pro point clouds:
   - `~/Simon_ws/dataset/sonata_depth_pro/` with `sequences/XX/{velodyne,labels,poses.txt,calib.txt}`.
3. Generate **ground truth maps** from this dataset (`map_from_scans.py`, GPU voxelization).
4. Train **Sonata-LiDiff diffusion** and **refinement** on this Depth‑Pro dataset.

Command:

```bash
python scripts/run_depthpro_to_sonata.py
```

Outputs:

- Depth Pro point clouds & labels:
  - `~/Simon_ws/dataset/VoxFormerDepthPro_out/...`
  - `~/Simon_ws/dataset/sonata_depth_pro/sequences/XX/{velodyne,labels,...}`
- GT maps (Depth Pro dataset):
  - `~/Simon_ws/dataset/sonata_depth_pro/ground_truth/XX/*.npz`
- Checkpoints:
  - `checkpoints/diffusion_depthpro`, `checkpoints/refinement_depthpro`
- Logs:
  - `logs/diffusion_depthpro`, `logs/refinement_depthpro`

---

## 4. Individual components (if you want to run them manually)

### 4.1 VoxFormerDepthPro

Entry scripts (defaults use paths in `VoxFormerDepthPro/paths_config.py`):

- `scripts/1_prepare_labels.py` – voxel label preprocessing.
- `scripts/2_run_depth_pro.py` – run Depth Pro on one sequence (`--seq 00` etc.).
- `scripts/3_depth_to_pointcloud.py` – depth → `.bin` for multiple sequences.
- `scripts/4_assign_labels_from_voxels.py` – voxel labels → per-point labels.

See `VoxFormerDepthPro/README.md` and `VoxFormerDepthPro/DATASET_AND_RUN.md` for details.

### 4.2 Ground truth generation (`map_from_scans.py`)

Generates complete scene maps from sequential scans:

```bash
python data/map_from_scans.py \
  --path   ~/Simon_ws/dataset/SemanticKITTI/dataset/sequences \
  --output ~/Simon_ws/dataset/SemanticKITTI/dataset \
  --voxel_size 0.1 \
  --backend torch \
  --sequences 00 01 02 03 04 05 06 07 08 09 10
```

- Uses **GPU** if available when `--backend torch`.
- Filters moving-object labels (252–259) when labels are present.

### 4.3 Training scripts

- Diffusion model:

  ```bash
  python training/train_diffusion.py \
    --data_path ~/Simon_ws/dataset/SemanticKITTI/dataset \
    --output_dir checkpoints/diffusion_lidar \
    --log_dir logs/diffusion_lidar
  ```

- Refinement network:

  ```bash
  python training/train_refinement.py \
    --data_path ~/Simon_ws/dataset/SemanticKITTI/dataset \
    --output_dir checkpoints/refinement_lidar \
    --log_dir logs/refinement_lidar
  ```

Replace `--data_path` with `~/Simon_ws/dataset/sonata_depth_pro` to train on Depth Pro instead of LiDAR.

---

## 5. Comparison idea (LiDAR vs RGB→Depth Pro)

After running both orchestration scripts:

- **LiDAR‑trained models**:
  - `checkpoints/diffusion_lidar`, `checkpoints/refinement_lidar`
- **Depth‑Pro‑trained models**:
  - `checkpoints/diffusion_depthpro`, `checkpoints/refinement_depthpro`

You can now:

- Use the same evaluation code / inference scripts (e.g. `inference.py`, `evaluation/`) with different checkpoints.
- Compare how Sonata performs when:
  - Input = original LiDAR scans vs.
  - Input = point clouds reconstructed from RGB via Depth Pro.

This is the core experiment this workspace is structured to support.

---


## 6. Point-cloud latent pipeline (Gaussian VAE + VQ-VAE, fixed)

This section describes the current **fixed** point-cloud latent pipelines used
before latent diffusion training.

### 6.1 Gaussian VAE pipeline (fixed)

The Gaussian VAE is now a **multi-token latent VAE** (not the old single-vector
maxpool + MLP decoder design):

- Encoder: residual PointNet-style per-point MLP + cross-attention pooler.
- Latent: `num_latent_tokens` Gaussian tokens flattened to `latent_dim`.
- Decoder: learned point queries + cross-attention decoder blocks.

This improves reconstruction quality and keeps the external interface unchanged
for latent diffusion (`encode_batched` still returns `(mu, logvar)` with shape
`(B, latent_dim)`).

Recommended command:

```bash
cd sonata-workspace
python training/train_point_vae.py \
  --data_path /path/to/SemanticKITTI/dataset \
  --output_dir checkpoints/point_vae \
  --latent_dim 1024 \
  --num_latent_tokens 16 \
  --num_decoded_points 4096 \
  --num_dec_blocks 3 \
  --beta_kl 1e-4
```

### 6.2 VQ-VAE pipeline (fixed)

The VQ-VAE now uses **Residual Vector Quantization (RVQ)** instead of a single
codebook embedding per scene:

- Quantizer: multi-level residual quantization (`num_quantizers`, default 8).
- Effective capacity: `num_codes^num_quantizers` combinations.
- Better resistance to codebook collapse than single-codebook setup.

Recommended command:

```bash
python training/train_point_vq_vae.py \
  --data_path /path/to/SemanticKITTI/dataset \
  --output_dir checkpoints/point_vq_vae \
  --latent_dim 256 \
  --num_codes 1024 \
  --num_quantizers 8 \
  --num_decoded_points 2048
```

### 6.3 Latent diffusion on top of either autoencoder

Train latent diffusion (DiT-style denoiser + latent normalizer + Perceiver
condition pooler) after training one autoencoder:

```bash
# Gaussian VAE checkpoint example:
python training/train_diffusion_latent.py \
  --vae_ckpt checkpoints/point_vae/best_point_vae.pth \
  --data_path /path/to/SemanticKITTI/dataset \
  --freeze_encoder

# VQ-VAE checkpoint example:
# python training/train_diffusion_latent.py \
#   --vae_ckpt checkpoints/point_vq_vae/best_point_vq_vae.pth \
#   --data_path /path/to/SemanticKITTI/dataset \
#   --freeze_encoder

# Inference -> PLY (K decoded points)
python inference_latent.py \
  --input /path/to/00/velodyne/000000.bin \
  --checkpoint checkpoints/latent_diffusion/best_latent_diffusion.pth \
  --output completion.ply
```

Notes:

- `SemanticKITTI(..., use_point_cloud=True)` is used in these pipelines. Raw
  points are subsampled; complete-side semantic labels are placeholders in this mode.
- Checkpoint loading in `train_diffusion_latent.py` and `inference_latent.py`
  now auto-detects:
  - Gaussian VAE v2 (`mu_proj`, `enc_pooler`, `point_queries` keys),
  - RVQ VQ-VAE (`residual_vq.codebooks.*.weight` keys).

