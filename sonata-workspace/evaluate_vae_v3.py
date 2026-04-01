#!/usr/bin/env python3
"""
Evaluate VAE v3 (multi-token, centered+scaled) on val split (seq 08).

Key: VAE v3 was trained with per-sample centering + scaling to [-1,1].
We must denormalize predictions back to original space for fair CD comparison.

Flow per sample:
  1. Load raw GT from .npz
  2. Load LiDAR scan, compute scan_center = lidar.mean(axis=0)
  3. gt_shifted = gt_raw - scan_center  (what dataloader does)
  4. Normalize: centroid = gt_shifted.mean(0), scale = |gt_shifted - centroid|.max()
     gt_norm = (gt_shifted - centroid) / scale
  5. Encode+decode through VAE -> recon_norm
  6. Denormalize: recon_shifted = recon_norm * scale + centroid
  7. recon_original = recon_shifted + scan_center
  8. CD(recon_original, gt_raw)
"""

import os
import sys
import json
import time
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.point_cloud_vae import PointCloudVAE
from models.refinement_net import chamfer_distance


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path):
    return np.load(path)["points"]


def subsample(pts, max_pts):
    if pts.shape[0] > max_pts:
        idx = np.random.choice(pts.shape[0], max_pts, replace=False)
        return pts[idx]
    return pts


def normalize_points(pts):
    """Same as training: center on mean, scale to [-1,1]."""
    centroid = pts.mean(dim=0)
    pts_c = pts - centroid
    scale = pts_c.abs().max().clamp(min=1e-6)
    pts_n = pts_c / scale
    return pts_n, centroid, scale


def bev_plot(pts_dict, title, save_path):
    n = len(pts_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, pts) in zip(axes, pts_dict.items()):
        if pts is not None and len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3,
                       cmap="viridis", vmin=-2, vmax=4)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=12)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_cd(pred, gt, max_pts=10000):
    pred_s = subsample(pred, max_pts)
    gt_s = subsample(gt, max_pts)
    cd = chamfer_distance(
        torch.from_numpy(pred_s).float().cuda(),
        torch.from_numpy(gt_s).float().cuda(),
        chunk_size=512,
    )
    return cd.item()


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str,
                   default=os.path.expanduser("~/sonata_ws/dataset/sonata_depth_pro"))
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/point_vae_v3/best_point_vae.pth")
    p.add_argument("--output_dir", type=str, default="evaluation_vae_v3")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--sequence", type=str, default="08")
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--use_mean", action="store_true", default=False,
                   help="Use mu directly instead of sampling z")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    # Load checkpoint and extract config
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  Epoch: {ckpt.get('epoch', '?')}, val loss: {ckpt.get('best_val_loss', '?')}")

    # Build model with saved config
    model = PointCloudVAE(
        latent_dim=ckpt.get("latent_dim", 1024),
        num_decoded_points=ckpt.get("num_decoded_points", 8000),
        num_latent_tokens=ckpt.get("num_latent_tokens", 32),
        internal_dim=ckpt.get("internal_dim", 256),
        num_heads=ckpt.get("num_heads", 4),
        num_dec_blocks=ckpt.get("num_dec_blocks", 5),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    nparams = sum(pp.numel() for pp in model.parameters())
    print(f"  Model params: {nparams:,}")

    # Data paths
    seq_dir = os.path.join(args.data_path, "sequences", args.sequence)
    vel_dir = os.path.join(seq_dir, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", args.sequence)

    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])
    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][:args.num_samples]
    print(f"Evaluating {len(sample_frames)} frames from seq {args.sequence}")

    results = []
    cds_original = []
    cds_normalized = []

    for i, fid in enumerate(sample_frames):
        # Load raw data
        lidar_raw = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        if not os.path.exists(gt_path):
            print(f"  [{i+1}/{len(sample_frames)}] {fid}: no GT, skipping")
            continue

        gt_raw = load_gt(gt_path)

        # Step 1: scan centering (same as dataloader)
        scan_center = lidar_raw.mean(axis=0)

        # GT after scan-centering (what the dataloader gives to training)
        gt_shifted = gt_raw - scan_center
        gt_shifted_sub = subsample(gt_shifted, args.point_max_complete)

        # Step 2: normalize (same as training)
        gt_tensor = torch.from_numpy(gt_shifted_sub).float().to(device)
        gt_norm, centroid, scale = normalize_points(gt_tensor)

        # Step 3: encode + decode
        t0 = time.time()
        if args.use_mean:
            mu, logvar = model.encode(gt_norm)
            recon_norm = model.decode(mu)
        else:
            recon_norm, mu, logvar = model(gt_norm)
        elapsed = time.time() - t0

        # CD in normalized space (sanity check, should match val loss ~0.0005)
        cd_norm = chamfer_distance(
            recon_norm, gt_norm, chunk_size=512
        ).item()

        # Step 4: denormalize back to original space
        recon_shifted = recon_norm * scale + centroid  # back to scan-centered space
        recon_original = recon_shifted.cpu().numpy() + scan_center  # back to world space

        # Step 5: CD in original space (comparable to teacher CD 0.608)
        cd_orig = compute_cd(recon_original, gt_raw)

        cds_original.append(cd_orig)
        cds_normalized.append(cd_norm)

        r = {
            "frame": fid,
            "cd_original": cd_orig,
            "cd_normalized": cd_norm,
            "time": elapsed,
            "gt_points": int(gt_raw.shape[0]),
            "recon_points": int(recon_norm.shape[0]),
            "scale": float(scale.item()),
        }
        results.append(r)

        status = f"  [{i+1}/{len(sample_frames)}] {fid}: CD_orig={cd_orig:.4f}, CD_norm={cd_norm:.6f}, scale={scale.item():.2f}, time={elapsed:.3f}s"
        print(status)

        # BEV visualization (first 20 samples)
        if i < 20:
            bev_plot(
                {
                    "Input (LiDAR)": lidar_raw,
                    "VAE v3 Recon": recon_original,
                    "GT": gt_raw,
                },
                f"VAE v3 | Frame {fid} | CD={cd_orig:.3f}",
                os.path.join(args.output_dir, f"bev_{fid}.png"),
            )

    # Summary
    print("\n" + "=" * 70)
    print(f"VAE v3 Evaluation Summary ({len(cds_original)} samples)")
    print("=" * 70)
    if cds_original:
        arr = np.array(cds_original)
        arr_n = np.array(cds_normalized)
        print(f"  CD (original space):   {arr.mean():.4f} +/- {arr.std():.4f}")
        print(f"  CD (normalized space): {arr_n.mean():.6f} +/- {arr_n.std():.6f}")
        print(f"  CD median (original):  {np.median(arr):.4f}")
        print(f"  CD min/max (original): {arr.min():.4f} / {arr.max():.4f}")
        print(f"  Avg time per sample:   {np.mean([r['time'] for r in results]):.3f}s")
        print(f"  Avg scale factor:      {np.mean([r['scale'] for r in results]):.2f}")

    # Save metrics
    summary = {
        "model": "point_vae_v3",
        "checkpoint": args.ckpt,
        "epoch": ckpt.get("epoch", "?"),
        "val_loss": ckpt.get("best_val_loss", "?"),
        "num_samples": len(cds_original),
        "cd_original_mean": float(np.mean(cds_original)) if cds_original else None,
        "cd_original_std": float(np.std(cds_original)) if cds_original else None,
        "cd_normalized_mean": float(np.mean(cds_normalized)) if cds_normalized else None,
        "cd_normalized_std": float(np.std(cds_normalized)) if cds_normalized else None,
        "use_mean": args.use_mean,
        "results": results,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"BEV plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
