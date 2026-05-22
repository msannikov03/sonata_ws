#!/usr/bin/env python3
"""
Evaluate VAE-GAN checkpoint on the SemanticKITTI validation split (seq 08).

Produces the full RA-L metric suite matching evaluate_vae_v3.py, plus two
additional signals unique to the GAN:

  Discriminator realism gap:
    D(real) − D(fake) measured on the critic trained alongside the VAE.
    A gap near zero means the generator is fooling the critic well.
    Reported per-sample and as a mean ± std.

  Stochastic spread (optional, --n_stochastic > 1):
    The same frame is decoded N times using reparameterised z samples.
    The per-sample CD variance reflects how much the GAN uncertainty changes
    the shape of the output — lower variance means the generator is more
    deterministic/confident.

Normalisation flow (must match training exactly):
  gt_raw → subtract scan_center → subsample → centre + scale to [-1,1]
  → encode → reparameterise → decode → recon_norm
  → × scale + centroid + scan_center → recon_original
  → compare with gt_raw using the full metric suite.

Usage:
  # Compare GAN checkpoint against optional V3 baseline:
  python evaluate_vae_gan.py \\
      --ckpt checkpoints/vae_gan/best_vae_gan.pth \\
      --output_dir evaluation_vae_gan \\
      [--baseline_json evaluation_vae_v3/metrics.json] \\
      [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.discriminator import MultiScalePointDiscriminator
from models.point_cloud_vae import PointCloudVAE
from models.refinement_net import chamfer_distance
from evaluation.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path: str) -> np.ndarray:
    return np.load(path)["points"]


def subsample(pts: np.ndarray, max_pts: int) -> np.ndarray:
    if pts.shape[0] > max_pts:
        idx = np.random.choice(pts.shape[0], max_pts, replace=False)
        return pts[idx]
    return pts


def normalize_points(pts: torch.Tensor):
    """Centre + scale to [-1, 1]. Returns (pts_norm, centroid, scale)."""
    centroid = pts.mean(dim=0)
    pts_c = pts - centroid
    scale = pts_c.abs().max().clamp(min=1e-6)
    return pts_c / scale, centroid, scale


def to_device(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).float().to(device)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def bev_plot(pts_dict: dict, title: str, save_path: str):
    n = len(pts_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, pts) in zip(axes, pts_dict.items()):
        if pts is not None and len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2],
                       s=0.3, cmap="viridis", vmin=-2, vmax=4)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=11)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def disc_score_plot(real_scores: list, fake_scores: list, save_path: str):
    """Histogram of critic scores for real vs. generated clouds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(real_scores, bins=20, alpha=0.6, label="Real (GT)", color="steelblue")
    ax.hist(fake_scores, bins=20, alpha=0.6, label="Generated (VAE-GAN)", color="coral")
    ax.set_xlabel("Critic score D(·)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Discriminator realism: real vs. generated", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_frame(
    vae: PointCloudVAE,
    disc: MultiScalePointDiscriminator | None,
    lidar_raw: np.ndarray,
    gt_raw: np.ndarray,
    args,
    device: torch.device,
):
    """
    Full evaluation pipeline for a single frame.

    Returns a dict with all metrics plus timing info.
    """
    # ------ normalisation (must match train_vae_gan exactly) ------
    scan_center = lidar_raw.mean(axis=0)
    gt_shifted = gt_raw - scan_center
    gt_shifted_sub = subsample(gt_shifted, args.point_max_complete)

    gt_tensor = to_device(gt_shifted_sub, device)
    gt_norm, centroid, scale = normalize_points(gt_tensor)

    # ------ encode + decode ------
    t0 = time.time()
    if args.use_mean:
        mu, logvar = vae.encode(gt_norm)
        recon_norm = vae.decode(mu)
    else:
        recon_norm, mu, logvar = vae(gt_norm)
    elapsed = time.time() - t0

    # ------ CD in normalised space (matches training loss) ------
    cd_norm = chamfer_distance(recon_norm, gt_norm, chunk_size=512).item()

    # ------ denormalise → original coordinate space ------
    recon_shifted = (recon_norm * scale + centroid).cpu().numpy()
    recon_original = recon_shifted + scan_center

    # ------ full RA-L metric suite in original space ------
    ral = compute_all_metrics(
        recon_original, gt_raw,
        f_thresholds=(0.1, 0.2),
        iou_voxel_sizes=(0.1, 0.2),
        jsd_voxel_size=0.5,
    )

    # ------ discriminator realism scores ------
    real_score, fake_score = None, None
    if disc is not None:
        # Both clouds in normalised space; subsample real to K pts for fair input
        K = recon_norm.shape[0]
        gt_norm_sub = gt_norm[:K] if gt_norm.shape[0] >= K else gt_norm
        real_score = disc(gt_norm_sub.unsqueeze(0)).item()
        fake_score = disc(recon_norm.unsqueeze(0)).item()

    # ------ stochastic spread (if requested) ------
    stoch_cd_norm = None
    if args.n_stochastic > 1:
        mu, logvar = vae.encode(gt_norm)
        cds = []
        for _ in range(args.n_stochastic):
            z = vae.reparameterize(mu, logvar)
            r = vae.decode(z)
            cds.append(chamfer_distance(r, gt_norm, chunk_size=512).item())
        stoch_cd_norm = float(np.std(cds))

    result = {
        "cd_norm": cd_norm,
        "time": elapsed,
        "gt_points": int(gt_raw.shape[0]),
        "recon_points": int(recon_norm.shape[0]),
        "scale": float(scale.item()),
        "real_score": real_score,
        "fake_score": fake_score,
        "stoch_cd_std": stoch_cd_norm,
    }
    result.update(ral)  # cd, cd_sq, jsd, f_score@*, precision@*, recall@*, iou@*, hausdorff_95

    return result, recon_original


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _mean_std(values: list, key: str) -> tuple:
    vals = [r[key] for r in values if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def build_summary(results: list, args, ckpt: dict) -> dict:
    keys_to_aggregate = [
        "cd_norm", "cd", "cd_sq",
        "jsd",
        "f_score@0.1", "f_score@0.2",
        "precision@0.1", "precision@0.2",
        "recall@0.1", "recall@0.2",
        "iou@0.1", "iou@0.2",
        "hausdorff_95",
        "real_score", "fake_score",
        "time", "scale",
    ]
    agg = {}
    for k in keys_to_aggregate:
        mean, std = _mean_std(results, k)
        agg[k + "_mean"] = mean
        agg[k + "_std"] = std

    if not any(np.isnan(v) for v in [agg.get("real_score_mean", float("nan")),
                                      agg.get("fake_score_mean", float("nan"))]):
        agg["critic_realism_gap"] = agg["real_score_mean"] - agg["fake_score_mean"]
    else:
        agg["critic_realism_gap"] = float("nan")

    if args.n_stochastic > 1:
        agg["stoch_cd_std_mean"], agg["stoch_cd_std_std"] = _mean_std(results, "stoch_cd_std")

    return {
        "model": "vae_gan",
        "checkpoint": args.ckpt,
        "epoch": ckpt.get("epoch", "?"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "weights_used": "live" if args.no_ema or "vae_ema_state_dict" not in ckpt else "EMA",
        "num_samples": len(results),
        "use_mean": args.use_mean,
        "n_stochastic": args.n_stochastic,
        "aggregated": agg,
        "per_frame": results,
    }


def print_summary(summary: dict, baseline: dict | None = None):
    agg = summary["aggregated"]
    print()
    print("=" * 72)
    print(f"  VAE-GAN Evaluation Summary  —  {summary['num_samples']} frames")
    print(f"  Checkpoint : {summary['checkpoint']}")
    print(f"  Epoch      : {summary['epoch']}   Val loss: {summary['best_val_loss']}")
    print("=" * 72)

    def row(label, key, fmt=".4f", scale=1.0):
        val = agg.get(key + "_mean", float("nan")) * scale
        std = agg.get(key + "_std", float("nan")) * scale
        b_val = ""
        if baseline:
            b = baseline.get("aggregated", {}).get(key + "_mean")
            if b is not None and not np.isnan(b):
                delta = val - b * scale
                sign = "+" if delta >= 0 else ""
                b_val = f"  [baseline {b * scale:{fmt}}  Δ{sign}{delta:{fmt}}]"
        print(f"  {label:<32s} {val:{fmt}} ± {std:{fmt}}{b_val}")

    print("\n  --- Chamfer (training metric) ---")
    row("CD normalised (m²)",  "cd_norm",       ".6f")
    row("CD linear (m)",       "cd",            ".4f")
    row("CD squared (m²)",     "cd_sq",         ".6f")

    print("\n  --- Distribution similarity ---")
    row("JSD-BEV (0.5 m vox)", "jsd",           ".4f")

    print("\n  --- Point coverage ---")
    row("F-score @ 0.1 m",     "f_score@0.1",   ".4f")
    row("F-score @ 0.2 m",     "f_score@0.2",   ".4f")
    row("Precision @ 0.1 m",   "precision@0.1", ".4f")
    row("Recall @ 0.1 m",      "recall@0.1",    ".4f")

    print("\n  --- Voxel occupancy ---")
    row("Voxel IoU @ 0.1 m",   "iou@0.1",       ".4f")
    row("Voxel IoU @ 0.2 m",   "iou@0.2",       ".4f")

    print("\n  --- Worst-case geometry ---")
    row("Hausdorff-95 (m)",     "hausdorff_95",  ".4f")

    print("\n  --- GAN-specific ---")
    rs = agg.get("real_score_mean", float("nan"))
    fs = agg.get("fake_score_mean", float("nan"))
    gap = agg.get("critic_realism_gap", float("nan"))
    print(f"  {'Critic score  real':<32s} {rs:.4f} ± {agg.get('real_score_std', float('nan')):.4f}")
    print(f"  {'Critic score  generated':<32s} {fs:.4f} ± {agg.get('fake_score_std', float('nan')):.4f}")
    if not np.isnan(gap):
        interp = "generator is NOT fooling critic" if gap > 0.5 else \
                 ("generator is PARTIALLY fooling critic" if gap > 0.1 else
                  "generator is WELL fooling critic")
        print(f"  {'Critic realism gap (real−fake)':<32s} {gap:.4f}  [{interp}]")

    if summary["n_stochastic"] > 1:
        sc = agg.get("stoch_cd_std_mean", float("nan"))
        print(f"\n  {'Stochastic CD-norm std (N=%d)' % summary['n_stochastic']:<32s} {sc:.6f}")

    row("\n  Avg inference time (s)",  "time",   ".3f")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VAE-GAN on SemanticKITTI val split")
    p.add_argument("--data_path", type=str,
                   default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"))
    p.add_argument("--ckpt", type=str, default="checkpoints/vae_gan/best_vae_gan.pth")
    p.add_argument("--output_dir", type=str, default="evaluation_vae_gan")
    p.add_argument("--sequence", type=str, default="08")
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--use_mean", action="store_true", default=False,
                   help="Use mu directly instead of sampling z (deterministic mode)")
    p.add_argument("--n_stochastic", type=int, default=1,
                   help="Re-decode N times per frame to measure output variance (>1 enables)")
    p.add_argument("--no_disc", action="store_true", default=False,
                   help="Skip loading the discriminator (faster if critic scores not needed)")
    p.add_argument("--no_ema", action="store_true", default=False,
                   help="Use live VAE weights instead of EMA (default: prefer EMA when present)")
    p.add_argument("--baseline_json", type=str, default=None,
                   help="Path to a previous metrics.json (e.g., evaluate_vae_v3) for side-by-side Δ")
    p.add_argument("--device", type=str, default=None,
                   help="'cuda', 'cuda:1', 'cpu' — auto-detects if omitted")
    p.add_argument("--gt_subdir", type=str, default="ground_truth",
                   help="GT subdirectory name inside data_path")

    # VAE architecture (defaults match V3 training defaults)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--num_latent_tokens", type=int, default=32)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_dec_blocks", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  Epoch: {ckpt.get('epoch', '?')}   Val loss: {ckpt.get('best_val_loss', '?')}")

    # ---- Build VAE ----
    vae = PointCloudVAE(
        latent_dim=args.latent_dim,
        num_decoded_points=args.num_decoded_points,
        num_latent_tokens=args.num_latent_tokens,
        internal_dim=args.internal_dim,
        num_heads=args.num_heads,
        num_dec_blocks=args.num_dec_blocks,
    ).to(device)
    # Prefer EMA weights (publication-grade evaluation) when present.
    if not args.no_ema and "vae_ema_state_dict" in ckpt:
        vae.load_state_dict(ckpt["vae_ema_state_dict"])
        weight_source = "EMA"
    else:
        vae.load_state_dict(ckpt["vae_state_dict"])
        weight_source = "live"
    vae.eval()
    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"  VAE parameters: {vae_params:,}   (weights: {weight_source})")

    # ---- Build discriminator ----
    disc = None
    if not args.no_disc and "disc_state_dict" in ckpt:
        disc = MultiScalePointDiscriminator(in_dim=3).to(device)
        disc.load_state_dict(ckpt["disc_state_dict"])
        disc.eval()
        disc_params = sum(p.numel() for p in disc.parameters())
        print(f"  Discriminator parameters: {disc_params:,}")
    elif not args.no_disc:
        print("  Warning: no disc_state_dict in checkpoint — discriminator scores skipped")

    # ---- Load baseline for comparison ----
    baseline = None
    if args.baseline_json and os.path.exists(args.baseline_json):
        with open(args.baseline_json) as f:
            baseline = json.load(f)
        print(f"  Baseline: {args.baseline_json}")
    elif args.baseline_json:
        print(f"  Warning: baseline_json not found at {args.baseline_json}")

    # ---- Data paths ----
    seq_dir = os.path.join(args.data_path, "sequences", args.sequence)
    vel_dir = os.path.join(seq_dir, "velodyne")
    gt_dir = os.path.join(args.data_path, args.gt_subdir, args.sequence)

    frames = sorted(
        [f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")]
    )
    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][: args.num_samples]
    print(f"\nEvaluating {len(sample_frames)} frames from seq {args.sequence}")

    # ---- Evaluation loop ----
    results = []
    real_disc_scores = []
    fake_disc_scores = []

    for i, fid in enumerate(sample_frames):
        lidar_path = os.path.join(vel_dir, f"{fid}.bin")
        gt_path = os.path.join(gt_dir, f"{fid}.npz")

        if not os.path.exists(gt_path):
            print(f"  [{i + 1}/{len(sample_frames)}] {fid}: no GT, skipping")
            continue

        lidar_raw = load_bin(lidar_path)
        gt_raw = load_gt(gt_path)

        frame_result, recon_original = evaluate_frame(
            vae, disc, lidar_raw, gt_raw, args, device
        )
        frame_result["frame"] = fid
        results.append(frame_result)

        if frame_result["real_score"] is not None:
            real_disc_scores.append(frame_result["real_score"])
            fake_disc_scores.append(frame_result["fake_score"])

        # Per-frame progress line
        gap_str = ""
        if frame_result["fake_score"] is not None:
            gap = frame_result["real_score"] - frame_result["fake_score"]
            gap_str = f"  D_gap={gap:+.3f}"
        print(
            f"  [{i + 1:3d}/{len(sample_frames)}] {fid} | "
            f"CD_norm={frame_result['cd_norm']:.6f}  "
            f"CD={frame_result['cd']:.4f}  "
            f"F@0.1={frame_result['f_score@0.1']:.3f}  "
            f"JSD={frame_result['jsd']:.4f}"
            f"{gap_str}"
        )

        # BEV visualisation for first 20 samples
        if i < 20:
            bev_plot(
                {
                    "Input (LiDAR)": lidar_raw,
                    "VAE-GAN Recon": recon_original,
                    "GT": gt_raw,
                },
                f"VAE-GAN | Frame {fid} | CD={frame_result['cd']:.3f}  F@0.1={frame_result['f_score@0.1']:.3f}",
                os.path.join(args.output_dir, f"bev_{fid}.png"),
            )

    # ---- Discriminator score histogram ----
    if real_disc_scores:
        disc_score_plot(
            real_disc_scores, fake_disc_scores,
            os.path.join(args.output_dir, "critic_score_histogram.png"),
        )

    # ---- Build and print summary ----
    summary = build_summary(results, args, ckpt)
    print_summary(summary, baseline=baseline)

    # ---- Save JSON ----
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved  → {metrics_path}")
    print(f"BEV plots      → {args.output_dir}/bev_*.png")
    if real_disc_scores:
        print(f"Critic histogram → {args.output_dir}/critic_score_histogram.png")


if __name__ == "__main__":
    main()
