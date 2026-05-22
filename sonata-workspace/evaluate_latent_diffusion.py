#!/usr/bin/env python3
"""
Evaluate the latent diffusion model on SemanticKITTI val split (seq 08).

Unlike the VAE evaluations (which reconstruct from the GT cloud), this script
runs the full scene *completion* pipeline: it feeds only the raw partial LiDAR
scan to the model and asks it to generate the complete scene from scratch.

Inference flow per frame:
  1. Load raw LiDAR .bin   → (N, 3) partial scan in sensor frame
  2. Compute scan_center   = partial_raw.mean(0)
  3. partial_shifted       = partial_raw − scan_center
  4. Build Sonata dict     (coord, color, normal, grid_coord, batch)
  5. model.complete_scene() → DDIM denoise → VAE.decode() → (K, 3) scan-centred pts
  6. recon_original        = pts + scan_center   (back to world frame)
  7. Metrics vs GT in world frame

Diffusion-specific signals (beyond the shared RA-L suite):
  • Denoising steps sweep (--steps_sweep): runs the same frame with multiple
    DDIM step counts (e.g. 10, 20, 50) and reports how quality changes with
    compute budget.
  • Stochastic diversity (--n_samples > 1): runs N completions per frame,
    reports mean CD across samples and the pairwise spread — measures how
    deterministic / diverse the diffusion process is.
  • eta sweep (--eta_sweep): compares DDIM deterministic (eta=0) vs DDPM
    stochastic (eta=1) sampling.

Usage on GPU server:
  # Standard evaluation (50 DDIM steps, one sample per frame):
  python evaluate_latent_diffusion.py \\
      --ckpt checkpoints/latent_diffusion/best_latent_diffusion.pth \\
      --vae_ckpt checkpoints/point_vae_v3/best_point_vae.pth \\
      --output_dir evaluation_latent_diffusion \\
      [--baseline_json evaluation_vae_gan/metrics.json] \\
      [--device cuda]

  # Diversity / stochastic spread:
  python evaluate_latent_diffusion.py --ckpt ... --n_samples 5

  # DDIM steps ablation:
  python evaluate_latent_diffusion.py --ckpt ... --steps_sweep 10 20 50 100
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

from models.latent_diffusion import SceneCompletionLatentDiffusion
from models.point_cloud_vae import PointCloudVAE
from models.refinement_net import chamfer_distance
from models.sonata_encoder import ConditionalFeatureExtractor, SonataEncoder
from evaluation.metrics import compute_all_metrics
from training.train_diffusion_latent import (
    _infer_vae_from_full_state_dict,
    _infer_denoiser_hparams,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path: str) -> np.ndarray:
    return np.load(path)["points"]


def subsample(pts: np.ndarray, n: int) -> np.ndarray:
    if pts.shape[0] > n:
        idx = np.random.choice(pts.shape[0], n, replace=False)
        return pts[idx]
    return pts


def build_partial_scan_dict(
    partial_xyz: np.ndarray,
    device: torch.device,
    voxel_size: float = 0.05,
) -> dict:
    """
    Convert raw (scan-centred) numpy array to the dict expected by
    SceneCompletionLatentDiffusion.encode_condition().

    Sonata needs 9-channel input features = [coord | color | normal].
    Color is approximated from normalised height; normals are zero-initialised
    (matching the SemanticKITTI dataloader's default behaviour).
    """
    coord = torch.from_numpy(partial_xyz).float().to(device)   # (N, 3)
    N = coord.shape[0]

    # Height-to-colour (same as dataloader)
    z_vals = coord[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_norm = (z_vals - z_min) / (z_max - z_min + 1e-6)
    color = z_norm.unsqueeze(1).expand(-1, 3)  # (N, 3) greyscale height

    normal = torch.zeros_like(coord)            # (N, 3) — no normals available

    # Grid coordinates: floor(coord / voxel_size), shifted to non-negative
    gc = torch.floor(coord / voxel_size).long()
    gc = gc - gc.min(dim=0)[0]

    batch = torch.zeros(N, dtype=torch.long, device=device)

    return {
        "coord": coord,
        "color": color,
        "normal": normal,
        "grid_coord": gc,
        "batch": batch,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(
    ckpt: dict,
    device: torch.device,
    denoising_steps: int | None = None,
) -> SceneCompletionLatentDiffusion:
    """
    Reconstruct SceneCompletionLatentDiffusion entirely from a checkpoint dict.

    Uses pretrained="random" for the Sonata encoder so that no HuggingFace
    download is required — the checkpoint already contains all Sonata weights.
    """
    sd = ckpt["model_state_dict"]

    # --- infer VAE architecture ---
    vae_kind, ld, nk, num_codes, num_q, vae_extra = _infer_vae_from_full_state_dict(sd)
    if vae_kind != "gaussian_vae":
        raise ValueError(
            f"Only Gaussian VAE (V3) is supported by this eval script. "
            f"Got: {vae_kind}"
        )
    from models.point_cloud_vq_vae import PointCloudVQVAE
    vae = PointCloudVAE(latent_dim=ld, num_decoded_points=nk, **vae_extra).to(device)

    # --- infer denoiser architecture ---
    dh = _infer_denoiser_hparams(sd)

    # --- infer feature-extractor out_dim from cond_pooler ---
    # cond_pooler.attn.in_proj_weight shape: (3*feat_dim, feat_dim)
    cond_pooler_key = "cond_pooler.attn.in_proj_weight"
    if cond_pooler_key in sd:
        cond_dim = sd[cond_pooler_key].shape[1]
    else:
        cond_dim = 256  # default (one level, concat)

    # --- build Sonata encoder (random init; weights loaded from ckpt below) ---
    # Use enable_flash=False for eval stability; can be overridden if needed.
    encoder = SonataEncoder(
        pretrained="random",
        freeze=True,
        enable_flash=False,
        feature_levels=[0],
    ).to(device)

    # Derive how many feature levels were used from cond_dim
    # (each level contributes 256 dims in concat mode)
    n_levels = max(1, cond_dim // 256)
    levels = list(range(n_levels))

    cond_ext = ConditionalFeatureExtractor(
        encoder, feature_levels=levels, fusion_type="concat",
    ).to(device)

    # --- build full model ---
    num_t = ckpt.get("num_timesteps", 1000)
    sched = ckpt.get("schedule", "cosine")
    den_steps = denoising_steps if denoising_steps is not None else ckpt.get("denoising_steps", 50)

    model = SceneCompletionLatentDiffusion(
        vae=vae,
        condition_extractor=cond_ext,
        num_timesteps=num_t,
        schedule=sched,
        denoising_steps=den_steps,
        hidden_dim=dh["hidden_dim"],
        num_denoiser_blocks=dh["num_denoiser_blocks"],
        num_latent_tokens=dh["num_latent_tokens"],
        num_cond_tokens=dh["num_cond_tokens"],
        num_heads=4,
        time_embed_dim=dh["time_embed_dim"],
    ).to(device)

    # --- load full state dict (includes Sonata, VAE, denoiser, normalizer) ---
    model.load_state_dict(sd)
    model.eval()
    return model


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


def steps_quality_plot(steps_arr: list, cd_arr: list, f_arr: list, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps_arr, cd_arr, "o-", color="steelblue")
    ax1.set_xlabel("DDIM steps")
    ax1.set_ylabel("CD linear (m)")
    ax1.set_title("Chamfer vs. denoising steps")
    ax1.grid(True, alpha=0.4)

    ax2.plot(steps_arr, f_arr, "o-", color="coral")
    ax2.set_xlabel("DDIM steps")
    ax2.set_ylabel("F-score @ 0.1 m")
    ax2.set_title("F-score vs. denoising steps")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def diversity_plot(cd_between_samples: list, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cd_between_samples, bins=20, color="mediumseagreen", alpha=0.8)
    ax.set_xlabel("Mean pairwise CD between completions (m)")
    ax.set_ylabel("Count")
    ax.set_title("Stochastic diversity of diffusion completions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Single-frame inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def complete_frame(
    model: SceneCompletionLatentDiffusion,
    partial_dict: dict,
    num_steps: int,
    eta: float = 0.0,
) -> np.ndarray:
    """Run one DDIM completion, returns (K, 3) numpy array in scan-centred space."""
    pts = model.complete_scene(partial_dict, num_steps=num_steps, eta=eta)
    return pts.cpu().numpy()


@torch.no_grad()
def evaluate_frame(
    model: SceneCompletionLatentDiffusion,
    partial_raw: np.ndarray,
    gt_raw: np.ndarray,
    args,
    device: torch.device,
):
    """
    Full evaluation for one frame.

    Returns (result_dict, recon_original_np).
    """
    scan_center = partial_raw.mean(axis=0)
    partial_shifted = partial_raw - scan_center

    # Subsample partial scan (Sonata handles variable-length input, but cap for memory)
    if args.point_max_partial > 0:
        partial_shifted = subsample(partial_shifted, args.point_max_partial)

    partial_dict = build_partial_scan_dict(partial_shifted, device, args.voxel_size)

    t0 = time.time()
    recon_shifted = complete_frame(model, partial_dict, args.num_steps, args.eta)
    elapsed = time.time() - t0

    recon_original = recon_shifted + scan_center

    # RA-L metric suite in original world space
    ral = compute_all_metrics(
        recon_original, gt_raw,
        f_thresholds=(0.1, 0.2),
        iou_voxel_sizes=(0.1, 0.2),
        jsd_voxel_size=0.5,
    )

    result = {
        "time": elapsed,
        "gt_points": int(gt_raw.shape[0]),
        "recon_points": int(recon_shifted.shape[0]),
        "partial_points": int(partial_shifted.shape[0]),
    }
    result.update(ral)
    return result, recon_original


# ---------------------------------------------------------------------------
# Stochastic diversity analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def stochastic_diversity(
    model: SceneCompletionLatentDiffusion,
    partial_dict: dict,
    n_samples: int,
    num_steps: int,
    eta: float,
    device: torch.device,
) -> float:
    """
    Generate n_samples completions, return mean pairwise CD between them.
    Measures how diverse (or deterministic) the diffusion output is.
    """
    samples = []
    for _ in range(n_samples):
        pts = complete_frame(model, partial_dict, num_steps, eta)
        samples.append(torch.from_numpy(pts).float().to(device))

    pairwise_cds = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            cd = chamfer_distance(samples[i], samples[j], chunk_size=512).item()
            pairwise_cds.append(cd)

    return float(np.mean(pairwise_cds)) if pairwise_cds else 0.0


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _mean_std(records: list, key: str) -> tuple:
    vals = [r[key] for r in records
            if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def build_summary(results: list, args, ckpt: dict) -> dict:
    keys = [
        "cd", "cd_sq",
        "jsd",
        "f_score@0.1", "f_score@0.2",
        "precision@0.1", "precision@0.2",
        "recall@0.1", "recall@0.2",
        "iou@0.1", "iou@0.2",
        "hausdorff_95",
        "time",
        "diversity_cd",
    ]
    agg = {}
    for k in keys:
        m, s = _mean_std(results, k)
        agg[k + "_mean"] = m
        agg[k + "_std"] = s

    return {
        "model": "latent_diffusion",
        "checkpoint": args.ckpt,
        "epoch": ckpt.get("epoch", "?"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "num_samples": len(results),
        "num_steps": args.num_steps,
        "eta": args.eta,
        "n_samples_per_frame": args.n_samples,
        "aggregated": agg,
        "per_frame": results,
    }


def print_summary(summary: dict, baseline: dict | None = None):
    agg = summary["aggregated"]
    print()
    print("=" * 72)
    print(f"  Latent Diffusion Evaluation Summary  —  {summary['num_samples']} frames")
    print(f"  Checkpoint : {summary['checkpoint']}")
    print(f"  Epoch      : {summary['epoch']}   Val loss: {summary['best_val_loss']}")
    print(f"  DDIM steps : {summary['num_steps']}   eta={summary['eta']}")
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

    print("\n  --- Chamfer distance ---")
    row("CD linear (m)",        "cd",            ".4f")
    row("CD squared (m²)",      "cd_sq",         ".6f")

    print("\n  --- Distribution similarity ---")
    row("JSD-BEV (0.5 m vox)", "jsd",            ".4f")

    print("\n  --- Point coverage ---")
    row("F-score @ 0.1 m",      "f_score@0.1",   ".4f")
    row("F-score @ 0.2 m",      "f_score@0.2",   ".4f")
    row("Precision @ 0.1 m",    "precision@0.1", ".4f")
    row("Recall @ 0.1 m",       "recall@0.1",    ".4f")

    print("\n  --- Voxel occupancy ---")
    row("Voxel IoU @ 0.1 m",    "iou@0.1",       ".4f")
    row("Voxel IoU @ 0.2 m",    "iou@0.2",       ".4f")

    print("\n  --- Worst-case geometry ---")
    row("Hausdorff-95 (m)",      "hausdorff_95",  ".4f")

    if summary["n_samples_per_frame"] > 1:
        print("\n  --- Diffusion stochastic diversity ---")
        row("Mean pairwise CD (m)",  "diversity_cd",  ".4f")

    row("\n  Avg inference time (s)", "time",        ".2f")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Steps sweep
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_steps_sweep(
    model: SceneCompletionLatentDiffusion,
    partial_dict: dict,
    gt_raw: np.ndarray,
    scan_center: np.ndarray,
    steps_list: list,
    device: torch.device,
) -> list:
    """Run the same frame with different DDIM step counts, return list of metric dicts."""
    sweep_results = []
    for n in steps_list:
        recon_shifted = complete_frame(model, partial_dict, num_steps=n, eta=0.0)
        recon_original = recon_shifted + scan_center
        ral = compute_all_metrics(recon_original, gt_raw)
        ral["num_steps"] = n
        sweep_results.append(ral)
        print(f"    steps={n:3d}: CD={ral['cd']:.4f}  F@0.1={ral['f_score@0.1']:.3f}")
    return sweep_results


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate latent diffusion for scene completion on SemanticKITTI"
    )
    p.add_argument("--data_path", type=str,
                   default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"))
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/latent_diffusion/best_latent_diffusion.pth")
    p.add_argument("--output_dir", type=str, default="evaluation_latent_diffusion")
    p.add_argument("--sequence", type=str, default="08")
    p.add_argument("--num_samples", type=int, default=50,
                   help="Number of frames to evaluate")
    p.add_argument("--point_max_partial", type=int, default=20000,
                   help="Max partial scan points fed to Sonata (0 = no limit)")
    p.add_argument("--voxel_size", type=float, default=0.05,
                   help="Sonata voxel size for grid_coord (must match training)")
    p.add_argument("--num_steps", type=int, default=None,
                   help="DDIM denoising steps (default: from checkpoint)")
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM eta: 0.0=deterministic, 1.0=DDPM stochastic")

    # Stochastic diversity
    p.add_argument("--n_samples", type=int, default=1,
                   help="Completions per frame for diversity analysis (>1 enables)")

    # DDIM steps sweep (runs first frame only)
    p.add_argument("--steps_sweep", type=int, nargs="+", default=None,
                   metavar="N",
                   help="Run a DDIM steps ablation on first frame (e.g. 10 20 50 100)")

    # Baseline comparison
    p.add_argument("--baseline_json", type=str, default=None,
                   help="metrics.json from a previous eval for side-by-side Δ display")

    # GT dir
    p.add_argument("--gt_subdir", type=str, default="ground_truth")

    # Device
    p.add_argument("--device", type=str, default=None,
                   help="'cuda', 'cuda:1', 'cpu' — auto-detects if omitted")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint and model ----
    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"  Epoch: {ckpt.get('epoch', '?')}   Val loss: {ckpt.get('best_val_loss', '?')}")

    num_steps = args.num_steps or ckpt.get("denoising_steps", 50)
    print(f"  DDIM steps: {num_steps}   eta={args.eta}")

    model = build_model_from_checkpoint(ckpt, device, denoising_steps=num_steps)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ---- Baseline for comparison ----
    baseline = None
    if args.baseline_json and os.path.exists(args.baseline_json):
        with open(args.baseline_json) as f:
            baseline = json.load(f)
        print(f"  Baseline: {args.baseline_json}")
    elif args.baseline_json:
        print(f"  Warning: baseline_json not found: {args.baseline_json}")

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

    # ---- Optional: DDIM steps sweep on first available frame ----
    sweep_results = None
    if args.steps_sweep:
        for fid in sample_frames:
            gt_path = os.path.join(gt_dir, f"{fid}.npz")
            if not os.path.exists(gt_path):
                continue
            lidar_raw = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
            gt_raw = load_gt(gt_path)
            scan_center = lidar_raw.mean(axis=0)
            partial_shifted = lidar_raw - scan_center
            if args.point_max_partial > 0:
                partial_shifted = subsample(partial_shifted, args.point_max_partial)
            partial_dict = build_partial_scan_dict(partial_shifted, device, args.voxel_size)

            print(f"\n  DDIM steps sweep on frame {fid}:")
            sweep_results = run_steps_sweep(
                model, partial_dict, gt_raw, scan_center,
                args.steps_sweep, device,
            )
            steps_arr = [r["num_steps"] for r in sweep_results]
            cd_arr = [r["cd"] for r in sweep_results]
            f_arr = [r["f_score@0.1"] for r in sweep_results]
            steps_quality_plot(
                steps_arr, cd_arr, f_arr,
                os.path.join(args.output_dir, "steps_sweep.png"),
            )
            with open(os.path.join(args.output_dir, "steps_sweep.json"), "w") as f:
                json.dump(sweep_results, f, indent=2)
            print(f"  Steps sweep saved → {args.output_dir}/steps_sweep.png")
            break

    # ---- Main evaluation loop ----
    results = []
    diversity_cds = []

    for i, fid in enumerate(sample_frames):
        lidar_path = os.path.join(vel_dir, f"{fid}.bin")
        gt_path = os.path.join(gt_dir, f"{fid}.npz")

        if not os.path.exists(gt_path):
            print(f"  [{i + 1}/{len(sample_frames)}] {fid}: no GT, skipping")
            continue

        lidar_raw = load_bin(lidar_path)
        gt_raw = load_gt(gt_path)

        frame_result, recon_original = evaluate_frame(
            model, lidar_raw, gt_raw, args, device
        )
        frame_result["frame"] = fid

        # Stochastic diversity across N completions
        div_cd = 0.0
        if args.n_samples > 1:
            scan_center = lidar_raw.mean(axis=0)
            partial_shifted = lidar_raw - scan_center
            if args.point_max_partial > 0:
                partial_shifted = subsample(partial_shifted, args.point_max_partial)
            partial_dict = build_partial_scan_dict(partial_shifted, device, args.voxel_size)
            div_cd = stochastic_diversity(
                model, partial_dict, args.n_samples, num_steps, args.eta, device,
            )
            diversity_cds.append(div_cd)
        frame_result["diversity_cd"] = div_cd

        results.append(frame_result)

        div_str = f"  div_CD={div_cd:.4f}" if args.n_samples > 1 else ""
        print(
            f"  [{i + 1:3d}/{len(sample_frames)}] {fid} | "
            f"CD={frame_result['cd']:.4f}  "
            f"F@0.1={frame_result['f_score@0.1']:.3f}  "
            f"JSD={frame_result['jsd']:.4f}  "
            f"H95={frame_result['hausdorff_95']:.3f}  "
            f"t={frame_result['time']:.1f}s"
            f"{div_str}"
        )

        # BEV visualisation for first 20 frames
        if i < 20:
            bev_plot(
                {
                    "Partial Input": lidar_raw,
                    "Diffusion Completion": recon_original,
                    "GT Complete": gt_raw,
                },
                f"Latent Diffusion | {fid} | CD={frame_result['cd']:.3f}  "
                f"F@0.1={frame_result['f_score@0.1']:.3f}  steps={num_steps}",
                os.path.join(args.output_dir, f"bev_{fid}.png"),
            )

    # Diversity histogram
    if diversity_cds:
        diversity_plot(
            diversity_cds,
            os.path.join(args.output_dir, "diversity_histogram.png"),
        )

    # ---- Summary ----
    summary = build_summary(results, args, ckpt)
    summary["num_steps"] = num_steps
    print_summary(summary, baseline=baseline)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved  → {metrics_path}")
    print(f"BEV plots      → {args.output_dir}/bev_*.png")
    if diversity_cds:
        print(f"Diversity plot → {args.output_dir}/diversity_histogram.png")


if __name__ == "__main__":
    main()
