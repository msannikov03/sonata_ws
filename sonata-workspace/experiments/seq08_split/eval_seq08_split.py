#!/usr/bin/env python3
"""
Evaluation script for seq 08 split experiment.

Loads a trained checkpoint, runs single-step x0 at t=200 on the last 1071
frames of seq 08, computes Chamfer distance against raw GT.

Reports mean, std, median CD.
"""

import os, sys, argparse, time, json
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance


# ---------------------------------------------------------------------------
# Model builder (identical to train_seq08_split.py)
# ---------------------------------------------------------------------------

def build_model():
    encoder = SonataEncoder(
        pretrained="facebook/sonata",
        freeze=True,
        enable_flash=False,
        feature_levels=[0],
    )
    cond = ConditionalFeatureExtractor(
        encoder,
        feature_levels=[0],
        fusion_type="concat",
    )
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=cond,
        num_timesteps=1000,
        schedule="cosine",
        denoising_steps=50,
    )
    return model


# ---------------------------------------------------------------------------
# Helpers (from evaluate.py)
# ---------------------------------------------------------------------------

def prepare_scan(pts_raw, device="cuda", max_points=20000, voxel_size=0.05):
    """Voxelize, subsample, build Sonata input dict. Returns (dict, center)."""
    center = pts_raw.mean(axis=0)
    pts = pts_raw - center

    vc = np.floor(pts / voxel_size).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    pts = pts[idx]

    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]

    z = pts[:, 2]
    zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1)

    return {
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(pts.shape[0], 3).float().to(device),
        "batch": torch.zeros(pts.shape[0], dtype=torch.long).to(device),
    }, center


def prepare_gt_target(gt_raw, center, device="cuda",
                      max_points=20000, voxel_size=0.05):
    """Center GT on the same center as input, voxelize, subsample."""
    gt = gt_raw - center
    vc = np.floor(gt / voxel_size).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    gt = gt[idx]
    if gt.shape[0] > max_points:
        sel = np.random.choice(gt.shape[0], max_points, replace=False)
        gt = gt[sel]
    return torch.from_numpy(gt).float().to(device)


@torch.no_grad()
def run_completion_x0(model, point_dict, target_coords=None, t_val=200):
    """Single-step x0 prediction at t=t_val."""
    model.eval()
    device = point_dict["coord"].device

    # Conditioning
    cond_features, _ = model.condition_extractor(point_dict)

    if target_coords is not None:
        coords = target_coords
        cond_features = knn_interpolate(cond_features, point_dict["coord"], coords)
    else:
        coords = point_dict["coord"]

    model.scheduler._to_device(device)

    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * coords + som * noise

    pred_noise = model.denoiser(noisy, coords, t_tensor, {"features": cond_features})
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0.cpu().numpy()


def compute_cd(pred, gt, max_pts=10000):
    """Symmetric Chamfer distance between two point clouds."""
    if pred.shape[0] > max_pts:
        pred = pred[np.random.choice(pred.shape[0], max_pts, replace=False)]
    if gt.shape[0] > max_pts:
        gt = gt[np.random.choice(gt.shape[0], max_pts, replace=False)]
    cd = chamfer_distance(
        torch.from_numpy(pred).float().cuda(),
        torch.from_numpy(gt).float().cuda(),
        chunk_size=512,
    )
    return cd.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Seq 08 split evaluation")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained checkpoint (.pth)")
    p.add_argument("--npz_dir", type=str, required=True,
                   help="Directory with pre-voxelized .npz frames")
    p.add_argument("--input_type", type=str, default="lidar",
                   choices=["lidar", "da2"],
                   help="Which input modality was used for training")
    p.add_argument("--data_path", type=str, default=None,
                   help="Path to SemanticKITTI dataset root (for raw GT). "
                        "If not given, uses gt_coords from npz.")
    p.add_argument("--output_dir", type=str, default="eval_results/seq08_split")
    p.add_argument("--max_points", type=int, default=20000)
    p.add_argument("--t_val", type=int, default=200,
                   help="Noise timestep for single-step x0 prediction")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    # Load model
    print("Building model...")
    model = build_model().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {epoch})")

    # Validation frame indices (last 1071 of seq 08)
    val_indices = list(range(3000, 4071))

    # Check which frames exist
    frames = []
    for idx in val_indices:
        npz_path = os.path.join(args.npz_dir, f"{idx:06d}.npz")
        if os.path.exists(npz_path):
            frames.append((idx, npz_path))
    print(f"Evaluating {len(frames)} val frames\n")

    # Optional: raw GT from SemanticKITTI for absolute CD
    use_raw_gt = args.data_path is not None
    if use_raw_gt:
        gt_dir = os.path.join(args.data_path, "ground_truth", "08")
        vel_dir = os.path.join(args.data_path, "sequences", "08", "velodyne")
        print(f"Using raw GT from {gt_dir}")

    cds = []
    results = []

    for idx, npz_path in tqdm(frames, desc="Evaluating"):
        data = np.load(npz_path)

        if args.input_type == "lidar":
            input_key = "lidar_coords"
            center_key = "lidar_center"
            gt_key = "gt_coords_lidar"
        else:
            input_key = "da2_coords"
            center_key = "da2_center"
            gt_key = "gt_coords_da2"

        input_coords = data[input_key].astype(np.float32)
        input_center = data[center_key].astype(np.float32)
        gt_centered = data[gt_key].astype(np.float32)

        # Build Sonata input (already centered in npz)
        pts = input_coords.copy()
        if pts.shape[0] > args.max_points:
            sel = np.random.choice(pts.shape[0], args.max_points, replace=False)
            pts = pts[sel]

        z = pts[:, 2]
        zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
        colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1)

        point_dict = {
            "coord": torch.from_numpy(pts).float().to(device),
            "color": torch.from_numpy(colors).float().to(device),
            "normal": torch.zeros(pts.shape[0], 3).float().to(device),
            "batch": torch.zeros(pts.shape[0], dtype=torch.long).to(device),
        }

        # GT target coords (centered, voxelized, subsampled -- from npz)
        gt_sub = gt_centered.copy()
        if gt_sub.shape[0] > args.max_points:
            sel = np.random.choice(gt_sub.shape[0], args.max_points, replace=False)
            gt_sub = gt_sub[sel]
        gt_target = torch.from_numpy(gt_sub).float().to(device)

        # Run single-step x0
        pred_x0 = run_completion_x0(model, point_dict, target_coords=gt_target,
                                     t_val=args.t_val)

        # Un-center for CD against raw GT
        pred_world = pred_x0 + input_center

        if use_raw_gt:
            # Load raw GT
            raw_gt_path = os.path.join(gt_dir, f"{idx:06d}.npz")
            if os.path.exists(raw_gt_path):
                raw_gt = np.load(raw_gt_path)["points"]
            else:
                # Fall back to un-centered npz GT
                raw_gt = gt_centered + input_center
        else:
            # Use GT from npz (un-centered)
            raw_gt = gt_centered + input_center

        cd = compute_cd(pred_world, raw_gt)
        cds.append(cd)
        results.append({"frame": f"{idx:06d}", "cd": cd})

    # Summary
    cds = np.array(cds)
    print(f"\n{'='*60}")
    print(f"Results: {args.input_type} model, {len(cds)} frames")
    print(f"  Mean CD:   {cds.mean():.4f}")
    print(f"  Std CD:    {cds.std():.4f}")
    print(f"  Median CD: {np.median(cds):.4f}")
    print(f"  Min CD:    {cds.min():.4f}")
    print(f"  Max CD:    {cds.max():.4f}")
    print(f"{'='*60}")

    # Save results
    summary = {
        "checkpoint": args.checkpoint,
        "input_type": args.input_type,
        "num_frames": len(cds),
        "mean_cd": float(cds.mean()),
        "std_cd": float(cds.std()),
        "median_cd": float(np.median(cds)),
        "min_cd": float(cds.min()),
        "max_cd": float(cds.max()),
        "t_val": args.t_val,
        "per_frame": results,
    }
    out_path = os.path.join(args.output_dir, f"eval_{args.input_type}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
