#!/usr/bin/env python3
"""
Evaluate teacher model with LiDAR + DA2 fusion input.
Two modes:
  1. Naive fusion: concatenate all LiDAR + DA2 points
  2. Gap-filling: only add DA2 points that are >0.5m from any LiDAR point
Both inputs are centered on the FUSED cloud's mean. GT is centered on the same mean.
"""
import os, sys, torch, numpy as np, argparse, time, json
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion
from models.refinement_net import chamfer_distance


def build_model(device="cuda"):
    encoder = SonataEncoder(
        pretrained="facebook/sonata", freeze=True,
        enable_flash=False, feature_levels=[0]
    )
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    model = SceneCompletionDiffusion(
        encoder=encoder, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50
    )
    return model.to(device)


def load_ckpt(model, path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {path} (epoch {epoch})")
    return model


def prepare_scan(pts_raw, device="cuda", max_points=20000, voxel_size=0.05):
    """Center, voxelize, subsample. Returns point dict and center."""
    center = pts_raw.mean(axis=0)
    pts = pts_raw - center
    # Voxel downsample
    vc = np.floor(pts / voxel_size).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    pts = pts[idx]
    # Subsample if too many
    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]
    # Height-based colors
    z = pts[:, 2]
    zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1)
    return {
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(pts.shape[0], 3).float().to(device),
        "batch": torch.zeros(pts.shape[0], dtype=torch.long).to(device),
    }, center


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path):
    return np.load(path)["points"]


def gap_fill(lidar, da2, gap_threshold=0.5):
    """Return DA2 points that are >gap_threshold from any LiDAR point."""
    tree = cKDTree(lidar)
    dists, _ = tree.query(da2, k=1)
    mask = dists > gap_threshold
    return da2[mask]


def bev_plot(pts_dict, title, save_path):
    n = len(pts_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, pts) in zip(axes, pts_dict.items()):
        if pts is not None and len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.1, cmap="viridis", vmin=-2, vmax=4)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_title(name)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def run_completion_x0(model, point_dict, target_coords=None):
    model.eval()
    device = point_dict['coord'].device
    cond_features, _ = model.condition_extractor(point_dict)
    if target_coords is not None:
        coords = target_coords
        from models.diffusion_module import knn_interpolate
        cond_features = knn_interpolate(cond_features, point_dict['coord'], coords)
    else:
        coords = point_dict['coord']
    model.scheduler._to_device(device)
    t_val = 200
    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * coords + som * noise
    pred_noise = model.denoiser(noisy, coords, t_tensor, {'features': cond_features})
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0.cpu().numpy()


def compute_cd(pred, gt, max_pts=10000):
    if pred.shape[0] > max_pts:
        pred = pred[np.random.choice(pred.shape[0], max_pts, replace=False)]
    if gt.shape[0] > max_pts:
        gt = gt[np.random.choice(gt.shape[0], max_pts, replace=False)]
    cd = chamfer_distance(
        torch.from_numpy(pred).float().cuda(),
        torch.from_numpy(gt).float().cuda(),
        chunk_size=512
    )
    return cd.item()


def run_one_frame(model, fused_pts, gt_raw, device="cuda", max_points=20000, voxel_size=0.05):
    """
    Run completion on fused input. Centers both input and GT on fused mean.
    Returns (completion_global, cd_value).
    """
    # Prepare input: center on fused mean
    input_dict, center = prepare_scan(fused_pts, device, max_points, voxel_size)

    # Prepare GT: center on SAME fused mean (critical!)
    gt_centered = gt_raw - center
    vc = np.floor(gt_centered / voxel_size).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    gt_sub = gt_centered[idx]
    if gt_sub.shape[0] > max_points:
        sel = np.random.choice(gt_sub.shape[0], max_points, replace=False)
        gt_sub = gt_sub[sel]
    gt_target = torch.from_numpy(gt_sub).float().to(device)

    # Run teacher
    comp = run_completion_x0(model, input_dict, target_coords=gt_target)
    comp_global = comp + center  # back to global coords

    # CD against raw GT
    cd_val = compute_cd(comp_global, gt_raw)
    return comp_global, cd_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--lidar_dir", type=str, required=True)
    parser.add_argument("--da2_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sequence", type=str, default="08")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="evaluation_fusion")
    parser.add_argument("--gap_threshold", type=float, default=0.5,
                        help="Gap-filling: min distance from LiDAR to keep DA2 point")
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    print("=" * 70)
    print("LiDAR + DA2 FUSION EVALUATION")
    print("=" * 70)
    print(f"  LiDAR dir:      {args.lidar_dir}")
    print(f"  DA2 dir:        {args.da2_dir}")
    print(f"  GT dir:         {args.data_path}/ground_truth/{args.sequence}")
    print(f"  Checkpoint:     {args.teacher_ckpt}")
    print(f"  Gap threshold:  {args.gap_threshold}m")
    print(f"  Max points:     {args.max_points}")
    print(f"  Num samples:    {args.num_samples}")
    print()

    # Build and load model
    print("Building teacher model...")
    model = load_ckpt(build_model(device), args.teacher_ckpt, device)

    # Get frame list
    gt_dir = os.path.join(args.data_path, "ground_truth", args.sequence)
    frames = sorted([f.replace(".bin", "") for f in os.listdir(args.lidar_dir) if f.endswith(".bin")])
    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][:args.num_samples]
    print(f"Evaluating {len(sample_frames)} frames\n")

    results = []
    for i, fid in enumerate(sample_frames):
        print(f"--- Frame {fid} ({i+1}/{len(sample_frames)}) ---")

        lidar_path = os.path.join(args.lidar_dir, fid + ".bin")
        da2_path = os.path.join(args.da2_dir, fid + ".bin")
        gt_path = os.path.join(gt_dir, fid + ".npz")

        if not os.path.exists(da2_path):
            print(f"  SKIP: no DA2 at {da2_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"  SKIP: no GT at {gt_path}")
            continue

        lidar = load_bin(lidar_path)
        da2 = load_bin(da2_path)
        gt_raw = load_gt(gt_path)

        # --- Mode 1: Naive fusion (concatenate everything) ---
        fused_naive = np.concatenate([lidar[:, :3], da2[:, :3]], axis=0)
        comp_naive, cd_naive = run_one_frame(model, fused_naive, gt_raw, device, args.max_points)
        print(f"  Naive fusion:    {lidar.shape[0]} + {da2.shape[0]} = {fused_naive.shape[0]} pts -> CD={cd_naive:.4f}")

        # --- Mode 2: Gap-filling fusion ---
        da2_gap = gap_fill(lidar[:, :3], da2[:, :3], args.gap_threshold)
        fused_gap = np.concatenate([lidar[:, :3], da2_gap], axis=0)
        comp_gap, cd_gap = run_one_frame(model, fused_gap, gt_raw, device, args.max_points)
        print(f"  Gap-fill fusion: {lidar.shape[0]} + {da2_gap.shape[0]} = {fused_gap.shape[0]} pts -> CD={cd_gap:.4f}")

        # --- Mode 3: LiDAR-only baseline (for direct comparison) ---
        comp_lidar, cd_lidar = run_one_frame(model, lidar[:, :3], gt_raw, device, args.max_points)
        print(f"  LiDAR-only:      {lidar.shape[0]} pts -> CD={cd_lidar:.4f}")

        results.append({
            "frame": fid,
            "n_lidar": int(lidar.shape[0]),
            "n_da2": int(da2.shape[0]),
            "n_da2_gap": int(da2_gap.shape[0]),
            "cd_naive_fusion": cd_naive,
            "cd_gap_fusion": cd_gap,
            "cd_lidar_only": cd_lidar,
        })

        # BEV viz for first 5 frames
        if i < 5:
            bev_plot(
                {"LiDAR": lidar, "DA2": da2, "Naive Fused": fused_naive,
                 "Gap-Filled": fused_gap, "GT": gt_raw},
                f"Fusion Inputs - Frame {fid}",
                os.path.join(args.output_dir, f"input_bev_{fid}.png")
            )
            bev_plot(
                {"LiDAR-only": comp_lidar, "Naive Fusion": comp_naive,
                 "Gap-Fill Fusion": comp_gap, "GT": gt_raw},
                f"Completions - Frame {fid}",
                os.path.join(args.output_dir, f"completion_bev_{fid}.png")
            )

    # ===== Summary =====
    print("\n" + "=" * 80)
    print(f"{'Frame':<10} {'LiDAR-only':<14} {'Naive Fusion':<14} {'Gap-Fill':<14} {'N_LiDAR':<10} {'N_DA2':<10} {'N_DA2_gap':<10}")
    print("=" * 80)
    for r in results:
        print(f"{r['frame']:<10} {r['cd_lidar_only']:<14.4f} {r['cd_naive_fusion']:<14.4f} {r['cd_gap_fusion']:<14.4f} {r['n_lidar']:<10} {r['n_da2']:<10} {r['n_da2_gap']:<10}")

    cd_lidar = [r["cd_lidar_only"] for r in results]
    cd_naive = [r["cd_naive_fusion"] for r in results]
    cd_gap = [r["cd_gap_fusion"] for r in results]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  LiDAR-only:      CD = {np.mean(cd_lidar):.4f} +/- {np.std(cd_lidar):.4f}")
    print(f"  Naive fusion:    CD = {np.mean(cd_naive):.4f} +/- {np.std(cd_naive):.4f}")
    print(f"  Gap-fill fusion: CD = {np.mean(cd_gap):.4f} +/- {np.std(cd_gap):.4f}")
    print()
    print(f"  Naive vs LiDAR:    {(np.mean(cd_naive) - np.mean(cd_lidar))/np.mean(cd_lidar)*100:+.2f}%")
    print(f"  Gap-fill vs LiDAR: {(np.mean(cd_gap) - np.mean(cd_lidar))/np.mean(cd_lidar)*100:+.2f}%")
    print(f"  N frames: {len(results)}")

    # Avg DA2 gap-fill retention
    avg_gap_pct = np.mean([r["n_da2_gap"] / r["n_da2"] * 100 for r in results])
    print(f"  Avg DA2 gap-fill retention: {avg_gap_pct:.1f}% of DA2 points kept")

    # Save results
    summary = {
        "lidar_only": {"mean": float(np.mean(cd_lidar)), "std": float(np.std(cd_lidar))},
        "naive_fusion": {"mean": float(np.mean(cd_naive)), "std": float(np.std(cd_naive))},
        "gap_fill_fusion": {"mean": float(np.mean(cd_gap)), "std": float(np.std(cd_gap))},
        "gap_threshold": args.gap_threshold,
        "n_frames": len(results),
        "per_frame": results,
    }
    with open(os.path.join(args.output_dir, "fusion_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/fusion_metrics.json")


if __name__ == "__main__":
    main()
