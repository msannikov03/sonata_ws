#!/usr/bin/env python3
"""
Sensor Degradation Evaluation for 3D Scene Completion

Measures teacher (LiDAR-based) and student (RGB/DA2-based) model robustness
under various sensor degradation conditions.

Key insight: the student uses monocular RGB (via Depth Anything v2) at inference,
so LiDAR degradations do NOT affect it. At some degradation threshold, the
student surpasses the degraded teacher — this is the crossover point.

Degradation types:
  1. LiDAR beam dropout (random point removal)
  2. LiDAR Gaussian noise (coordinate perturbation)
  3. Angular sector occlusion (azimuth-based blockage)
  4. Depth estimation noise (student-only, pseudo-depth perturbation)

Usage:
    python sensor_degradation_eval.py \
        --data_path /home/anywherevla/sonata_ws/dataset/sonata_depth_pro \
        --teacher_ckpt checkpoints/diffusion_depthpro/best_model.pth \
        --student_ckpt checkpoints/distill_task_only/best_model.pth \
        --output_dir evaluation_degradation \
        --num_samples 50
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add workspace to path
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(WORKSPACE, "models")):
    # Running from /tmp — point to actual workspace
    WORKSPACE = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
sys.path.insert(0, WORKSPACE)

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance


# =============================================================================
# Model utilities (from evaluate.py)
# =============================================================================

def build_model(device="cuda"):
    encoder = SonataEncoder(
        pretrained="facebook/sonata", freeze=True,
        enable_flash=False, feature_levels=[0]
    )
    cond = ConditionalFeatureExtractor(
        encoder, feature_levels=[0], fusion_type="concat"
    )
    model = SceneCompletionDiffusion(
        encoder=encoder, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50
    )
    return model.to(device)


def load_ckpt(model, path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  loaded {path} (epoch {ckpt.get('epoch', '?')})")
    return model


def prepare_scan(pts_raw, device="cuda", max_points=20000, voxel_size=0.05):
    """Prepare a raw point cloud for model input. Returns (point_dict, center)."""
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


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path):
    return np.load(path)["points"]


@torch.no_grad()
def run_completion_x0(model, point_dict, target_coords=None):
    """Single-step x0 prediction at t=200 (matches evaluate.py)."""
    model.eval()
    device = point_dict["coord"].device

    cond_features, _ = model.condition_extractor(point_dict)

    if target_coords is not None:
        coords = target_coords
        cond_features = knn_interpolate(cond_features, point_dict["coord"], coords)
    else:
        coords = point_dict["coord"]

    model.scheduler._to_device(device)

    t_val = 200
    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * coords + som * noise

    pred_noise = model.denoiser(noisy, coords, t_tensor, {"features": cond_features})
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
        chunk_size=512,
    )
    return cd.item()


# =============================================================================
# Degradation functions
# =============================================================================

def degrade_beam_dropout(points, drop_rate):
    """Randomly remove drop_rate fraction of points (simulates beam failure)."""
    if drop_rate <= 0.0:
        return points.copy()
    n = points.shape[0]
    mask = np.random.rand(n) > drop_rate
    remaining = points[mask]
    # Ensure at least 100 points survive
    if remaining.shape[0] < 100:
        keep = np.random.choice(n, 100, replace=True)
        remaining = points[keep]
    return remaining


def degrade_gaussian_noise(points, sigma):
    """Add Gaussian noise to point coordinates."""
    if sigma <= 0.0:
        return points.copy()
    return points + np.random.randn(*points.shape).astype(np.float32) * sigma


def degrade_sector_occlusion(points, angle_deg):
    """Remove all points within an azimuth sector of given angular width.

    The occluded sector starts at a random azimuth and spans angle_deg degrees.
    Simulates physical blockage (e.g., mud, snow on sensor, nearby object).
    """
    if angle_deg <= 0.0:
        return points.copy()

    azimuth = np.arctan2(points[:, 1], points[:, 0])  # [-pi, pi]
    # Random sector start
    sector_start = np.random.uniform(-np.pi, np.pi)
    half_angle = np.deg2rad(angle_deg) / 2.0

    # Angular distance (wrapped)
    diff = np.abs(azimuth - sector_start)
    diff = np.minimum(diff, 2 * np.pi - diff)
    mask = diff > half_angle

    remaining = points[mask]
    if remaining.shape[0] < 100:
        keep = np.random.choice(points.shape[0], 100, replace=True)
        remaining = points[keep]
    return remaining


def degrade_depth_noise(points, mae_target):
    """Add noise to pseudo-depth point cloud along the radial direction.

    For each point, compute its range from origin and perturb it by a
    noise sampled so that the expected absolute error is ~mae_target.
    This simulates depth estimation errors from monocular models.
    """
    if mae_target <= 0.0:
        return points.copy()

    pts = points.copy()
    ranges = np.linalg.norm(pts, axis=2 if pts.ndim == 3 else 1, keepdims=True)
    ranges = np.clip(ranges, 1e-3, None)
    directions = pts / ranges

    # Laplace noise with mean=0 and b=mae_target/sqrt(2) gives E[|noise|]=mae_target
    # But simpler: normal with std such that E[|N(0,std)|] = mae_target
    # E[|N(0,s)|] = s * sqrt(2/pi), so s = mae_target * sqrt(pi/2)
    sigma = mae_target * np.sqrt(np.pi / 2.0)
    range_noise = np.random.randn(*ranges.shape).astype(np.float32) * sigma
    pts = pts + directions * range_noise
    return pts


# =============================================================================
# Degradation configs
# =============================================================================

DEGRADATION_CONFIGS = {
    "beam_dropout": {
        "label": "LiDAR Beam Dropout",
        "xlabel": "Drop Rate (%)",
        "levels": [0.0, 0.25, 0.50, 0.75, 0.90],
        "level_labels": ["0%", "25%", "50%", "75%", "90%"],
        "fn": degrade_beam_dropout,
        "applies_to": "teacher",  # only degrades teacher input
    },
    "gaussian_noise": {
        "label": "LiDAR Gaussian Noise",
        "xlabel": "Noise σ (m)",
        "levels": [0.0, 0.05, 0.10, 0.20, 0.50],
        "level_labels": ["0.0", "0.05", "0.10", "0.20", "0.50"],
        "fn": degrade_gaussian_noise,
        "applies_to": "teacher",
    },
    "sector_occlusion": {
        "label": "Angular Sector Occlusion",
        "xlabel": "Occluded Sector (°)",
        "levels": [0.0, 30.0, 60.0, 90.0, 180.0],
        "level_labels": ["0°", "30°", "60°", "90°", "180°"],
        "fn": degrade_sector_occlusion,
        "applies_to": "teacher",
    },
    "depth_noise": {
        "label": "Depth Estimation Noise",
        "xlabel": "MAE (m)",
        "levels": [0.0, 0.3, 0.6, 1.0, 2.0],
        "level_labels": ["0.0", "0.3", "0.6", "1.0", "2.0"],
        "fn": degrade_depth_noise,
        "applies_to": "student",
    },
}


# =============================================================================
# Main evaluation loop
# =============================================================================

def evaluate_degradation(args):
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load models ---
    print("Building teacher model...")
    teacher = load_ckpt(build_model(device), args.teacher_ckpt, device)

    print("Building student model...")
    student = load_ckpt(build_model(device), args.student_ckpt, device)

    # --- Discover frames ---
    seq = args.sequence
    vel_dir = os.path.join(args.data_path, "sequences", seq, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", seq)
    da2_dir = os.path.join(args.data_path, "da2_output", "pointclouds", "sequences", seq)

    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])
    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][: args.num_samples]
    print(f"Evaluating {len(sample_frames)} frames from seq {seq}")
    print(f"  vel_dir: {vel_dir}")
    print(f"  gt_dir:  {gt_dir}")
    print(f"  da2_dir: {da2_dir}")

    # --- Pre-load all frame data ---
    print("\nPre-loading frame data...")
    frame_data = []
    skipped = 0
    for fid in sample_frames:
        lidar_path = os.path.join(vel_dir, f"{fid}.bin")
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        da2_path = os.path.join(da2_dir, f"{fid}.bin")

        if not os.path.exists(gt_path):
            skipped += 1
            continue
        if not os.path.exists(da2_path):
            skipped += 1
            continue

        lidar = load_bin(lidar_path)
        gt = load_gt(gt_path)
        da2 = load_bin(da2_path)
        frame_data.append({"fid": fid, "lidar": lidar, "gt": gt, "da2": da2})

    print(f"  loaded {len(frame_data)} frames ({skipped} skipped)")
    if len(frame_data) == 0:
        print("ERROR: no valid frames found. Check paths.")
        return

    # --- Pre-compute GT targets (same for all degradations) ---
    print("Pre-computing GT targets...")
    gt_targets = []
    for fd in frame_data:
        gt_center = fd["lidar"].mean(axis=0)
        gt_centered = fd["gt"] - gt_center
        vc = np.floor(gt_centered / 0.05).astype(np.int32)
        _, idx = np.unique(vc, axis=0, return_index=True)
        gt_sub = gt_centered[idx]
        if gt_sub.shape[0] > 20000:
            sel = np.random.choice(gt_sub.shape[0], 20000, replace=False)
            gt_sub = gt_sub[sel]
        gt_targets.append(torch.from_numpy(gt_sub).float().to(device))

    # --- Compute baseline (clean) CDs once ---
    print("\nComputing clean baselines...")
    teacher_baseline_cds = []
    student_baseline_cds = []

    for i, fd in enumerate(frame_data):
        # Teacher on clean LiDAR
        t_input, t_center = prepare_scan(fd["lidar"], device)
        t_comp = run_completion_x0(teacher, t_input, target_coords=gt_targets[i])
        t_comp += t_center
        tcd = compute_cd(t_comp, fd["gt"])
        teacher_baseline_cds.append(tcd)

        # Student on clean DA2
        s_input, s_center = prepare_scan(fd["da2"], device)
        s_comp = run_completion_x0(student, s_input, target_coords=gt_targets[i])
        s_comp += s_center
        scd = compute_cd(s_comp, fd["gt"])
        student_baseline_cds.append(scd)

        if (i + 1) % 10 == 0:
            print(f"  baseline: {i+1}/{len(frame_data)} frames done")

    t_base_mean = np.mean(teacher_baseline_cds)
    t_base_std = np.std(teacher_baseline_cds)
    s_base_mean = np.mean(student_baseline_cds)
    s_base_std = np.std(student_baseline_cds)
    print(f"  teacher clean: {t_base_mean:.4f} +/- {t_base_std:.4f}")
    print(f"  student clean: {s_base_mean:.4f} +/- {s_base_std:.4f}")

    # --- Run degradation experiments ---
    all_results = {}

    for deg_name, deg_cfg in DEGRADATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Degradation: {deg_cfg['label']}")
        print(f"{'='*60}")

        results_for_deg = {
            "label": deg_cfg["label"],
            "xlabel": deg_cfg["xlabel"],
            "levels": deg_cfg["levels"],
            "level_labels": deg_cfg["level_labels"],
            "applies_to": deg_cfg["applies_to"],
            "teacher_cd_mean": [],
            "teacher_cd_std": [],
            "student_cd_mean": [],
            "student_cd_std": [],
        }

        for level_idx, level in enumerate(deg_cfg["levels"]):
            level_label = deg_cfg["level_labels"][level_idx]
            print(f"\n  Level: {level_label}")

            teacher_cds = []
            student_cds = []

            for i, fd in enumerate(frame_data):
                if deg_cfg["applies_to"] == "teacher":
                    # Degrade LiDAR for teacher, student uses clean DA2
                    degraded_lidar = deg_cfg["fn"](fd["lidar"], level)
                    t_input, t_center = prepare_scan(degraded_lidar, device)
                    t_comp = run_completion_x0(teacher, t_input, target_coords=gt_targets[i])
                    t_comp += t_center
                    tcd = compute_cd(t_comp, fd["gt"])
                    teacher_cds.append(tcd)

                    # Student unchanged — use precomputed baseline
                    student_cds.append(student_baseline_cds[i])

                elif deg_cfg["applies_to"] == "student":
                    # Teacher unchanged — use precomputed baseline
                    teacher_cds.append(teacher_baseline_cds[i])

                    # Degrade DA2 for student
                    degraded_da2 = deg_cfg["fn"](fd["da2"], level)
                    s_input, s_center = prepare_scan(degraded_da2, device)
                    s_comp = run_completion_x0(student, s_input, target_coords=gt_targets[i])
                    s_comp += s_center
                    scd = compute_cd(s_comp, fd["gt"])
                    student_cds.append(scd)

            t_mean, t_std = np.mean(teacher_cds), np.std(teacher_cds)
            s_mean, s_std = np.mean(student_cds), np.std(student_cds)

            results_for_deg["teacher_cd_mean"].append(float(t_mean))
            results_for_deg["teacher_cd_std"].append(float(t_std))
            results_for_deg["student_cd_mean"].append(float(s_mean))
            results_for_deg["student_cd_std"].append(float(s_std))

            marker = " <-- CROSSOVER" if s_mean < t_mean and level_idx > 0 else ""
            print(f"    teacher: {t_mean:.4f} +/- {t_std:.4f}")
            print(f"    student: {s_mean:.4f} +/- {s_std:.4f}{marker}")

        all_results[deg_name] = results_for_deg

    # --- Save raw results ---
    results_path = os.path.join(args.output_dir, "degradation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Generate plots ---
    generate_plots(all_results, args.output_dir)
    print(f"Plots saved to {args.output_dir}/")


# =============================================================================
# Plotting
# =============================================================================

def generate_plots(all_results, output_dir):
    """Generate individual degradation plots and a combined summary."""

    # Color scheme
    teacher_color = "#2196F3"  # blue
    student_color = "#F44336"  # red
    crossover_color = "#4CAF50"  # green

    # --- Individual plots ---
    for deg_name, res in all_results.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        levels = res["levels"]
        t_means = np.array(res["teacher_cd_mean"])
        t_stds = np.array(res["teacher_cd_std"])
        s_means = np.array(res["student_cd_mean"])
        s_stds = np.array(res["student_cd_std"])

        # Plot with error bands
        ax.plot(levels, t_means, "o-", color=teacher_color, linewidth=2,
                markersize=8, label="Teacher (LiDAR)", zorder=5)
        ax.fill_between(levels, t_means - t_stds, t_means + t_stds,
                         color=teacher_color, alpha=0.15)

        ax.plot(levels, s_means, "s-", color=student_color, linewidth=2,
                markersize=8, label="Student (RGB/DA2)", zorder=5)
        ax.fill_between(levels, s_means - s_stds, s_means + s_stds,
                         color=student_color, alpha=0.15)

        # Find and mark crossover point (linear interpolation)
        crossover_level = _find_crossover(levels, t_means, s_means)
        if crossover_level is not None:
            crossover_cd = np.interp(crossover_level, levels, t_means)
            ax.axvline(x=crossover_level, color=crossover_color, linestyle="--",
                       alpha=0.7, linewidth=1.5, label=f"Crossover ({crossover_level:.2f})")
            ax.plot(crossover_level, crossover_cd, "*", color=crossover_color,
                    markersize=15, zorder=10)

        ax.set_xlabel(res["xlabel"], fontsize=13)
        ax.set_ylabel("Chamfer Distance", fontsize=13)
        ax.set_title(res["label"], fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(levels)
        ax.set_xticklabels(res["level_labels"])

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"degradation_{deg_name}.png"),
                    dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"degradation_{deg_name}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    # --- Combined 2x2 summary plot ---
    deg_names = list(all_results.keys())
    n_plots = len(deg_names)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, deg_name in enumerate(deg_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        res = all_results[deg_name]

        levels = res["levels"]
        t_means = np.array(res["teacher_cd_mean"])
        t_stds = np.array(res["teacher_cd_std"])
        s_means = np.array(res["student_cd_mean"])
        s_stds = np.array(res["student_cd_std"])

        ax.plot(levels, t_means, "o-", color=teacher_color, linewidth=2,
                markersize=6, label="Teacher (LiDAR)")
        ax.fill_between(levels, t_means - t_stds, t_means + t_stds,
                         color=teacher_color, alpha=0.15)

        ax.plot(levels, s_means, "s-", color=student_color, linewidth=2,
                markersize=6, label="Student (RGB/DA2)")
        ax.fill_between(levels, s_means - s_stds, s_means + s_stds,
                         color=student_color, alpha=0.15)

        crossover_level = _find_crossover(levels, t_means, s_means)
        if crossover_level is not None:
            crossover_cd = np.interp(crossover_level, levels, t_means)
            ax.axvline(x=crossover_level, color=crossover_color, linestyle="--",
                       alpha=0.7, linewidth=1.5)
            ax.plot(crossover_level, crossover_cd, "*", color=crossover_color,
                    markersize=12, zorder=10)

        ax.set_xlabel(res["xlabel"], fontsize=11)
        ax.set_ylabel("Chamfer Distance", fontsize=11)
        ax.set_title(res["label"], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(levels)
        ax.set_xticklabels(res["level_labels"], fontsize=9)

    # Hide extra subplots if odd number
    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Sensor Degradation Robustness: Teacher vs Student",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "degradation_summary.png"),
                dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "degradation_summary.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Crossover summary table ---
    print("\n" + "=" * 60)
    print("CROSSOVER SUMMARY")
    print("=" * 60)
    print(f"{'Degradation':<30} {'Crossover Point':<20} {'Teacher CD':<15} {'Student CD':<15}")
    print("-" * 80)
    for deg_name, res in all_results.items():
        levels = res["levels"]
        t_means = np.array(res["teacher_cd_mean"])
        s_means = np.array(res["student_cd_mean"])
        crossover_level = _find_crossover(levels, t_means, s_means)
        if crossover_level is not None:
            tcd_at = np.interp(crossover_level, levels, t_means)
            scd_at = np.interp(crossover_level, levels, s_means)
            print(f"{res['label']:<30} {crossover_level:<20.3f} {tcd_at:<15.4f} {scd_at:<15.4f}")
        else:
            if res["applies_to"] == "student":
                print(f"{res['label']:<30} {'N/A (student-only)':<20}")
            else:
                print(f"{res['label']:<30} {'No crossover':<20} "
                      f"{t_means[-1]:<15.4f} {s_means[-1]:<15.4f}")
    print("=" * 60)


def _find_crossover(levels, t_means, s_means):
    """Find the degradation level where student first beats teacher.

    Uses linear interpolation between consecutive levels for a precise estimate.
    Returns None if no crossover occurs (or if the degradation only applies
    to the student, meaning the teacher line is flat and already lower).
    """
    # Check if teacher starts below student (normal case)
    if t_means[0] >= s_means[0]:
        return None  # student already better at baseline — not a meaningful crossover

    for i in range(1, len(levels)):
        # Teacher was better (lower) before, now student is better
        if t_means[i - 1] < s_means[i - 1] and t_means[i] >= s_means[i]:
            # Linear interpolation
            dt = (t_means[i] - t_means[i - 1])
            ds = (s_means[i] - s_means[i - 1])
            # t_means[i-1] + dt*alpha = s_means[i-1] + ds*alpha
            # alpha * (dt - ds) = s_means[i-1] - t_means[i-1]
            denom = dt - ds
            if abs(denom) < 1e-10:
                alpha = 0.5
            else:
                alpha = (s_means[i - 1] - t_means[i - 1]) / denom
            alpha = np.clip(alpha, 0, 1)
            crossover_level = levels[i - 1] + alpha * (levels[i] - levels[i - 1])
            return float(crossover_level)

    return None


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sensor Degradation Evaluation for Scene Completion"
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="SemanticKITTI dataset root")
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Teacher model checkpoint")
    parser.add_argument("--student_ckpt", type=str, required=True,
                        help="Student model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_degradation",
                        help="Output directory for results and plots")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of validation frames to evaluate")
    parser.add_argument("--sequence", type=str, default="08",
                        help="SemanticKITTI sequence (08=val)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--plot_only", type=str, default=None,
                        help="Path to existing results JSON — skip eval, only regenerate plots")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.plot_only:
        print(f"Loading existing results from {args.plot_only}")
        with open(args.plot_only, "r") as f:
            all_results = json.load(f)
        os.makedirs(args.output_dir, exist_ok=True)
        generate_plots(all_results, args.output_dir)
        print(f"Plots regenerated in {args.output_dir}/")
        return

    t0 = time.time()
    evaluate_degradation(args)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
