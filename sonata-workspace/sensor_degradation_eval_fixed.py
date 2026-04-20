#!/usr/bin/env python3
"""
Sensor Degradation Evaluation for 3D Scene Completion (Fixed version)

Fixes:
1. GT targets centered on input mean (not always LiDAR mean) -- fixes student CD
2. Memory-efficient: single model, swap denoiser weights for teacher/student
3. Uses v2GT checkpoints

Key insight: teacher and student share the SAME frozen Sonata encoder.
Only the denoiser weights differ. So we keep one model in VRAM and swap denoisers.
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

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(WORKSPACE, "models")):
    WORKSPACE = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
sys.path.insert(0, WORKSPACE)

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance


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


def load_denoiser_weights(model, ckpt_path, device="cuda"):
    """Load only the denoiser weights from a checkpoint into existing model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"  loaded denoiser from {ckpt_path} (epoch {epoch})")
    return model


def prepare_scan(pts_raw, device="cuda", max_points=20000, voxel_size=0.05):
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


def prepare_gt_target(gt_raw, center, device, voxel_size=0.05, max_points=20000):
    """Prepare GT target coords centered on given center."""
    gt_centered = gt_raw - center
    vc = np.floor(gt_centered / voxel_size).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    gt_sub = gt_centered[idx]
    if gt_sub.shape[0] > max_points:
        sel = np.random.choice(gt_sub.shape[0], max_points, replace=False)
        gt_sub = gt_sub[sel]
    return torch.from_numpy(gt_sub).float().to(device)


# Degradation functions
def degrade_beam_dropout(points, drop_rate):
    if drop_rate <= 0.0:
        return points.copy()
    n = points.shape[0]
    mask = np.random.rand(n) > drop_rate
    remaining = points[mask]
    if remaining.shape[0] < 100:
        keep = np.random.choice(n, 100, replace=True)
        remaining = points[keep]
    return remaining


def degrade_gaussian_noise(points, sigma):
    if sigma <= 0.0:
        return points.copy()
    return points + np.random.randn(*points.shape).astype(np.float32) * sigma


def degrade_sector_occlusion(points, angle_deg):
    if angle_deg <= 0.0:
        return points.copy()
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    sector_start = np.random.uniform(-np.pi, np.pi)
    half_angle = np.deg2rad(angle_deg) / 2.0
    diff = np.abs(azimuth - sector_start)
    diff = np.minimum(diff, 2 * np.pi - diff)
    mask = diff > half_angle
    remaining = points[mask]
    if remaining.shape[0] < 100:
        keep = np.random.choice(points.shape[0], 100, replace=True)
        remaining = points[keep]
    return remaining


def degrade_depth_noise(points, mae_target):
    if mae_target <= 0.0:
        return points.copy()
    pts = points.copy()
    ranges = np.linalg.norm(pts, axis=1, keepdims=True)
    ranges = np.clip(ranges, 1e-3, None)
    directions = pts / ranges
    sigma = mae_target * np.sqrt(np.pi / 2.0)
    range_noise = np.random.randn(*ranges.shape).astype(np.float32) * sigma
    pts = pts + directions * range_noise
    return pts


DEGRADATION_CONFIGS = {
    "beam_dropout": {
        "label": "LiDAR Beam Dropout",
        "xlabel": "Drop Rate (%)",
        "levels": [0.0, 0.25, 0.50, 0.75, 0.90],
        "level_labels": ["0%", "25%", "50%", "75%", "90%"],
        "fn": degrade_beam_dropout,
        "applies_to": "teacher",
    },
    "gaussian_noise": {
        "label": "LiDAR Gaussian Noise",
        "xlabel": "Noise s (m)",
        "levels": [0.0, 0.05, 0.10, 0.20, 0.50],
        "level_labels": ["0.0", "0.05", "0.10", "0.20", "0.50"],
        "fn": degrade_gaussian_noise,
        "applies_to": "teacher",
    },
    "sector_occlusion": {
        "label": "Angular Sector Occlusion",
        "xlabel": "Occluded Sector (deg)",
        "levels": [0.0, 30.0, 60.0, 90.0, 180.0],
        "level_labels": ["0", "30", "60", "90", "180"],
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


def evaluate_degradation(args):
    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Build single model (shared encoder) ---
    print("Building model (shared encoder)...")
    model = build_model(device)

    # --- Discover frames ---
    seq = args.sequence
    vel_dir = os.path.join(args.data_path, "sequences", seq, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", seq)
    da2_dir = os.path.join(args.data_path, "da2_output", "pointclouds", "sequences", seq)

    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])
    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][: args.num_samples]
    print(f"Evaluating {len(sample_frames)} frames from seq {seq}")

    # --- Pre-load all frame data ---
    print("\nPre-loading frame data...")
    frame_data = []
    skipped = 0
    for fid in sample_frames:
        lidar_path = os.path.join(vel_dir, f"{fid}.bin")
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        da2_path = os.path.join(da2_dir, f"{fid}.bin")
        if not os.path.exists(gt_path) or not os.path.exists(da2_path):
            skipped += 1
            continue
        lidar = load_bin(lidar_path)
        gt = load_gt(gt_path)
        da2 = load_bin(da2_path)
        frame_data.append({"fid": fid, "lidar": lidar, "gt": gt, "da2": da2})
    print(f"  loaded {len(frame_data)} frames ({skipped} skipped)")
    if len(frame_data) == 0:
        print("ERROR: no valid frames found.")
        return

    # =========================================================================
    # PHASE 1: Teacher evaluation (load teacher weights)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Teacher (LiDAR) evaluation")
    print("=" * 60)
    model = load_denoiser_weights(model, args.teacher_ckpt, device)

    print("\nComputing teacher clean baseline...")
    teacher_baseline_cds = []
    for i, fd in enumerate(frame_data):
        t_input, t_center = prepare_scan(fd["lidar"], device)
        gt_target = prepare_gt_target(fd["gt"], t_center, device)
        t_comp = run_completion_x0(model, t_input, target_coords=gt_target)
        t_comp += t_center
        tcd = compute_cd(t_comp, fd["gt"])
        teacher_baseline_cds.append(tcd)
        if (i + 1) % 10 == 0:
            print(f"  teacher baseline: {i+1}/{len(frame_data)} frames done")

    t_base_mean = np.mean(teacher_baseline_cds)
    t_base_std = np.std(teacher_baseline_cds)
    print(f"  teacher clean: {t_base_mean:.4f} +/- {t_base_std:.4f}")

    # Teacher degradation experiments (beam dropout, gaussian noise, sector occlusion)
    teacher_degradation_results = {}
    for deg_name, deg_cfg in DEGRADATION_CONFIGS.items():
        if deg_cfg["applies_to"] != "teacher":
            continue
        print(f"\n  Degradation: {deg_cfg['label']}")
        results_for_deg = {"levels": [], "cd_mean": [], "cd_std": []}
        for level_idx, level in enumerate(deg_cfg["levels"]):
            cds = []
            for i, fd in enumerate(frame_data):
                degraded = deg_cfg["fn"](fd["lidar"], level)
                t_input, t_center = prepare_scan(degraded, device)
                gt_target = prepare_gt_target(fd["gt"], t_center, device)
                t_comp = run_completion_x0(model, t_input, target_coords=gt_target)
                t_comp += t_center
                tcd = compute_cd(t_comp, fd["gt"])
                cds.append(tcd)
            m, s = np.mean(cds), np.std(cds)
            results_for_deg["levels"].append(level)
            results_for_deg["cd_mean"].append(float(m))
            results_for_deg["cd_std"].append(float(s))
            print(f"    {deg_cfg['level_labels'][level_idx]}: {m:.4f} +/- {s:.4f}")
        teacher_degradation_results[deg_name] = results_for_deg

    # =========================================================================
    # PHASE 2: Student evaluation (swap to student weights)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Student (DA2) evaluation")
    print("=" * 60)
    model = load_denoiser_weights(model, args.student_ckpt, device)

    print("\nComputing student clean baseline...")
    student_baseline_cds = []
    for i, fd in enumerate(frame_data):
        # FIX: center GT on DA2 mean, not LiDAR mean
        s_input, s_center = prepare_scan(fd["da2"], device)
        gt_target = prepare_gt_target(fd["gt"], s_center, device)
        s_comp = run_completion_x0(model, s_input, target_coords=gt_target)
        s_comp += s_center
        scd = compute_cd(s_comp, fd["gt"])
        student_baseline_cds.append(scd)
        if (i + 1) % 10 == 0:
            print(f"  student baseline: {i+1}/{len(frame_data)} frames done")

    s_base_mean = np.mean(student_baseline_cds)
    s_base_std = np.std(student_baseline_cds)
    print(f"  student clean: {s_base_mean:.4f} +/- {s_base_std:.4f}")

    # Student degradation experiment (depth noise only)
    student_degradation_results = {}
    for deg_name, deg_cfg in DEGRADATION_CONFIGS.items():
        if deg_cfg["applies_to"] != "student":
            continue
        print(f"\n  Degradation: {deg_cfg['label']}")
        results_for_deg = {"levels": [], "cd_mean": [], "cd_std": []}
        for level_idx, level in enumerate(deg_cfg["levels"]):
            cds = []
            for i, fd in enumerate(frame_data):
                degraded = deg_cfg["fn"](fd["da2"], level)
                s_input, s_center = prepare_scan(degraded, device)
                gt_target = prepare_gt_target(fd["gt"], s_center, device)
                s_comp = run_completion_x0(model, s_input, target_coords=gt_target)
                s_comp += s_center
                scd = compute_cd(s_comp, fd["gt"])
                cds.append(scd)
            m, s = np.mean(cds), np.std(cds)
            results_for_deg["levels"].append(level)
            results_for_deg["cd_mean"].append(float(m))
            results_for_deg["cd_std"].append(float(s))
            print(f"    {deg_cfg['level_labels'][level_idx]}: {m:.4f} +/- {s:.4f}")
        student_degradation_results[deg_name] = results_for_deg

    # =========================================================================
    # Combine results for JSON and plotting
    # =========================================================================
    all_results = {}
    for deg_name, deg_cfg in DEGRADATION_CONFIGS.items():
        res = {
            "label": deg_cfg["label"],
            "xlabel": deg_cfg["xlabel"],
            "levels": deg_cfg["levels"],
            "level_labels": deg_cfg["level_labels"],
            "applies_to": deg_cfg["applies_to"],
        }
        if deg_cfg["applies_to"] == "teacher":
            t_res = teacher_degradation_results[deg_name]
            res["teacher_cd_mean"] = t_res["cd_mean"]
            res["teacher_cd_std"] = t_res["cd_std"]
            # Student is constant (uses clean DA2)
            res["student_cd_mean"] = [float(s_base_mean)] * len(deg_cfg["levels"])
            res["student_cd_std"] = [float(s_base_std)] * len(deg_cfg["levels"])
        elif deg_cfg["applies_to"] == "student":
            # Teacher is constant (uses clean LiDAR)
            res["teacher_cd_mean"] = [float(t_base_mean)] * len(deg_cfg["levels"])
            res["teacher_cd_std"] = [float(t_base_std)] * len(deg_cfg["levels"])
            s_res = student_degradation_results[deg_name]
            res["student_cd_mean"] = s_res["cd_mean"]
            res["student_cd_std"] = s_res["cd_std"]
        all_results[deg_name] = res

    # --- Save results ---
    results_path = os.path.join(args.output_dir, "degradation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Generate plots ---
    generate_plots(all_results, args.output_dir)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Teacher clean baseline: {t_base_mean:.4f} +/- {t_base_std:.4f}")
    print(f"Student clean baseline: {s_base_mean:.4f} +/- {s_base_std:.4f}")
    print()
    for deg_name, res in all_results.items():
        print(f"\n{res['label']}:")
        for i, ll in enumerate(res["level_labels"]):
            tm = res["teacher_cd_mean"][i]
            sm = res["student_cd_mean"][i]
            marker = " <-- CROSSOVER" if i > 0 and sm < tm and res["teacher_cd_mean"][i-1] <= res["student_cd_mean"][i-1] else ""
            print(f"  {ll:>6}:  teacher={tm:.4f}  student={sm:.4f}{marker}")
    print("=" * 70)


def generate_plots(all_results, output_dir):
    teacher_color = "#2196F3"
    student_color = "#F44336"
    crossover_color = "#4CAF50"

    for deg_name, res in all_results.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        levels = res["levels"]
        t_means = np.array(res["teacher_cd_mean"])
        t_stds = np.array(res["teacher_cd_std"])
        s_means = np.array(res["student_cd_mean"])
        s_stds = np.array(res["student_cd_std"])

        ax.plot(levels, t_means, "o-", color=teacher_color, linewidth=2,
                markersize=8, label="Teacher (LiDAR)", zorder=5)
        ax.fill_between(levels, t_means - t_stds, t_means + t_stds,
                        color=teacher_color, alpha=0.15)
        ax.plot(levels, s_means, "s-", color=student_color, linewidth=2,
                markersize=8, label="Student (RGB/DA2)", zorder=5)
        ax.fill_between(levels, s_means - s_stds, s_means + s_stds,
                        color=student_color, alpha=0.15)

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
        plt.close(fig)

    # Combined 2x2
    deg_names = list(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, deg_name in enumerate(deg_names):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        res = all_results[deg_name]
        levels = res["levels"]
        t_means = np.array(res["teacher_cd_mean"])
        s_means = np.array(res["student_cd_mean"])
        ax.plot(levels, t_means, "o-", color=teacher_color, linewidth=2, markersize=6, label="Teacher (LiDAR)")
        ax.plot(levels, s_means, "s-", color=student_color, linewidth=2, markersize=6, label="Student (RGB/DA2)")
        ax.set_xlabel(res["xlabel"], fontsize=11)
        ax.set_ylabel("Chamfer Distance", fontsize=11)
        ax.set_title(res["label"], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(levels)
        ax.set_xticklabels(res["level_labels"], fontsize=9)
    fig.suptitle("Sensor Degradation Robustness", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "degradation_summary.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved to {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_degradation")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--sequence", type=str, default="08")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    evaluate_degradation(args)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
