#!/usr/bin/env python3
"""Evaluate teacher vs student scene completion on SemanticKITTI val set (seq 08)."""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion
from models.refinement_net import chamfer_distance
from data.semantickitti import SemanticKITTI


def build_model(device="cuda"):
    """Build SceneCompletionDiffusion model."""
    encoder = SonataEncoder(
        pretrained="facebook/sonata",
        freeze=True,
        enable_flash=False,
        feature_levels=[0],
    )
    condition_extractor = ConditionalFeatureExtractor(
        encoder, feature_levels=[0], fusion_type="concat"
    )
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=condition_extractor,
        num_timesteps=1000,
        schedule="cosine",
        denoising_steps=50,
    )
    return model.to(device)


def load_checkpoint(model, path, device="cuda"):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded {path} (epoch {ckpt.get('epoch', '?')})")
    return model


def bev_plot(points, title, path, xlim=(-40, 40), ylim=(-40, 40)):
    """Save bird eye view scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    z = points[:, 2]
    sc = ax.scatter(points[:, 0], points[:, 1], c=z, s=0.1, cmap="viridis",
                    vmin=-3, vmax=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(sc, ax=ax, label="Z (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def comparison_plot(input_pts, teacher_pts, student_pts, gt_pts, 
                    cd_teacher, cd_student, frame_id, out_dir):
    """Save 4-panel comparison."""
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    xlim, ylim = (-40, 40), (-40, 40)
    
    panels = [
        (input_pts, "Input (LiDAR scan)"),
        (teacher_pts, f"Teacher (CD={cd_teacher:.4f})"),
        (student_pts, f"Student (CD={cd_student:.4f})"),
        (gt_pts, "Ground Truth"),
    ]
    
    for ax, (pts, title) in zip(axes, panels):
        if pts is not None and len(pts) > 0:
            z = pts[:, 2]
            ax.scatter(pts[:, 0], pts[:, 1], c=z, s=0.1, cmap="viridis",
                       vmin=-3, vmax=3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    
    plt.suptitle(f"Scene Completion — Frame {frame_id}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"comparison_{frame_id}.png"), dpi=150)
    plt.close()


def prepare_scan(points, max_points=20000, voxel_size=0.05, device="cuda"):
    """Prepare point cloud for model input."""
    # Center
    center = points.mean(axis=0)
    points = points - center
    
    # Voxelize
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    unique, idx = np.unique(voxel_coords, axis=0, return_index=True)
    points = points[idx]
    
    # Limit points
    if len(points) > max_points:
        sel = np.random.choice(len(points), max_points, replace=False)
        points = points[sel]
    
    coord = torch.from_numpy(points).float().to(device)
    
    # Height as color
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    color = np.stack([z_norm, 1 - np.abs(z_norm - 0.5) * 2, 1 - z_norm], axis=1)
    
    point_dict = {
        "coord": coord,
        "color": torch.from_numpy(color).float().to(device),
        "normal": torch.zeros_like(coord),
        "batch": torch.zeros(len(coord), dtype=torch.long, device=device),
    }
    return point_dict, center


@torch.no_grad()
def run_completion(model, point_dict):
    """Run scene completion inference."""
    model.eval()
    completed = model.complete_scene(point_dict, num_steps=50)
    return completed.cpu().numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/anywherevla/sonata_ws/dataset/sonata_depth_pro")
    parser.add_argument("--da2_path", type=str,
                        default="/home/anywherevla/sonata_ws/dataset/sonata_depth_pro/da2_output/pointclouds/sequences/08")
    parser.add_argument("--teacher_ckpt", type=str,
                        default="checkpoints/diffusion_depthpro/best_model.pth")
    parser.add_argument("--student_ckpt", type=str,
                        default="checkpoints/distill_task_only/best_model.pth")
    parser.add_argument("--out_dir", type=str, default="evaluation_results")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    
    # Build and load models
    print("Building teacher model...")
    teacher = build_model(device)
    teacher = load_checkpoint(teacher, args.teacher_ckpt, device)
    
    print("Building student model...")
    student = build_model(device)
    student = load_checkpoint(student, args.student_ckpt, device)
    
    # Get val frames (seq 08)
    scan_dir = os.path.join(args.data_path, "sequences", "08", "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", "08")
    scans = sorted([f for f in os.listdir(scan_dir) if f.endswith(".bin")])
    
    # Sample evenly across sequence
    step = max(1, len(scans) // args.num_samples)
    sample_ids = list(range(0, len(scans), step))[:args.num_samples]
    
    teacher_cds = []
    student_cds = []
    
    print(f"\nEvaluating {len(sample_ids)} samples...\n")
    
    for i, idx in enumerate(sample_ids):
        frame_id = scans[idx].replace(".bin", "")
        print(f"[{i+1}/{len(sample_ids)}] Frame {frame_id}")
        
        # Load LiDAR scan
        lidar = np.fromfile(os.path.join(scan_dir, scans[idx]), dtype=np.float32).reshape(-1, 4)[:, :3]
        
        # Load DA2 pseudo cloud
        da2_file = os.path.join(args.da2_path, scans[idx])
        if os.path.exists(da2_file):
            da2 = np.fromfile(da2_file, dtype=np.float32).reshape(-1, 4)[:, :3]
        else:
            print(f"  DA2 file not found: {da2_file}, skipping student")
            da2 = None
        
        # Load GT
        gt_file = os.path.join(gt_dir, f"{frame_id}.npz")
        if os.path.exists(gt_file):
            gt = np.load(gt_file)["points"]
        else:
            print(f"  GT not found: {gt_file}, skipping")
            continue
        
        # Center GT same as input
        lidar_dict, center = prepare_scan(lidar, device=device)
        gt_centered = gt - center
        
        # Teacher completion
        print("  Running teacher...")
        teacher_out = run_completion(teacher, lidar_dict)
        
        # Student completion
        cd_student = float("nan")
        student_out = None
        if da2 is not None:
            da2_dict, _ = prepare_scan(da2, device=device)
            # Re-center DA2 with same center as LiDAR for fair comparison
            print("  Running student...")
            student_out = run_completion(student, da2_dict)
        
        # Compute Chamfer distances
        gt_t = torch.from_numpy(gt_centered).float().to(device)
        
        # Subsample GT if too large
        if len(gt_t) > 20000:
            sel = torch.randperm(len(gt_t))[:20000]
            gt_sub = gt_t[sel]
        else:
            gt_sub = gt_t
        
        teacher_t = torch.from_numpy(teacher_out).float().to(device)
        cd_teacher = chamfer_distance(teacher_t, gt_sub).item()
        teacher_cds.append(cd_teacher)
        
        if student_out is not None:
            student_t = torch.from_numpy(student_out).float().to(device)
            cd_student = chamfer_distance(student_t, gt_sub).item()
            student_cds.append(cd_student)
        
        print(f"  CD teacher={cd_teacher:.4f}  student={cd_student:.4f}")
        
        # Save comparison plot
        comparison_plot(
            lidar - center, teacher_out, student_out, gt_centered,
            cd_teacher, cd_student, frame_id, args.out_dir
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Teacher avg CD: {np.mean(teacher_cds):.4f} (+/- {np.std(teacher_cds):.4f})")
    if student_cds:
        print(f"Student avg CD: {np.mean(student_cds):.4f} (+/- {np.std(student_cds):.4f})")
    print(f"Teacher val loss (training): 0.213")
    print(f"Student val loss (training): 0.207")
    print(f"Samples evaluated: {len(teacher_cds)}")
    print("=" * 60)
    
    # Save metrics
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(f"Teacher avg CD: {np.mean(teacher_cds):.4f} (+/- {np.std(teacher_cds):.4f})\n")
        if student_cds:
            f.write(f"Student avg CD: {np.mean(student_cds):.4f} (+/- {np.std(student_cds):.4f})\n")
        f.write(f"\nPer-frame:\n")
        for i, idx in enumerate(sample_ids):
            frame_id = scans[idx].replace(".bin", "")
            t = teacher_cds[i] if i < len(teacher_cds) else float("nan")
            s = student_cds[i] if i < len(student_cds) else float("nan")
            f.write(f"  {frame_id}: teacher={t:.4f} student={s:.4f}\n")
    
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
