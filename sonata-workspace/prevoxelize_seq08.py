"""
Pre-voxelize seq 08 data for transfer to vast.ai.

Matches SemanticKITTI dataset preprocessing exactly:
  - voxel_size = 0.05
  - max_points = 20000
  - voxelize via floor divide + unique -> voxel centers
  - center on scan mean

Output per frame: compressed npz with:
  lidar_coords, lidar_center, da2_coords, da2_center,
  gt_coords_lidar, gt_coords_da2, gt_raw
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Paths
BASE = "/home/anywherevla/sonata_ws/dataset/sonata_depth_pro"
VEL_DIR = os.path.join(BASE, "sequences", "08", "velodyne")
DA2_DIR = os.path.join(BASE, "da2_output", "pointclouds", "sequences", "08")
GT_DIR = os.path.join(BASE, "ground_truth", "08")
OUT_DIR = "/home/anywherevla/sonata_ws/prevoxelized_seq08"

VOXEL_SIZE = 0.05
MAX_POINTS = 20000


def load_bin(path):
    """Load binary point cloud (float32, Nx4, take xyz)."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path):
    """Load ground truth npz."""
    return np.load(path)["points"]


def voxelize(pts, voxel_size=VOXEL_SIZE):
    """
    Voxelize point cloud matching SemanticKITTI._voxelize:
    floor divide -> unique -> voxel centers.
    """
    voxel_coords = np.floor(pts / voxel_size).astype(np.int32)
    unique_voxels, _ = np.unique(voxel_coords, axis=0, return_inverse=True)
    voxel_centers = unique_voxels.astype(np.float32) * voxel_size + voxel_size / 2.0
    return voxel_centers


def subsample(pts, max_points=MAX_POINTS):
    """Random subsample to max_points."""
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


def process_scan(raw_pts):
    """Center, voxelize, subsample a scan. Returns (coords, center)."""
    center = raw_pts.mean(axis=0).astype(np.float32)
    centered = raw_pts - center
    voxelized = voxelize(centered)
    subsampled = subsample(voxelized)
    return subsampled.astype(np.float32), center


def process_gt(raw_gt, center):
    """Center GT on given center, voxelize, subsample."""
    centered = raw_gt - center
    voxelized = voxelize(centered)
    subsampled = subsample(voxelized)
    return subsampled.astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Discover frames
    frames = sorted(
        f.replace(".bin", "")
        for f in os.listdir(VEL_DIR)
        if f.endswith(".bin")
    )
    n_frames = len(frames)
    print(f"Found {n_frames} frames in seq 08")

    # Set seed for reproducible subsampling
    np.random.seed(42)

    t0 = time.time()
    total_bytes = 0

    for i, fid in enumerate(frames):
        # Load raw data
        lidar_raw = load_bin(os.path.join(VEL_DIR, f"{fid}.bin"))
        da2_raw = load_bin(os.path.join(DA2_DIR, f"{fid}.bin"))
        gt_raw = load_gt(os.path.join(GT_DIR, f"{fid}.npz"))

        # Process LiDAR
        lidar_coords, lidar_center = process_scan(lidar_raw)

        # Process DA2
        da2_coords, da2_center = process_scan(da2_raw)

        # Process GT (centered on LiDAR mean)
        gt_coords_lidar = process_gt(gt_raw, lidar_center)

        # Process GT (centered on DA2 mean)
        gt_coords_da2 = process_gt(gt_raw, da2_center)

        # Save compressed npz
        out_path = os.path.join(OUT_DIR, f"{fid}.npz")
        np.savez_compressed(
            out_path,
            lidar_coords=lidar_coords,
            lidar_center=lidar_center,
            da2_coords=da2_coords,
            da2_center=da2_center,
            gt_coords_lidar=gt_coords_lidar,
            gt_coords_da2=gt_coords_da2,
            gt_raw=gt_raw.astype(np.float32),
        )

        fsize = os.path.getsize(out_path)
        total_bytes += fsize

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_frames - i - 1) / rate
            print(
                f"  [{i+1:4d}/{n_frames}] {fid} | "
                f"lidar={lidar_coords.shape[0]:5d} da2={da2_coords.shape[0]:5d} "
                f"gt_l={gt_coords_lidar.shape[0]:5d} gt_d={gt_coords_da2.shape[0]:5d} "
                f"gt_raw={gt_raw.shape[0]:6d} | "
                f"{fsize/1024:.0f}KB | "
                f"{rate:.1f} fr/s | ETA {eta:.0f}s"
            )

    elapsed = time.time() - t0
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)

    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {OUT_DIR}")
    print(f"Files: {n_frames}")
    print(f"Total size: {total_mb:.1f} MB ({total_gb:.2f} GB)")

    # Print sample shapes from first and last frame
    for label, fid in [("First", frames[0]), ("Last", frames[-1])]:
        d = np.load(os.path.join(OUT_DIR, f"{fid}.npz"))
        print(f"\n{label} frame ({fid}):")
        for key in sorted(d.keys()):
            print(f"  {key}: shape={d[key].shape}, dtype={d[key].dtype}")


if __name__ == "__main__":
    main()
