#!/usr/bin/env python3
"""
Depth Anything V2 Pipeline for SemanticKITTI
Produces: depth maps (.npy) + point clouds (.bin) for all sequences.

Usage:
    python run_depth_anything.py --kitti_root /path/to/SemanticKITTI/dataset
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

from kitti_util import Calibration


def load_model(model_name="depth-anything/Depth-Anything-V2-Large-hf", device="cuda"):
    """Load Depth Anything V2 from HuggingFace."""
    print(f"Loading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    return model, processor


def predict_depth(model, processor, image_path, device="cuda"):
    """
    Run depth prediction on a single image.
    Returns depth map in relative scale (needs rescaling for metric).
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    h, w = image.size[1], image.size[0]
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth


def depth_to_pointcloud(depth_map, calib, max_depth=80.0, min_depth=0.1):
    """Back-project depth map to 3D point cloud."""
    rows, cols = depth_map.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    depth_flat = depth_map.reshape(-1)
    valid = (depth_flat >= min_depth) & (depth_flat <= max_depth)
    u = c.reshape(-1)[valid].astype(np.float64)
    v = r.reshape(-1)[valid].astype(np.float64)
    d = depth_flat[valid]
    uv_depth = np.stack([u, v, d], axis=1)
    if calib.has_Tr:
        cloud = calib.project_image_to_velo(uv_depth)
        # Velo frame: X=forward, Y=left, Z=up. Keep forward-facing points.
        valid_pts = cloud[:, 0] >= 0
    else:
        cloud = calib.project_image_to_camera(uv_depth)
        # Camera frame: Z=forward. Keep valid depth range.
        valid_pts = (cloud[:, 2] > 0) & (cloud[:, 2] < max_depth)
    return cloud[valid_pts]


def rescale_depth_to_metric(depth_relative, lidar_scan_path, calib):
    """
    Rescale relative depth to metric using sparse LiDAR as reference.
    Projects LiDAR points to image, finds scale factor via least squares.
    """
    if not os.path.exists(lidar_scan_path):
        return depth_relative  # No LiDAR available, return as-is

    # Load LiDAR scan
    scan = np.fromfile(lidar_scan_path, dtype=np.float32).reshape(-1, 4)
    pts_3d = scan[:, :3]

    if not calib.has_Tr:
        return depth_relative  # Can't rescale without Tr

    # Project LiDAR to camera rect
    pts_ref = np.dot(calib.cart2hom(pts_3d), np.transpose(calib.V2C))
    pts_rect = np.transpose(np.dot(calib.R0, np.transpose(pts_ref)))

    # Project to image
    pts_2d = np.dot(calib.P, np.vstack([pts_rect.T, np.ones((1, pts_rect.shape[0]))]))
    pts_2d[0, :] /= pts_2d[2, :]
    pts_2d[1, :] /= pts_2d[2, :]

    u = pts_2d[0, :].astype(np.int32)
    v = pts_2d[1, :].astype(np.int32)
    z = pts_rect[:, 2]  # Depth in camera frame

    h, w = depth_relative.shape
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0.1) & (z < 80.0)
    u, v, z = u[mask], v[mask], z[mask]

    if len(z) < 10:
        return depth_relative

    # Get relative depth at LiDAR projected points
    rel_at_lidar = depth_relative[v, u]
    valid = rel_at_lidar > 0
    if valid.sum() < 10:
        return depth_relative

    # Least squares: metric = scale * relative + offset
    A = np.vstack([rel_at_lidar[valid], np.ones(valid.sum())]).T
    result = np.linalg.lstsq(A, z[valid], rcond=None)
    scale, offset = result[0]

    return depth_relative * scale + offset


def process_sequence(seq_id, kitti_root, output_root, model, processor,
                     device="cuda", max_depth=80.0, use_lidar_scale=True):
    """Process one sequence: depth inference + point cloud generation."""
    image_dir = os.path.join(kitti_root, "sequences", seq_id, "image_2")
    calib_file = os.path.join(kitti_root, "sequences", seq_id, "calib.txt")
    lidar_dir = os.path.join(kitti_root, "sequences", seq_id, "velodyne")

    if not os.path.isdir(image_dir):
        print(f"  Skipping seq {seq_id}: no image_2/")
        return 0
    if not os.path.exists(calib_file):
        print(f"  Skipping seq {seq_id}: no calib.txt")
        return 0

    depth_out = os.path.join(output_root, "depth", "sequences", seq_id)
    cloud_out = os.path.join(output_root, "pointclouds", "sequences", seq_id)
    os.makedirs(depth_out, exist_ok=True)
    os.makedirs(cloud_out, exist_ok=True)

    calib = Calibration(calib_file)
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))

    for fn in tqdm(files, desc=f"Seq {seq_id}", leave=False):
        stem = fn.replace(".png", "")
        depth_path = os.path.join(depth_out, stem + ".npy")
        cloud_path = os.path.join(cloud_out, stem + ".bin")

        # Skip if already done
        if os.path.exists(cloud_path):
            continue

        # 1. Depth inference
        image_path = os.path.join(image_dir, fn)
        depth_rel = predict_depth(model, processor, image_path, device)

        # 2. Rescale to metric using LiDAR (if available)
        if use_lidar_scale:
            lidar_path = os.path.join(lidar_dir, stem + ".bin")
            depth_metric = rescale_depth_to_metric(depth_rel, lidar_path, calib)
        else:
            depth_metric = depth_rel

        # 3. Save depth map
        np.save(depth_path, depth_metric.astype(np.float32))

        # 4. Back-project to point cloud
        cloud = depth_to_pointcloud(depth_metric, calib, max_depth=max_depth)
        cloud_with_intensity = np.concatenate(
            [cloud, np.ones((cloud.shape[0], 1))], axis=1
        ).astype(np.float32)

        # 5. Save point cloud
        cloud_with_intensity.tofile(cloud_path)

    return len(files)


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 pipeline for SemanticKITTI")
    parser.add_argument("--kitti_root", type=str, required=True,
                        help="SemanticKITTI dataset root (contains sequences/)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Output root (default: <kitti_root>/depth_anything_v2)")
    parser.add_argument("--sequences", type=str, nargs="+",
                        default=["00","01","02","03","04","05","06","07","08","09","10"],
                        help="Sequences to process (default: 00-10)")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Large-hf",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_depth", type=float, default=80.0)
    parser.add_argument("--no_lidar_scale", action="store_true",
                        help="Skip LiDAR-based metric rescaling")
    args = parser.parse_args()

    if args.output_root is None:
        args.output_root = os.path.join(args.kitti_root, "depth_anything_v2")

    model, processor = load_model(args.model, args.device)

    total = 0
    for seq in args.sequences:
        print(f"\nProcessing sequence {seq}...")
        n = process_sequence(
            seq, args.kitti_root, args.output_root,
            model, processor, args.device, args.max_depth,
            use_lidar_scale=not args.no_lidar_scale,
        )
        total += n
        print(f"  Seq {seq}: {n} frames")

    print(f"\nDone! {total} frames processed.")
    print(f"Depth maps: {args.output_root}/depth/sequences/")
    print(f"Point clouds: {args.output_root}/pointclouds/sequences/")


if __name__ == "__main__":
    main()
