"""
Ground Truth Map Generation for SemanticKITTI

Generates complete scene maps by aggregating sequential LiDAR scans using poses.
Output format matches what SemanticKITTI dataset expects: ground_truth/{seq}/{scan_id}.npz
Each npz contains 'points': (N, 3) in the scan's coordinate frame.

Based on LiDiff's map_from_scans.py, rewritten to avoid MinkowskiEngine.
Supports numpy, open3d (faster CPU), and torch (GPU) backends.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def parse_calibration(filename: str) -> dict:
    """Parse SemanticKITTI calibration file."""
    calib = {}
    if not os.path.exists(filename):
        return calib

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            key, content = line.split(":", 1)
            values = [float(v) for v in content.strip().split()]
            pose = np.eye(4)
            pose[0, :4] = values[0:4]
            pose[1, :4] = values[4:8]
            pose[2, :4] = values[8:12]
            calib[key.strip()] = pose
    return calib

def load_poses(calib_path: str, poses_path: str) -> list:
    """Load poses and apply calibration if available."""
    poses = []

    with open(poses_path) as f:
        for line in tqdm(f, desc="Loading poses", unit="pose"):
            values = [float(v) for v in line.strip().split()]
            pose = np.eye(4)
            pose[0, :4] = values[0:4]
            pose[1, :4] = values[4:8]
            pose[2, :4] = values[8:12]

            if os.path.exists(calib_path):
                calib = parse_calibration(calib_path)
                if "Tr" in calib:
                    Tr = calib["Tr"]
                    Tr_inv = np.linalg.inv(Tr)
                    pose = Tr_inv @ pose @ Tr

            poses.append(pose)

    return poses


def voxelize_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxelize point cloud using numpy (no MinkowskiEngine)."""
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    unique_voxels = np.unique(voxel_coords, axis=0)
    voxel_centers = unique_voxels.astype(np.float32) * voxel_size + voxel_size / 2
    return voxel_centers


def voxelize_open3d(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxelize using Open3D (faster CPU, already in Sonata)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down.points, dtype=np.float32)


def _voxel_centers_merge_duplicates(centers: np.ndarray, voxel_size: float) -> np.ndarray:
    """Merge duplicate voxel centers (e.g. after chunked voxelization)."""
    coords = np.floor(centers / voxel_size).astype(np.int32)
    unique_coords = np.unique(coords, axis=0)
    return (unique_coords.astype(np.float32) * voxel_size + voxel_size / 2).astype(np.float32)


def voxelize_torch(
    points: np.ndarray,
    voxel_size: float,
    device: str = "cuda",
    chunk_size: int = 60_000_000,
) -> np.ndarray:
    """Voxelize on GPU using PyTorch in chunks to reduce memory usage."""
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n_points = points.shape[0]
    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    n_chunks = (n_points + chunk_size - 1) // chunk_size
    all_centers = []
    for start in tqdm(
        range(0, n_points, chunk_size),
        total=n_chunks,
        desc="Voxelize (chunks)",
        unit="chunk",
        leave=False,
    ):
        end = min(start + chunk_size, n_points)
        chunk = points[start:end].astype(np.float32)
        pts = torch.from_numpy(chunk).to(dev, non_blocking=True)
        # int32 instead of int64 to save GPU memory
        coords = (pts / voxel_size).floor().to(torch.int32)
        del pts
        unique_coords = torch.unique(coords, dim=0)
        del coords
        centers = unique_coords.float() * voxel_size + voxel_size / 2
        del unique_coords
        all_centers.append(centers.cpu().numpy())
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    merged = np.vstack(all_centers)
    return _voxel_centers_merge_duplicates(merged, voxel_size)


def voxelize(points: np.ndarray, voxel_size: float, backend: str = "numpy") -> np.ndarray:
    """Voxelize point cloud. backend: 'numpy' | 'open3d' | 'torch'."""
    if backend == "open3d":
        return voxelize_open3d(points, voxel_size)
    if backend == "torch":
        return voxelize_torch(points, voxel_size)
    return voxelize_numpy(points, voxel_size)


def generate_sequence_map(
    seq_path: str,
    output_dir: str,
    voxel_size: float = 0.1,
    sequences: list = None,
    backend: str = "numpy",
) -> None:
    """
    Generate ground truth maps for a sequence.

    Args:
        seq_path: Path to sequences folder (e.g., .../dataset/sequences)
        output_dir: Output root for ground_truth (e.g., .../dataset)
        voxel_size: Voxel size for aggregation
        sequences: List of sequence IDs to process (default: 00-10)
    """
    if sequences is None:
        sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    # Moving object labels to filter (SemanticKITTI)
    MOVING_LABELS = {
        252, 253, 254, 255, 256, 257, 258, 259
    }  # moving-car, moving-person, etc.

    for seq in sequences:
        seq_folder = os.path.join(seq_path, seq)
        velodyne_dir = os.path.join(seq_folder, "velodyne")
        labels_dir = os.path.join(seq_folder, "labels")
        poses_path = os.path.join(seq_folder, "poses.txt")
        calib_path = os.path.join(seq_folder, "calib.txt")

        if not os.path.exists(velodyne_dir):
            print(f"Skipping {seq}: velodyne not found")
            continue
        if not os.path.exists(poses_path):
            print(f"Skipping {seq}: poses.txt not found")
            continue

        scan_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith(".bin")])
        if len(scan_files) == 0:
            print(f"Skipping {seq}: no .bin files")
            continue

        poses = load_poses(calib_path, poses_path)
        if len(poses) != len(scan_files):
            print(f"Warning {seq}: {len(poses)} poses vs {len(scan_files)} scans")

        # Sliding window GT generation: use +-WINDOW scans instead of full
        # sequence to avoid OOM with dense Depth Pro point clouds
        WINDOW = 25
        MAX_GT_POINTS = 200000
        gt_seq_dir = os.path.join(output_dir, "ground_truth", seq)
        os.makedirs(gt_seq_dir, exist_ok=True)

        def load_scan_world(idx):
            """Load a single scan, filter, subsample, transform to world frame."""
            sf = scan_files[idx]
            sp = os.path.join(velodyne_dir, sf)
            pts = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
            # Remove moving objects
            lp = os.path.join(labels_dir, sf.replace(".bin", ".label"))
            if os.path.exists(lp):
                lb = np.fromfile(lp, dtype=np.uint32) & 0xFFFF
                mask = (lb < 252) | (lb > 259)
                if mask.sum() < len(lb):
                    pts = pts[mask]
            # Remove flying artifacts near origin
            dist = np.linalg.norm(pts[:, :3], axis=1)
            pts = pts[dist > 3.5]
            # Subsample dense Depth Pro clouds to ~50k points
            if len(pts) > 50000:
                idx_sub = np.random.choice(len(pts), 50000, replace=False)
                pts = pts[idx_sub]
            ones = np.ones((pts.shape[0], 1))
            homo = np.hstack([pts[:, :3], ones])
            return (poses[idx] @ homo.T).T[:, :3]

        # Pre-load and cache all scans in world frame
        print(f"  Loading {len(scan_files)} scans...")
        scan_cache = {}
        for idx in tqdm(range(len(scan_files)), desc=f"Loading {seq}", leave=False):
            scan_cache[idx] = load_scan_world(idx)

        for i in tqdm(range(len(scan_files)), desc=f"Sequence {seq}"):
            scan_id = scan_files[i].replace(".bin", "")
            out_path = os.path.join(gt_seq_dir, f"{scan_id}.npz")
            if os.path.exists(out_path):
                continue
            lo = max(0, i - WINDOW)
            hi = min(len(scan_files), i + WINDOW + 1)
            local_pts = [scan_cache[j] for j in range(lo, hi)]
            all_pts = np.vstack(local_pts)
            map_voxel = voxelize(all_pts, voxel_size, backend=backend)
            del all_pts
            pose_inv = np.linalg.inv(poses[i])
            ones = np.ones((map_voxel.shape[0], 1))
            map_scan = (pose_inv @ np.hstack([map_voxel, ones]).T).T[:, :3]
            del map_voxel
            if len(map_scan) > MAX_GT_POINTS:
                idx_sub = np.random.choice(len(map_scan), MAX_GT_POINTS, replace=False)
                map_scan = map_scan[idx_sub]
            np.savez_compressed(out_path, points=map_scan.astype(np.float32))

        del scan_cache
        print(f"Saved ground truth for sequence {seq} ({len(scan_files)} scans)")


def main():
    _default_root = os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset")
    _default_sequences = os.path.join(_default_root, "sequences")
    parser = argparse.ArgumentParser(
        description="Generate ground truth maps from SemanticKITTI sequences"
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=_default_sequences,
        help="Path to dataset sequences folder (default: ~/Simon_ws/dataset/SemanticKITTI/dataset/sequences)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=_default_root,
        help="Output root for ground_truth/ (default: ~/Simon_ws/dataset/SemanticKITTI/dataset)",
    )
    parser.add_argument(
        "--voxel_size",
        "-v",
        type=float,
        default=0.1,
        help="Voxel size for aggregation (default: 0.1)",
    )
    parser.add_argument(
        "--sequences",
        "-s",
        type=str,
        nargs="+",
        default=None,
        help="Sequence IDs to process (default: 00 01 02 03 04 05 06 07 08 09 10)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=["numpy", "open3d", "torch"],
        default="numpy",
        help="Voxelization backend: numpy (default), open3d (faster CPU), torch (GPU)",
    )

    args = parser.parse_args()

    seq_path = args.path.rstrip("/")
    if args.output is None:
        output_dir = os.path.dirname(seq_path)
    else:
        output_dir = args.output.rstrip("/")

    sequences = args.sequences
    if sequences is None:
        sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    print(f"Sequences path: {seq_path}")
    print(f"Output dir: {output_dir}")
    print(f"Voxel size: {args.voxel_size}")
    print(f"Backend: {args.backend}")
    print(f"Sequences: {sequences}")

    generate_sequence_map(
        seq_path=seq_path,
        output_dir=output_dir,
        voxel_size=args.voxel_size,
        sequences=sequences,
        backend=args.backend,
    )

    print(f"\nDone. Ground truth saved to {output_dir}/ground_truth/")


if __name__ == "__main__":
    main()
