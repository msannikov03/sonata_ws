#!/usr/bin/env python3
"""
Boost 2.0: anchor ICP (no chain drift).

Pipeline: pose = база, ICP = малая коррекция к локальной карте окна.
  - ICP выравнивает каждый scan к reference (последние N сканов), НЕ к предыдущему.
  - Ограничение correction: если mean(delta) >= threshold → используем pose-only.

Отличие от boost v1: chain ICP (scan→scan→scan) заменён на anchor ICP (scan→map).

Скрипт самодостаточен: общие хелперы (воксель, SOR/ROR, финализация кадра) встроены здесь,
отдельный map_from_scans_boost.py не нужен.

  python data/map_from_scans_boost_v2.py -p .../sequences -s 00 --scan_ids 000000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DATA_DIR)
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from map_from_scans import load_poses, voxelize  # noqa: E402

_DEFAULT_GT_EXPORT = os.path.join(_REPO_ROOT, "gt_maps_refined")


@dataclass(frozen=True)
class BoostDefaults:
    """Shared GT fusion defaults (formerly map_from_scans_boost.py)."""

    voxel_size: float = 0.1
    backend: str = "open3d"
    output_subdir: str = "ground_truth"
    max_gt_points: int = 200_000
    window_half: int = 20
    accumulation_radius: float = 15.0
    output_radius: float = 20.0
    force: bool = False
    quiet: bool = False
    scan_ego_min_range_m: float = 3.5
    scan_load_presample_cap: int = 50_000
    use_sor: bool = True
    use_ror: bool = True
    sor_nb_neighbors: int = 12
    sor_std_ratio: float = 2.0
    ror_nb_points: int = 5
    ror_radius: float = 0.5
    use_icp: bool = True
    icp_max_iter: int = 5
    icp_threshold: float = 1.0
    icp_downsample: float = 0.25


BOOST = BoostDefaults()

BOOST_DEFAULT_SEQUENCES: tuple[str, ...] = (
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
)


def _sor_filter_scipy(
    points: np.ndarray, nb_neighbors: int, std_ratio: float
) -> np.ndarray:
    from scipy.spatial import cKDTree

    pts = np.asarray(points[:, :3], dtype=np.float64)
    n = pts.shape[0]
    if n < 50:
        return points
    k = min(max(2, nb_neighbors), n - 1)
    tree = cKDTree(pts)
    try:
        dists, _ = tree.query(pts, k=k + 1, workers=-1)
    except TypeError:
        dists, _ = tree.query(pts, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    mu, sigma = float(mean_d.mean()), float(mean_d.std()) + 1e-9
    keep = mean_d <= mu + std_ratio * sigma
    return np.asarray(points[keep], dtype=np.float64)


def _ror_filter_scipy(
    points: np.ndarray, nb_points: int, radius: float
) -> np.ndarray:
    from scipy.spatial import cKDTree

    pts = np.asarray(points[:, :3], dtype=np.float64)
    n = pts.shape[0]
    if n < 50:
        return points
    tree = cKDTree(pts)
    try:
        neighbors = tree.query_ball_point(pts, r=radius, workers=-1)
    except TypeError:
        neighbors = tree.query_ball_point(pts, r=radius)
    counts = np.array([len(neighbors[i]) for i in range(len(neighbors))])
    return np.asarray(points[counts >= nb_points], dtype=np.float64)


def sor_filter(
    points: np.ndarray,
    nb_neighbors: int = 12,
    std_ratio: float = 2.0,
    max_points: int = 100_000,
) -> np.ndarray:
    """Statistical outlier removal (Open3D if available, else scipy)."""
    if points.shape[0] < 50:
        return points
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(points[:, :3], dtype=np.float64)
        )
        if points.shape[0] > max_points:
            pcd = pcd.voxel_down_sample(0.12)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        return np.asarray(pcd.points, dtype=np.float64)
    except (ImportError, OSError):
        return _sor_filter_scipy(points, nb_neighbors, std_ratio)


def ror_filter(
    points: np.ndarray,
    nb_points: int = 5,
    radius: float = 0.5,
    max_points: int = 100_000,
) -> np.ndarray:
    """Radius outlier removal (Open3D if available, else scipy)."""
    if points.shape[0] < 50:
        return points
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(points[:, :3], dtype=np.float64)
        )
        if points.shape[0] > max_points:
            pcd = pcd.voxel_down_sample(0.12)
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius,
        )
        return np.asarray(pcd.points, dtype=np.float64)
    except (ImportError, OSError):
        return _ror_filter_scipy(points, nb_points, radius)


def _valid_icp(T: np.ndarray, max_t: float = 0.5, max_deg: float = 5.0) -> bool:
    t = np.linalg.norm(T[:3, 3])
    R = T[:3, :3]
    val = (np.trace(R) - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    ang = np.degrees(np.arccos(val))
    return t < max_t and ang < max_deg


def fast_icp_align(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    max_iter: int = 10,
    threshold: float = 1.0,
    downsample_voxel: float = 0.0,
) -> np.ndarray:
    """Point-to-point ICP (requires Open3D)."""
    if src_pts.shape[0] < 50 or tgt_pts.shape[0] < 50:
        return src_pts.copy()
    import open3d as o3d

    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(
        np.asarray(src_pts[:, :3], dtype=np.float64)
    )
    tgt.points = o3d.utility.Vector3dVector(
        np.asarray(tgt_pts[:, :3], dtype=np.float64)
    )
    if downsample_voxel > 0:
        src = src.voxel_down_sample(downsample_voxel)
        tgt = tgt.voxel_down_sample(downsample_voxel)
        if len(src.points) < 20 or len(tgt.points) < 20:
            src = o3d.geometry.PointCloud()
            tgt = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(
                np.asarray(src_pts[:, :3], dtype=np.float64)
            )
            tgt.points = o3d.utility.Vector3dVector(
                np.asarray(tgt_pts[:, :3], dtype=np.float64)
            )
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    T = np.asarray(reg.transformation, dtype=np.float64)
    if not _valid_icp(T):
        return np.asarray(src_pts[:, :3], dtype=np.float64).copy()
    pts_h = np.hstack(
        [np.asarray(src_pts[:, :3], dtype=np.float64), np.ones((src_pts.shape[0], 1))]
    )
    aligned_full = (T @ pts_h.T).T[:, :3]
    return aligned_full


def crop_by_radius(
    points: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    if radius is None or radius <= 0 or points.shape[0] == 0:
        return points
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    dist = np.linalg.norm(points[:, :3] - c, axis=1)
    return points[dist < radius]


def crop_window_scan_for_merge(
    pts: np.ndarray,
    scan_idx: int,
    poses: np.ndarray,
    radius: float,
) -> np.ndarray:
    if radius is None or radius <= 0 or pts.shape[0] == 0:
        return pts
    c = poses[scan_idx][:3, 3].astype(np.float64)
    return crop_by_radius(pts, c, radius)


def symmetric_window_bounds(i: int, n_scans: int, half: int) -> tuple[int, int]:
    if n_scans <= 0 or half < 0:
        return 0, 0
    lo = max(0, i - half)
    hi = min(n_scans, i + half + 1)
    return lo, hi


def _boost_gt_npz_path(
    gt_seq_dir: str, scan_files: list, frame_idx: int, name_suffix: str
) -> str:
    sid = scan_files[frame_idx].replace(".bin", "")
    base = f"{sid}{name_suffix}" if name_suffix else sid
    return os.path.join(gt_seq_dir, f"{base}.npz")


def _load_kitti_scan_to_world(
    idx: int,
    scan_files: list,
    velodyne_dir: str,
    labels_dir: str,
    poses: np.ndarray,
) -> np.ndarray:
    sf = scan_files[idx]
    sp = os.path.join(velodyne_dir, sf)
    pts = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
    lp = os.path.join(labels_dir, sf.replace(".bin", ".label"))
    if os.path.exists(lp):
        lb = np.fromfile(lp, dtype=np.uint32) & 0xFFFF
        mask = (lb < 252) | (lb > 259)
        if mask.sum() < len(lb):
            pts = pts[mask]
    dist = np.linalg.norm(pts[:, :3], axis=1)
    pts = pts[dist > BOOST.scan_ego_min_range_m]
    cap = BOOST.scan_load_presample_cap
    if len(pts) > cap:
        pts = pts[np.random.choice(len(pts), cap, replace=False)]
    ones = np.ones((pts.shape[0], 1))
    homo = np.hstack([pts[:, :3], ones])
    return (poses[idx] @ homo.T).T[:, :3]


def boost_finalize_frame_from_fused(
    i: int,
    all_pts: np.ndarray,
    local_pts: list,
    c: dict,
) -> None:
    del local_pts
    scan_files = c["scan_files"]
    poses = c["poses"]
    gt_seq_dir = c["gt_seq_dir"]
    output_name_suffix = c["output_name_suffix"]
    force = c["force"]
    voxel_size = c["voxel_size"]
    backend = c["backend"]

    out_path = _boost_gt_npz_path(gt_seq_dir, scan_files, i, output_name_suffix)
    if os.path.exists(out_path) and not force:
        return

    ego_w = np.asarray(poses[i][:3, 3], dtype=np.float64)
    pose_inv = np.linalg.inv(np.asarray(poses[i], dtype=np.float64))

    output_radius = float(c["output_radius"])
    if output_radius > 0 and all_pts.shape[0] > 0:
        all_pts_f64 = np.asarray(all_pts, dtype=np.float64)
        n_before = all_pts_f64.shape[0]
        cropped = crop_by_radius(all_pts_f64, ego_w, output_radius)
        if cropped.shape[0] == 0:
            print(f"WARNING: empty crop at frame {i}")
        elif n_before > 500 and cropped.shape[0] < n_before * 0.1:
            print(
                f"WARNING: crop removed >90% points at frame {i} "
                f"({n_before} -> {cropped.shape[0]})"
            )
        all_pts = cropped

    if all_pts.shape[0] < 100:
        print(f"WARNING: low point count after fusion at frame {i}: {all_pts.shape[0]}")

    map_voxel = voxelize(all_pts, voxel_size, backend=backend)
    if c.get("use_sor", False):
        map_voxel = sor_filter(
            map_voxel,
            nb_neighbors=c.get("sor_nb_neighbors", 20),
            std_ratio=c.get("sor_std_ratio", 2.0),
        )
    if c.get("use_ror", False):
        map_voxel = ror_filter(
            map_voxel,
            nb_points=c.get("ror_nb_points", 5),
            radius=c.get("ror_radius", 0.5),
        )

    ones = np.ones((map_voxel.shape[0], 1))
    map_scan = (
        pose_inv @ np.hstack([np.asarray(map_voxel, dtype=np.float64), ones]).T
    ).T[:, :3].astype(np.float32)

    max_gt_points = c["max_gt_points"]
    if len(map_scan) > max_gt_points:
        idx_sub = np.random.choice(len(map_scan), max_gt_points, replace=False)
        map_scan = map_scan[idx_sub]

    np.savez_compressed(out_path, points=map_scan.astype(np.float32))


@dataclass(frozen=True)
class BoostV2Defaults:
    """Boost 2.0: anchor ICP, drift-free (speed-optimized defaults)."""

    icp_correction_max: float = 0.15
    icp_reference_n: int = 3
    icp_downsample: float = 0.35
    icp_max_iter: int = 4
    icp_threshold: float = 1.0
    window_half: int = 17
    sor_nb_neighbors: int = 10


BOOST_V2 = BoostV2Defaults()


class SlidingMapFusionAnchor:
    """Anchor ICP: каждый scan выравнивается к локальной карте (reference), не к предыдущему."""

    def __init__(
        self,
        window_half: int,
        n_scans: int,
        poses: np.ndarray,
        accumulation_radius: float,
        use_icp: bool = True,
        icp_max_iter: int = 4,
        icp_threshold: float = 1.0,
        icp_downsample: float = 0.35,
        icp_correction_max: float = 0.15,
        icp_reference_n: int = 3,
    ) -> None:
        self.window_half = window_half
        self.n_scans = n_scans
        self.poses = poses
        self.accumulation_radius = accumulation_radius
        self.use_icp = use_icp
        self.icp_max_iter = icp_max_iter
        self.icp_threshold = icp_threshold
        self.icp_downsample = icp_downsample
        self.icp_correction_max = icp_correction_max
        self.icp_reference_n = icp_reference_n
        self._scan_cache: dict | None = None
        self.active_indices: list[int] = []
        self.active_scans: list[np.ndarray] = []
        self._output_idx: int = 0

    def _crop_scan(self, j: int) -> np.ndarray:
        pts_j = self._scan_cache[j]
        return crop_window_scan_for_merge(
            pts_j, j, self.poses, self.accumulation_radius
        )

    def window_cropped_raw(self) -> list:
        return [self._crop_scan(j) for j in self.active_indices]

    def _align_with_anchor_icp(
        self, raw64: np.ndarray, reference_list: list[np.ndarray] | None = None
    ) -> np.ndarray:
        """Anchor ICP: align to reference (last N scans), accept only if delta < threshold."""
        ref_list = reference_list if reference_list is not None else self.active_scans
        ref_scans = ref_list[-self.icp_reference_n :]
        if not ref_scans:
            return raw64.copy()
        reference = np.vstack(ref_scans).astype(np.float64)
        if reference.shape[0] < 50:
            return raw64.copy()

        aligned_icp = fast_icp_align(
            raw64,
            reference,
            max_iter=self.icp_max_iter,
            threshold=self.icp_threshold,
            downsample_voxel=self.icp_downsample,
        )
        delta = np.linalg.norm(aligned_icp - raw64, axis=1).mean()
        if delta < self.icp_correction_max:
            return aligned_icp
        return raw64.copy()

    def initialize(self, center_idx: int, scan_cache: dict) -> np.ndarray:
        self._scan_cache = scan_cache
        self._output_idx = center_idx
        lo, hi = symmetric_window_bounds(center_idx, self.n_scans, self.window_half)
        aligned_list: list[np.ndarray] = []
        for j in range(lo, hi):
            raw = self._crop_scan(j)
            raw64 = np.asarray(raw, dtype=np.float64)
            if self.use_icp and len(aligned_list) >= 1:
                aligned = self._align_with_anchor_icp(raw64, reference_list=aligned_list)
            else:
                aligned = raw64.copy()
            aligned_list.append(aligned)
        self.active_indices = list(range(lo, hi))
        self.active_scans = aligned_list
        return np.vstack(self.active_scans).astype(np.float32)

    def update(self, new_center_idx: int, scan_cache: dict) -> np.ndarray:
        self._scan_cache = scan_cache
        self._output_idx = new_center_idx
        new_lo, new_hi = symmetric_window_bounds(
            new_center_idx, self.n_scans, self.window_half
        )
        while self.active_indices and self.active_indices[0] < new_lo:
            self.active_indices.pop(0)
            self.active_scans.pop(0)

        j_next = new_lo if not self.active_indices else self.active_indices[-1] + 1
        while j_next < new_hi:
            raw = self._crop_scan(j_next)
            raw64 = np.asarray(raw, dtype=np.float64)
            if self.use_icp and len(self.active_scans) >= 1:
                aligned = self._align_with_anchor_icp(raw64)
            else:
                aligned = raw64.copy()
            self.active_indices.append(j_next)
            self.active_scans.append(aligned)
            j_next += 1

        if not self.active_scans:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(self.active_scans).astype(np.float32)


def generate_sequence_map_boost_v2(
    seq_path: str,
    output_dir: str,
    voxel_size: float = BOOST.voxel_size,
    sequences: list = None,
    backend: str = BOOST.backend,
    output_subdir: str = "ground_truth_v2",
    scan_ids_filter: set = None,
    output_name_suffix: str = "_v2",
    force: bool = BOOST.force,
    window_half: int = BOOST_V2.window_half,
    max_gt_points: int = BOOST.max_gt_points,
    accumulation_radius: float = BOOST.accumulation_radius,
    output_radius: float = BOOST.output_radius,
    quiet: bool = BOOST.quiet,
    use_sor: bool = BOOST.use_sor,
    use_ror: bool = BOOST.use_ror,
    sor_nb_neighbors: int = BOOST_V2.sor_nb_neighbors,
    sor_std_ratio: float = BOOST.sor_std_ratio,
    ror_nb_points: int = BOOST.ror_nb_points,
    ror_radius: float = BOOST.ror_radius,
    use_icp: bool = BOOST.use_icp,
    icp_max_iter: int = BOOST_V2.icp_max_iter,
    icp_threshold: float = BOOST.icp_threshold,
    icp_downsample: float = BOOST_V2.icp_downsample,
    icp_correction_max: float = BOOST_V2.icp_correction_max,
    icp_reference_n: int = BOOST_V2.icp_reference_n,
    **kwargs,
) -> None:
    _ = kwargs

    if sequences is None:
        sequences = list(BOOST_DEFAULT_SEQUENCES)

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

        poses = load_poses(calib_path, poses_path, show_progress=not quiet)
        if len(poses) != len(scan_files):
            print(f"Warning {seq}: {len(poses)} poses vs {len(scan_files)} scans")

        n_scans = len(scan_files)
        indices = list(range(n_scans))
        if scan_ids_filter is not None:
            indices = [
                i
                for i in range(n_scans)
                if scan_files[i].replace(".bin", "") in scan_ids_filter
            ]
            if len(indices) == 0:
                print(f"  No scan_ids from {scan_ids_filter} found in {seq}, skip")
                continue
            print(f"  Filter: only scan_ids {scan_ids_filter} -> {len(indices)} frames")

        if output_subdir:
            gt_seq_dir = os.path.join(output_dir, output_subdir, seq)
        else:
            gt_seq_dir = os.path.join(output_dir, seq)
        os.makedirs(gt_seq_dir, exist_ok=True)

        if scan_ids_filter is not None:
            needed = set()
            for i in indices:
                lo, hi = symmetric_window_bounds(i, n_scans, window_half)
                needed.update(range(lo, hi))
            to_load = sorted(needed)
            print(
                f"  Loading {len(to_load)} scans "
                f"(±{window_half} around index) for boost v2 GT..."
            )
        else:
            to_load = list(range(n_scans))
            print(f"  Loading {n_scans} scans...")

        scan_cache = {}
        for idx in tqdm(to_load, desc=f"Loading {seq}", leave=False, disable=quiet):
            scan_cache[idx] = _load_kitti_scan_to_world(
                idx, scan_files, velodyne_dir, labels_dir, poses
            )

        worker_ctx = {
            "scan_files": scan_files,
            "poses": poses,
            "gt_seq_dir": gt_seq_dir,
            "output_name_suffix": output_name_suffix,
            "force": force,
            "window_half": window_half,
            "n_scans": n_scans,
            "accumulation_radius": accumulation_radius,
            "voxel_size": voxel_size,
            "backend": backend,
            "max_gt_points": max_gt_points,
            "output_radius": output_radius,
            "quiet": quiet,
            "use_sor": use_sor,
            "use_ror": use_ror,
            "sor_nb_neighbors": sor_nb_neighbors,
            "sor_std_ratio": sor_std_ratio,
            "ror_nb_points": ror_nb_points,
            "ror_radius": ror_radius,
        }

        if not quiet:
            mode = f"anchor ICP (corr<{icp_correction_max}m)" if use_icp else "pose-only"
            print(f"  Boost v2: {mode}, reference=last {icp_reference_n} scans")
            print(
                f"  Processing {len(indices)} frames (~1–2 s/frame: ICP, voxelize, SOR, ROR)..."
            )

        fusion = SlidingMapFusionAnchor(
            window_half,
            n_scans,
            poses,
            accumulation_radius,
            use_icp=use_icp,
            icp_max_iter=icp_max_iter,
            icp_threshold=icp_threshold,
            icp_downsample=icp_downsample,
            icp_correction_max=icp_correction_max,
            icp_reference_n=icp_reference_n,
        )
        prev_processed: int | None = None
        t0 = time.perf_counter()
        pbar = tqdm(
            indices,
            desc=f"{seq} boost v2",
            unit="frame",
            disable=quiet,
            ncols=80,
        )
        for i in pbar:
            out_path = _boost_gt_npz_path(
                gt_seq_dir, scan_files, i, output_name_suffix
            )
            if os.path.exists(out_path) and not force:
                continue
            if prev_processed is not None and i == prev_processed + 1:
                all_pts = fusion.update(i, scan_cache)
            else:
                all_pts = fusion.initialize(i, scan_cache)
            local_pts = fusion.window_cropped_raw()
            boost_finalize_frame_from_fused(i, all_pts, local_pts, worker_ctx)
            prev_processed = i
            if not quiet:
                pbar.set_postfix(frame=i, refresh=False)

        elapsed = time.perf_counter() - t0
        del scan_cache
        rel = os.path.join(output_subdir, seq) if output_subdir else seq
        if not quiet and len(indices) > 0:
            per_frame = elapsed / len(indices)
            print(
                f"  [{seq}] {elapsed:.1f}s total, {per_frame:.2f}s/frame -> {rel}/"
            )
        print(f"Saved boost v2 ground truth for sequence {seq} -> {rel}/")


def main():
    _default_dataset = os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset")
    _default_sequences = os.path.join(_default_dataset, "sequences")
    parser = argparse.ArgumentParser(
        description="Boost 2.0: anchor ICP (no chain drift) → voxel → scan frame → NPZ"
    )
    parser.add_argument("-p", "--path", type=str, default=_default_sequences)
    parser.add_argument("-o", "--output", type=str, default=_DEFAULT_GT_EXPORT)
    parser.add_argument("--voxel_size", "-v", type=float, default=BOOST.voxel_size)
    parser.add_argument("--sequences", "-s", type=str, nargs="+", default=None)
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["numpy", "open3d", "torch", "auto", "auto_cuda"],
        default=BOOST.backend,
    )
    parser.add_argument("--gpu_voxelize", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_subdir", type=str, default="ground_truth_v2")
    parser.add_argument("--scan_ids", nargs="+", default=None)
    parser.add_argument("--name_suffix", type=str, default="_v2")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_gt_points", type=int, default=BOOST.max_gt_points)
    parser.add_argument("--window_size", type=int, default=BOOST_V2.window_half)
    parser.add_argument(
        "--accumulation_radius", "--crop_radius",
        type=float, default=BOOST.accumulation_radius, dest="accumulation_radius",
    )
    parser.add_argument("--output_radius", type=float, default=BOOST.output_radius)
    parser.add_argument("--no-sor", action="store_true", dest="no_sor")
    parser.add_argument("--no-ror", action="store_true", dest="no_ror")
    parser.add_argument("--no-icp", action="store_true", dest="no_icp")
    parser.add_argument("--sor-neighbors", type=int, default=BOOST_V2.sor_nb_neighbors)
    parser.add_argument("--sor-std-ratio", type=float, default=BOOST.sor_std_ratio)
    parser.add_argument("--ror-nb-points", type=int, default=BOOST.ror_nb_points)
    parser.add_argument("--ror-radius", type=float, default=BOOST.ror_radius)
    parser.add_argument("--icp-max-iter", type=int, default=BOOST_V2.icp_max_iter)
    parser.add_argument("--icp-threshold", type=float, default=BOOST.icp_threshold)
    parser.add_argument("--icp-downsample", type=float, default=BOOST_V2.icp_downsample)
    parser.add_argument(
        "--icp-correction-max",
        type=float,
        default=BOOST_V2.icp_correction_max,
        help="Max mean point displacement (m) to accept ICP result; else pose-only",
    )
    parser.add_argument(
        "--icp-reference-n",
        type=int,
        default=BOOST_V2.icp_reference_n,
        help="Last N scans as ICP reference (3=fast)",
    )

    args = parser.parse_args()

    if args.gpu_voxelize:
        try:
            import torch
            if torch.cuda.is_available():
                args.backend = "torch"
            elif not args.quiet:
                print("  Note: --gpu_voxelize ignored (no CUDA)")
        except ImportError:
            if not args.quiet:
                print("  Note: --gpu_voxelize ignored (PyTorch not installed)")

    seq_path = args.path.rstrip("/")
    output_dir = os.path.abspath(args.output.rstrip("/"))
    output_subdir = args.output_subdir.strip() if args.output_subdir else "ground_truth_v2"
    sequences = args.sequences or list(BOOST_DEFAULT_SEQUENCES)
    scan_filter = set(args.scan_ids) if args.scan_ids else None

    print(f"Sequences path: {seq_path}")
    print(f"Output: {os.path.join(output_dir, output_subdir)}")
    print(f"Voxel backend: {args.backend}")
    use_icp = not args.no_icp
    use_sor = not args.no_sor
    use_ror = not args.no_ror
    print(
        f"Boost v2: anchor ICP (corr<{args.icp_correction_max}m), ref={args.icp_reference_n} scans, "
        f"accum_R={args.accumulation_radius} m, SOR={use_sor}, ROR={use_ror}"
    )

    generate_sequence_map_boost_v2(
        seq_path=seq_path,
        output_dir=output_dir,
        voxel_size=args.voxel_size,
        sequences=sequences,
        backend=args.backend,
        output_subdir=output_subdir,
        scan_ids_filter=scan_filter,
        output_name_suffix=args.name_suffix,
        force=args.force,
        window_half=args.window_size,
        max_gt_points=args.max_gt_points,
        accumulation_radius=args.accumulation_radius,
        output_radius=args.output_radius,
        quiet=args.quiet,
        use_sor=use_sor,
        use_ror=use_ror,
        sor_nb_neighbors=args.sor_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        ror_nb_points=args.ror_nb_points,
        ror_radius=args.ror_radius,
        use_icp=use_icp,
        icp_max_iter=args.icp_max_iter,
        icp_threshold=args.icp_threshold,
        icp_downsample=args.icp_downsample,
        icp_correction_max=args.icp_correction_max,
        icp_reference_n=args.icp_reference_n,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
