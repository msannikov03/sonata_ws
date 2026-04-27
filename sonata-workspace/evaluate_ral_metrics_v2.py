#!/usr/bin/env python3
"""
RA-L metric suite evaluation — patched for cross-sequence + SLAM-like scaffold.

Adds:
  --sequence       SemanticKITTI sequence id (e.g. "00", "05", "08"). Default "08".
  --scaffold_source {gt, accumulated, raw_input}
                   gt          : default, GT coords noised at t=200 (original behavior)
                   accumulated : union of LiDAR scans in frames [t-k, t+k], pose-aligned
                                 (no GT peek; SLAM-like deployment scaffold)
                   raw_input   : current frame's raw LiDAR scan only (no GT, no accum)
  --scaffold_window K   half-window for accumulated mode (default 5)

DA2 input is only available for sequence 08.
"""
import os, sys, argparse, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.metrics import compute_all_metrics
from evaluate_fixed import (
    build_model, load_ckpt, prepare_scan, load_bin, load_gt, run_completion_x0,
)

# Reproducibility seeds
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------------------------------------------------------------
# Path templates (filled with --sequence)
# -----------------------------------------------------------------------------
ROOT = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
VEL_TPL = "/home/anywherevla/data2/dataset/sonata_depth_pro/sequences/{seq}/velodyne"
DA2_TPL = "/home/anywherevla/data2/dataset/sonata_depth_pro/da2_output/pointclouds/sequences/{seq}"
GT_V2_TPL = "/home/anywherevla/ground_truth_v2/{seq}"  # files: NNNNNN_v2.npz
GT_V1_TPL = "/home/anywherevla/data2/dataset/sonata_depth_pro/ground_truth_v1/{seq}"
POSES_TPL = "/home/anywherevla/data2/dataset/sonata_depth_pro/sequences/{seq}/poses.txt"

TEACHER_CKPT = f"{ROOT}/checkpoints/diffusion_v2gt/best_model.pth"
RAND_LIDAR_CKPT = f"{ROOT}/checkpoints/random_ptv3_lidar/random_unfrozen_lidar/best.pth"
RAND_DA2_CKPT = f"{ROOT}/checkpoints/random_ptv3_da2/random_unfrozen_da2/best.pth"


CONFIGS = {
    "teacher_v2gt_lidar_v2": {"ckpt": TEACHER_CKPT, "input": "lidar", "gt": "v2"},
    "teacher_v2gt_da2_v2":   {"ckpt": TEACHER_CKPT, "input": "da2",   "gt": "v2"},
    "teacher_v2gt_lidar_v1": {"ckpt": TEACHER_CKPT, "input": "lidar", "gt": "v1"},
    "random_ptv3_lidar_v2":  {"ckpt": RAND_LIDAR_CKPT, "input": "lidar", "gt": "v2",
                              "random_encoder": True},
    "random_ptv3_da2_v2":    {"ckpt": RAND_DA2_CKPT,   "input": "da2",   "gt": "v2",
                              "random_encoder": True},
}


def build_teacher_model(device="cuda"):
    return build_model(device)


def build_random_ptv3_model(device="cuda"):
    from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
    from models.diffusion_module import SceneCompletionDiffusion
    encoder = SonataEncoder(
        pretrained="random", freeze=False,
        enable_flash=False, feature_levels=[0]
    )
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    model = SceneCompletionDiffusion(
        encoder=encoder, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50
    )
    return model.to(device)


def load_npz_points(path):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    for key in ("points", "xyz", "arr_0"):
        if key in d:
            return d[key].astype(np.float32)
    return d[list(d.keys())[0]].astype(np.float32)


def load_gt_v2(gt_v2_dir, fid):
    return load_npz_points(os.path.join(gt_v2_dir, f"{fid}_v2.npz"))


def load_gt_v1(gt_v1_dir, fid):
    return load_npz_points(os.path.join(gt_v1_dir, f"{fid}.npz"))


def load_poses(poses_path):
    """Load KITTI poses.txt: each line is 12 floats = 3x4 [R|t] LiDAR->world."""
    if not os.path.exists(poses_path):
        return None
    arr = np.loadtxt(poses_path)
    if arr.ndim == 1:
        arr = arr[None, :]
    n = arr.shape[0]
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    poses[:, :3, :] = arr.reshape(n, 3, 4)
    poses[:, 3, 3] = 1.0
    return poses


def build_accumulated_scaffold(vel_dir, fid_int, poses, window=5,
                               max_pts=200000, voxel=0.05):
    """
    Union of LiDAR scans from frames [fid-window, fid+window], transformed
    into current frame's LiDAR coords using poses, then voxelized at 0.05m.
    Returns Nx3 numpy array (in current LiDAR frame), or None on failure.
    """
    if poses is None:
        return None
    n = poses.shape[0]
    if fid_int >= n:
        return None
    T_w_t = poses[fid_int]
    try:
        T_t_w = np.linalg.inv(T_w_t)
    except np.linalg.LinAlgError:
        return None

    pts_all = []
    for j in range(max(0, fid_int - window), min(n, fid_int + window + 1)):
        bin_path = os.path.join(vel_dir, f"{j:06d}.bin")
        if not os.path.exists(bin_path):
            continue
        scan = load_bin(bin_path)  # raw, vehicle frame
        if scan is None or scan.shape[0] == 0:
            continue
        if j == fid_int:
            pts_all.append(scan.astype(np.float32))
            continue
        T_w_j = poses[j]
        T_t_j = T_t_w @ T_w_j  # transform from frame j coords to frame t coords
        ones = np.ones((scan.shape[0], 1), dtype=np.float32)
        scan_h = np.hstack([scan, ones])
        scan_t = (T_t_j @ scan_h.T).T[:, :3]
        pts_all.append(scan_t.astype(np.float32))

    if not pts_all:
        return None
    pts = np.concatenate(pts_all, axis=0)

    # Voxelize at 0.05 m
    vc = np.floor(pts / voxel).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    pts = pts[idx]

    # Cap point count for memory
    if pts.shape[0] > max_pts:
        sel = np.random.choice(pts.shape[0], max_pts, replace=False)
        pts = pts[sel]
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--sequence", type=str, default="08",
                        help="SemanticKITTI sequence id, e.g. '00', '05', '08'")
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/apr17_morning")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scaffold_source", type=str, default="gt",
                        choices=["gt", "accumulated", "raw_input"],
                        help="gt = original (default); accumulated = SLAM-like neighbor scans; raw_input = current scan only")
    parser.add_argument("--scaffold_window", type=int, default=5,
                        help="half-window for accumulated scaffold")
    parser.add_argument("--scaffold_crop", type=str, default="none",
                        choices=["none", "gt_bbox", "frustum"],
                        help="Crop scaffold to: none, GT bbox (gt_bbox), or fixed deployment frustum (frustum)")
    parser.add_argument("--scaffold_jitter", type=float, default=0.0,
                        help="Add isotropic Gaussian noise (sigma) to scaffold before voxelization (deployment robustness)")
    args = parser.parse_args()

    seq = args.sequence
    vel_dir = VEL_TPL.format(seq=seq)
    da2_dir = DA2_TPL.format(seq=seq)
    gt_v2_dir = GT_V2_TPL.format(seq=seq)
    gt_v1_dir = GT_V1_TPL.format(seq=seq)
    poses_path = POSES_TPL.format(seq=seq)

    cfg_name = args.config
    cfg = CONFIGS[cfg_name]

    # Use scaffold-source-aware output dir tag so files don't clobber
    tag = cfg_name
    if seq != "08":
        tag = f"{cfg_name}_seq{seq}"
    if args.scaffold_source != "gt":
        tag = f"{tag}_scaff_{args.scaffold_source}_w{args.scaffold_window}"
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    preds_dir = os.path.join(out_dir, "preds")
    if args.save_preds:
        os.makedirs(preds_dir, exist_ok=True)

    poses = None
    if args.scaffold_source == "accumulated":
        poses = load_poses(poses_path)
        if poses is None:
            print(f"[{tag}] FATAL: poses.txt missing at {poses_path}; cannot use accumulated scaffold")
            sys.exit(1)
        print(f"[{tag}] Loaded {poses.shape[0]} poses from {poses_path}")

    print(f"[{tag}] Building model...")
    if cfg.get("random_encoder", False):
        model = build_random_ptv3_model(args.device)
    else:
        model = build_teacher_model(args.device)
    model = load_ckpt(model, cfg["ckpt"], args.device)
    model.eval()

    if not os.path.isdir(vel_dir):
        print(f"[{tag}] FATAL: VEL_DIR missing at {vel_dir}")
        sys.exit(1)
    all_frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])
    if args.frame_start is not None or args.frame_end is not None:
        fs = args.frame_start if args.frame_start is not None else 0
        fe = args.frame_end if args.frame_end is not None else len(all_frames)
        all_frames = [f for f in all_frames if fs <= int(f) < fe]
        print(f"[{tag}] Restricted to frames [{fs}, {fe}): {len(all_frames)} frames")
    if args.stride is None:
        step = max(1, len(all_frames) // args.num_frames)
    else:
        step = args.stride
    sample_frames = all_frames[::step][: args.num_frames]
    print(f"[{tag}] Evaluating {len(sample_frames)} frames (stride {step}) from seq {seq}, scaffold={args.scaffold_source}")

    per_frame = []
    for i, fid in enumerate(sample_frames):
        # Load GT for metric scoring (always GT, regardless of scaffold source)
        if cfg["gt"] == "v2":
            gt = load_gt_v2(gt_v2_dir, fid)
        else:
            gt = load_gt_v1(gt_v1_dir, fid)
        if gt is None:
            continue

        # Load input
        if cfg["input"] == "lidar":
            inp = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        else:
            da2_path = os.path.join(da2_dir, f"{fid}.bin")
            inp = load_bin(da2_path) if os.path.exists(da2_path) else None
        if inp is None:
            continue

        pt_dict, center = prepare_scan(inp, args.device)

        # Build scaffold (target_coords) according to --scaffold_source
        if args.scaffold_source == "gt":
            scaffold_world = gt
        elif args.scaffold_source == "raw_input":
            scaffold_world = inp
        else:  # accumulated
            acc = build_accumulated_scaffold(
                vel_dir, int(fid), poses, window=args.scaffold_window
            )
            if acc is None:
                continue
            scaffold_world = acc

        # Optional crop in world (vehicle) frame BEFORE centering
        if args.scaffold_crop == "gt_bbox":
            mn, mx = gt.min(axis=0), gt.max(axis=0)
            mask = ((scaffold_world >= mn) & (scaffold_world <= mx)).all(axis=1)
            scaffold_world = scaffold_world[mask]
            if scaffold_world.shape[0] < 100:
                continue
        elif args.scaffold_crop == "frustum":
            # Fixed deployment frustum in vehicle frame: forward 0..30m, side ±15m, vertical [-3,3]m
            mn = np.array([0.0, -15.0, -3.0], dtype=np.float32)
            mx = np.array([30.0, 15.0, 3.0], dtype=np.float32)
            mask = ((scaffold_world >= mn) & (scaffold_world <= mx)).all(axis=1)
            scaffold_world = scaffold_world[mask]
            if scaffold_world.shape[0] < 100:
                continue

        # Optional Gaussian jitter to test scaffold-quality robustness
        if args.scaffold_jitter > 0:
            scaffold_world = scaffold_world + np.random.normal(
                0, args.scaffold_jitter, scaffold_world.shape
            ).astype(np.float32)

        scaffold = scaffold_world - center

        # Voxelize and subsample scaffold (same recipe as GT path)
        vc = np.floor(scaffold / 0.05).astype(np.int32)
        _, idx = np.unique(vc, axis=0, return_index=True)
        scaffold_sub = scaffold[idx]
        if scaffold_sub.shape[0] > 20000:
            sel = np.random.choice(scaffold_sub.shape[0], 20000, replace=False)
            scaffold_sub = scaffold_sub[sel]
        target_coords = torch.from_numpy(scaffold_sub).float().to(args.device)

        t0 = time.time()
        with torch.no_grad():
            pred, _ = run_completion_x0(model, pt_dict, target_coords=target_coords)
        pred = pred + center  # back to world frame for metric vs world-frame GT
        elapsed = time.time() - t0

        if args.save_preds:
            np.savez_compressed(
                os.path.join(preds_dir, f"{fid}.npz"),
                pred=pred.astype(np.float32),
                gt=gt.astype(np.float32),
            )

        pmax = 20000
        if pred.shape[0] > pmax:
            pred_m = pred[np.random.choice(pred.shape[0], pmax, replace=False)]
        else:
            pred_m = pred
        if gt.shape[0] > pmax:
            gt_m = gt[np.random.choice(gt.shape[0], pmax, replace=False)]
        else:
            gt_m = gt

        m = compute_all_metrics(pred_m, gt_m)
        m["frame"] = fid
        m["time_s"] = elapsed
        per_frame.append(m)
        if (i + 1) % 20 == 0 or i == len(sample_frames) - 1:
            print(f"  [{i+1}/{len(sample_frames)}] CD={m['cd']:.4f} JSD={m['jsd']:.4f} "
                  f"F@0.1={m['f_score@0.1']:.3f} IoU@0.2={m['iou@0.2']:.3f} "
                  f"H95={m['hausdorff_95']:.3f}")

        del pt_dict, target_coords
        torch.cuda.empty_cache()

    def mean_std(key):
        vals = [r[key] for r in per_frame if key in r and not np.isnan(r[key])]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    keys = ["cd", "cd_sq", "jsd",
            "precision@0.1", "recall@0.1", "f_score@0.1",
            "precision@0.2", "recall@0.2", "f_score@0.2",
            "iou@0.1", "iou@0.2", "hausdorff_95"]
    summary = {"config": cfg_name, "sequence": seq,
               "scaffold_source": args.scaffold_source,
               "scaffold_window": args.scaffold_window,
               "n_frames": len(per_frame)}
    for k in keys:
        mu, sd = mean_std(k)
        summary[f"{k}_mean"] = mu
        summary[f"{k}_std"] = sd

    print(f"\n[{tag}] Summary over {len(per_frame)} frames:")
    for k in keys:
        mu = summary[f"{k}_mean"]
        sd = summary[f"{k}_std"]
        print(f"  {k:<20s}: {mu:.4f} +/- {sd:.4f}")

    out = {"summary": summary, "per_frame": per_frame}
    out_path = os.path.join(out_dir, "all_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] Wrote {out_path}")


if __name__ == "__main__":
    main()
