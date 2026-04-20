#!/usr/bin/env python3
"""
RA-L metric suite evaluation.

Runs the teacher / random-PTv3 / student checkpoints on seq 08, saves
predicted point clouds as .npz, then computes the full metric suite
(CD, JSD, F@0.1, F@0.2, IoU@0.1, IoU@0.2, Hausdorff-95) on CPU.

Designed to coexist with concurrent GPU jobs:
  * inference is short (~24 ms/frame) and uses <4 GB VRAM
  * only one config runs at a time
  * metrics are computed on CPU (scipy cKDTree)

Usage:
    python evaluate_ral_metrics.py \
        --config teacher_v2gt_lidar_v2 \
        --num_frames 200

Configs:
    teacher_v2gt_lidar_v2  -- teacher trained on v2GT, LiDAR input, v2 GT eval
    teacher_v2gt_da2_v2    -- teacher trained on v2GT, DA2 input, v2 GT eval
    teacher_v2gt_lidar_v1  -- teacher trained on v2GT, LiDAR input, v1 GT eval
    random_ptv3_lidar_v2   -- random PTv3 trained on LiDAR, LiDAR input, v2 GT eval
    random_ptv3_da2_v2     -- random PTv3 trained on DA2, DA2 input, v2 GT eval
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
# Config registry
# -----------------------------------------------------------------------------
ROOT = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
VEL_DIR = "/home/anywherevla/data2/dataset/sonata_depth_pro/sequences/08/velodyne"
DA2_DIR = "/home/anywherevla/data2/dataset/sonata_depth_pro/da2_output/pointclouds/sequences/08"
GT_V2_DIR = "/home/anywherevla/ground_truth_v2/08"  # files: NNNNNN_v2.npz
GT_V1_DIR = "/home/anywherevla/data2/dataset/sonata_depth_pro/ground_truth_v1/08"  # files: NNNNNN.npz (true v1, NOT symlinked to v2)

TEACHER_CKPT = f"{ROOT}/checkpoints/diffusion_v2gt/best_model.pth"
RAND_LIDAR_CKPT = f"{ROOT}/checkpoints/random_ptv3_lidar/random_unfrozen_lidar/best.pth"
RAND_DA2_CKPT = f"{ROOT}/checkpoints/random_ptv3_da2/random_unfrozen_da2/best.pth"


CONFIGS = {
    "teacher_v2gt_lidar_v2": {
        "ckpt": TEACHER_CKPT, "input": "lidar", "gt": "v2",
    },
    "teacher_v2gt_da2_v2": {
        "ckpt": TEACHER_CKPT, "input": "da2", "gt": "v2",
    },
    "teacher_v2gt_lidar_v1": {
        "ckpt": TEACHER_CKPT, "input": "lidar", "gt": "v1",
    },
    "random_ptv3_lidar_v2": {
        "ckpt": RAND_LIDAR_CKPT, "input": "lidar", "gt": "v2",
        "random_encoder": True,
    },
    "random_ptv3_da2_v2": {
        "ckpt": RAND_DA2_CKPT, "input": "da2", "gt": "v2",
        "random_encoder": True,
    },
}


# -----------------------------------------------------------------------------
# Model builder variants
# -----------------------------------------------------------------------------
def build_teacher_model(device="cuda"):
    return build_model(device)


def build_random_ptv3_model(device="cuda"):
    """
    Random-init PTv3 encoder (no pretrained weights), unfrozen.
    We construct via SonataEncoder(pretrained=None, freeze=False).
    """
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


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------
def load_gt_v2(fid: str):
    """Load v2 GT. Files are named NNNNNN_v2.npz with 'points' key."""
    path = os.path.join(GT_V2_DIR, f"{fid}_v2.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    # Try common keys
    for key in ("points", "xyz", "arr_0"):
        if key in d:
            return d[key].astype(np.float32)
    # Fallback: first array in file
    return d[list(d.keys())[0]].astype(np.float32)


def load_gt_v1(fid: str):
    path = os.path.join(GT_V1_DIR, f"{fid}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    for key in ("points", "xyz", "arr_0"):
        if key in d:
            return d[key].astype(np.float32)
    return d[list(d.keys())[0]].astype(np.float32)


def load_gt_for_config(cfg_name: str, fid: str):
    cfg = CONFIGS[cfg_name]
    return load_gt_v2(fid) if cfg["gt"] == "v2" else load_gt_v1(fid)


def load_input(cfg_name: str, fid: str):
    cfg = CONFIGS[cfg_name]
    if cfg["input"] == "lidar":
        path = os.path.join(VEL_DIR, f"{fid}.bin")
        return load_bin(path)
    else:
        path = os.path.join(DA2_DIR, f"{fid}.bin")
        return load_bin(path) if os.path.exists(path) else None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--stride", type=int, default=None,
                        help="Frame stride. If None, uniform over seq.")
    parser.add_argument("--frame_start", type=int, default=None,
                        help="First frame index (inclusive). None = 0.")
    parser.add_argument("--frame_end", type=int, default=None,
                        help="Last frame index (exclusive). None = end of seq.")
    parser.add_argument("--output_dir", type=str,
                        default="results/apr17_morning")
    parser.add_argument("--save_preds", action="store_true",
                        help="Save prediction .npz files per frame.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg_name = args.config
    cfg = CONFIGS[cfg_name]
    out_dir = os.path.join(args.output_dir, cfg_name)
    os.makedirs(out_dir, exist_ok=True)
    preds_dir = os.path.join(out_dir, "preds")
    if args.save_preds:
        os.makedirs(preds_dir, exist_ok=True)

    # Build model
    print(f"[{cfg_name}] Building model...")
    if cfg.get("random_encoder", False):
        model = build_random_ptv3_model(args.device)
    else:
        model = build_teacher_model(args.device)
    model = load_ckpt(model, cfg["ckpt"], args.device)
    model.eval()

    # Frame selection
    all_frames = sorted([f.replace(".bin", "") for f in os.listdir(VEL_DIR) if f.endswith(".bin")])
    # Optional restriction to a frame-index range (to respect train/val splits).
    if args.frame_start is not None or args.frame_end is not None:
        fs = args.frame_start if args.frame_start is not None else 0
        fe = args.frame_end if args.frame_end is not None else len(all_frames)
        all_frames = [f for f in all_frames if fs <= int(f) < fe]
        print(f"[{cfg_name}] Restricted to frames [{fs}, {fe}): {len(all_frames)} frames")
    if args.stride is None:
        step = max(1, len(all_frames) // args.num_frames)
    else:
        step = args.stride
    sample_frames = all_frames[::step][: args.num_frames]
    print(f"[{cfg_name}] Evaluating {len(sample_frames)} frames (stride {step}) from seq 08")

    per_frame = []
    for i, fid in enumerate(sample_frames):
        gt = load_gt_for_config(cfg_name, fid)
        if gt is None:
            continue
        inp = load_input(cfg_name, fid)
        if inp is None:
            continue

        # Prepare input and GT target coords (centered on input mean,
        # same as evaluate_fixed.py)
        pt_dict, center = prepare_scan(inp, args.device)
        gt_centered = gt - center
        vc = np.floor(gt_centered / 0.05).astype(np.int32)
        _, idx = np.unique(vc, axis=0, return_index=True)
        gt_sub = gt_centered[idx]
        if gt_sub.shape[0] > 20000:
            sel = np.random.choice(gt_sub.shape[0], 20000, replace=False)
            gt_sub = gt_sub[sel]
        gt_target = torch.from_numpy(gt_sub).float().to(args.device)

        t0 = time.time()
        with torch.no_grad():
            pred, _ = run_completion_x0(model, pt_dict, target_coords=gt_target)
        pred = pred + center  # back to world frame
        elapsed = time.time() - t0

        # Save prediction npz for later offline re-scoring
        if args.save_preds:
            np.savez_compressed(
                os.path.join(preds_dir, f"{fid}.npz"),
                pred=pred.astype(np.float32),
                gt=gt.astype(np.float32),
            )

        # Compute all metrics (CPU)
        # Subsample for speed (match evaluate_fixed max_pts=10000 for CD parity)
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

        # Free GPU
        del pt_dict, gt_target
        torch.cuda.empty_cache()

    # Aggregate
    def mean_std(key):
        vals = [r[key] for r in per_frame if key in r and not np.isnan(r[key])]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    keys = ["cd", "cd_sq", "jsd",
            "precision@0.1", "recall@0.1", "f_score@0.1",
            "precision@0.2", "recall@0.2", "f_score@0.2",
            "iou@0.1", "iou@0.2", "hausdorff_95"]
    summary = {"config": cfg_name, "n_frames": len(per_frame)}
    for k in keys:
        mu, sd = mean_std(k)
        summary[f"{k}_mean"] = mu
        summary[f"{k}_std"] = sd

    print(f"\n[{cfg_name}] Summary over {len(per_frame)} frames:")
    for k in keys:
        mu = summary[f"{k}_mean"]
        sd = summary[f"{k}_std"]
        print(f"  {k:<20s}: {mu:.4f} +/- {sd:.4f}")

    # Save
    out = {"summary": summary, "per_frame": per_frame}
    out_path = os.path.join(out_dir, "all_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{cfg_name}] Wrote {out_path}")


if __name__ == "__main__":
    main()
