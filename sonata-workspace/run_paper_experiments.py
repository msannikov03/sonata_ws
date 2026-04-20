#!/usr/bin/env python3
"""
Four paper experiments (no training, existing checkpoints only):

  1. v2GT teacher evaluated on v1 GT (single-step x0, t=200)
  2. Scaffold-from-voxelization of INPUT scan (NO GT at inference)
  3. Multi-seed robustness: teacher LiDAR/DA2 and random-PTv3 LiDAR/DA2, seeds {42, 123, 7}
  4. Cross-sequence generalization: teacher on seqs 00/02/05/06/07/09/10 (LiDAR only)

Runs sequentially, writes results to results/apr16_night/experiment_{1,2,3,4}.json
Logs to logs/paper_experiments.log (handled by nohup redirection).

Design notes / invariants carried from evaluate_fixed.py:
- Input scan prepared by `prepare_scan(raw_pts)`: centered on raw_pts.mean(), voxel 0.05,
  subsampled to 20k points. Returns centered tensor + numpy `center`.
- Diffusion runs in the CENTERED frame. To compute CD vs raw GT, add `center` back.
- target_coords (scaffold) must be in the SAME centered frame as the input coords
  (i.e. centered on the same `center` that was used to center the input scan).
- chamfer_distance is symmetric mean squared L2 (codebase convention).
- compute_cd subsamples both pred and GT to <=10k points (codebase convention).

CLI:
  python run_paper_experiments.py [--smoke]
    --smoke : run all 4 experiments with 5 frames each (verification)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
WORKDIR = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
DATA_ROOT = "/home/anywherevla/data2/dataset/sonata_depth_pro"
VEL_ROOT = os.path.join(DATA_ROOT, "sequences")
GT_V2_SYMLINK = os.path.join(DATA_ROOT, "ground_truth")            # {id}.npz (symlink -> v2)
GT_V1_DIR = os.path.join(DATA_ROOT, "ground_truth_v1")             # {id}.npz raw v1 (~200k pts)
DA2_DIR_SEQ08 = os.path.join(DATA_ROOT, "da2_output/pointclouds/sequences/08")

TEACHER_CKPT = os.path.join(WORKDIR, "checkpoints/diffusion_v2gt/best_model.pth")
RANDOM_LIDAR_CKPT = os.path.join(
    WORKDIR, "checkpoints/random_ptv3_lidar/random_unfrozen_lidar/best.pth"
)
RANDOM_DA2_CKPT = os.path.join(
    WORKDIR, "checkpoints/random_ptv3_da2/random_unfrozen_da2/best.pth"
)

RESULTS_DIR = os.path.join(WORKDIR, "results/apr16_night")
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path: str) -> np.ndarray:
    return np.load(path)["points"]


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------
def build_teacher(device="cuda"):
    enc = SonataEncoder(
        pretrained="facebook/sonata", freeze=True,
        enable_flash=False, feature_levels=[0],
    )
    cond = ConditionalFeatureExtractor(enc, feature_levels=[0], fusion_type="concat")
    m = SceneCompletionDiffusion(
        encoder=enc, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50,
    )
    return m.to(device)


def build_random_ptv3(device="cuda"):
    enc = SonataEncoder(
        pretrained="random", freeze=False,
        enable_flash=False, feature_levels=[0],
    )
    cond = ConditionalFeatureExtractor(enc, feature_levels=[0], fusion_type="concat")
    m = SceneCompletionDiffusion(
        encoder=enc, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50,
    )
    return m.to(device)


def load_ckpt(model, path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    log(f"  Loaded {path} (epoch {ck.get('epoch', '?')})")
    return model


# -----------------------------------------------------------------------------
# Scan preparation (matches evaluate_fixed.prepare_scan)
# -----------------------------------------------------------------------------
def prepare_scan(pts_raw: np.ndarray, device="cuda",
                 max_points: int = 20000, voxel_size: float = 0.05):
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
        "coord":  torch.from_numpy(pts).float().to(device),
        "color":  torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(pts.shape[0], 3).float().to(device),
        "batch":  torch.zeros(pts.shape[0], dtype=torch.long).to(device),
    }, center


def voxelize_scaffold(pts_raw: np.ndarray, center: np.ndarray,
                      voxel_size: float, max_points: int = 20000,
                      range_xy: float = None, range_z: float = None) -> torch.Tensor:
    """Voxelize pts_raw (raw coords) in the frame centered on `center`.
    Returns a (N,3) torch tensor on CUDA, still in the centered frame.
    Optional range crop (uses the LiDAR sensor origin ~= center in x,y).
    """
    pts = pts_raw - center
    if range_xy is not None:
        m = (np.abs(pts[:, 0]) < range_xy) & (np.abs(pts[:, 1]) < range_xy)
        pts = pts[m]
    if range_z is not None and pts.shape[0] > 0:
        m = np.abs(pts[:, 2]) < range_z
        pts = pts[m]
    if pts.shape[0] == 0:
        # empty: return a single dummy point at origin
        return torch.zeros(1, 3, device="cuda")
    vc = np.floor(pts / voxel_size).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    pts = pts[idx]
    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]
    return torch.from_numpy(pts).float().cuda()


def bbox_grid_scaffold(pts_raw: np.ndarray, center: np.ndarray,
                       voxel_size: float, max_points: int = 20000) -> torch.Tensor:
    """Dense voxel grid over the bbox of pts_raw (centered frame).
    Subsamples to max_points. Tests whether any dense scaffold works,
    not just the sparse voxelized input."""
    pts = pts_raw - center
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    # anchor grid to voxel corners
    lo_v = np.floor(lo / voxel_size) * voxel_size
    hi_v = np.ceil(hi / voxel_size) * voxel_size
    xs = np.arange(lo_v[0], hi_v[0] + voxel_size, voxel_size)
    ys = np.arange(lo_v[1], hi_v[1] + voxel_size, voxel_size)
    zs = np.arange(lo_v[2], hi_v[2] + voxel_size, voxel_size)
    # Full grid would be huge; instead jitter-subsample over bbox
    n = min(max_points, len(xs) * len(ys) * len(zs))
    rng = np.random.default_rng()
    X = rng.uniform(lo[0], hi[0], n)
    Y = rng.uniform(lo[1], hi[1], n)
    Z = rng.uniform(lo[2], hi[2], n)
    grid = np.stack([X, Y, Z], axis=1).astype(np.float32)
    # Snap to voxel centers
    grid = np.floor(grid / voxel_size) * voxel_size + 0.5 * voxel_size
    return torch.from_numpy(grid).float().cuda()


# -----------------------------------------------------------------------------
# Single-step x0 inference (matches evaluate_fixed.run_completion_x0)
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_completion_x0(model, point_dict, target_coords=None, t_val: int = 200):
    model.eval()
    device = point_dict["coord"].device
    cond_features, _ = model.condition_extractor(point_dict)
    if target_coords is not None:
        coords = target_coords
        cond_features = knn_interpolate(cond_features, point_dict["coord"], coords)
    else:
        coords = point_dict["coord"]
    model.scheduler._to_device(device)
    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * coords + som * noise
    t0 = time.time()
    pred_noise = model.denoiser(noisy, coords, t_tensor, {"features": cond_features})
    elapsed = time.time() - t0
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0.cpu().numpy(), elapsed


def compute_cd(pred: np.ndarray, gt: np.ndarray, max_pts: int = 10000) -> float:
    if pred.shape[0] > max_pts:
        pred = pred[np.random.choice(pred.shape[0], max_pts, replace=False)]
    if gt.shape[0] > max_pts:
        gt = gt[np.random.choice(gt.shape[0], max_pts, replace=False)]
    cd = chamfer_distance(
        torch.from_numpy(pred).float().cuda(),
        torch.from_numpy(gt).float().cuda(),
        chunk_size=512,
    )
    return float(cd.item())


def make_gt_target(gt_raw: np.ndarray, center: np.ndarray,
                   voxel_size: float = 0.05, max_points: int = 20000) -> torch.Tensor:
    gt_centered = gt_raw - center
    vc = np.floor(gt_centered / voxel_size).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    gt_sub = gt_centered[idx]
    if gt_sub.shape[0] > max_points:
        sel = np.random.choice(gt_sub.shape[0], max_points, replace=False)
        gt_sub = gt_sub[sel]
    return torch.from_numpy(gt_sub).float().cuda()


# -----------------------------------------------------------------------------
# Frame-sampling helpers
# -----------------------------------------------------------------------------
def sequence_frames(seq: str):
    vel_dir = os.path.join(VEL_ROOT, seq, "velodyne")
    return sorted(f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin"))


def seq08_full_frames():
    return sequence_frames("08")


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    log(f"  -> wrote {path}")


# -----------------------------------------------------------------------------
# EXPERIMENT 1: v2GT teacher evaluated on v1 GT
# -----------------------------------------------------------------------------
def experiment_1(num_frames: int, out_path: str):
    log("=" * 70)
    log(f"EXPERIMENT 1: v2GT teacher on v1 GT ({num_frames} frames of seq 08)")
    log("=" * 70)

    set_seed(42)
    teacher = load_ckpt(build_teacher(), TEACHER_CKPT)

    frames_all = seq08_full_frames()
    step = max(1, len(frames_all) // num_frames)
    frames = frames_all[::step][:num_frames]

    vel_dir = os.path.join(VEL_ROOT, "08", "velodyne")
    per_frame = []

    for i, fid in enumerate(frames):
        v1_path = os.path.join(GT_V1_DIR, "08", f"{fid}.npz")
        if not os.path.exists(v1_path):
            log(f"  [skip] no v1 GT: {fid}")
            continue
        lidar = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        gt_v1 = load_gt(v1_path)

        input_dict, center = prepare_scan(lidar)
        # Scaffold in centered frame uses v1 GT
        gt_target = make_gt_target(gt_v1, center)

        pred, t_elapsed = run_completion_x0(teacher, input_dict, target_coords=gt_target)
        pred += center
        cd_v1 = compute_cd(pred, gt_v1)
        per_frame.append({"frame": fid, "cd_v1": cd_v1, "time": t_elapsed})
        if (i + 1) % max(1, len(frames) // 10) == 0 or i < 3:
            log(f"  frame {fid} ({i+1}/{len(frames)}): CD(v1)={cd_v1:.4f}  t={t_elapsed:.3f}s")

    cds = [r["cd_v1"] for r in per_frame]
    m, s = mean_std(cds)
    summary = {
        "experiment": "1_teacher_v2gt_on_v1gt",
        "num_frames": len(per_frame),
        "cd_v1_mean": m, "cd_v1_std": s,
        "cd_v1_median": float(np.median(cds)) if cds else None,
        "per_frame": per_frame,
    }
    log(f"  RESULT: CD(v1) = {m:.4f} +/- {s:.4f}  (n={len(cds)})")
    save_json(summary, out_path)
    del teacher; torch.cuda.empty_cache()
    return summary


# -----------------------------------------------------------------------------
# EXPERIMENT 2: Scaffold-from-voxelization (no GT at inference)
# -----------------------------------------------------------------------------
def experiment_2(num_frames: int, out_path: str):
    log("=" * 70)
    log(f"EXPERIMENT 2: scaffold-from-voxelization (no-GT inference, {num_frames} frames)")
    log("=" * 70)

    set_seed(42)
    teacher = load_ckpt(build_teacher(), TEACHER_CKPT)

    frames_all = seq08_full_frames()
    step = max(1, len(frames_all) // num_frames)
    frames = frames_all[::step][:num_frames]

    vel_dir = os.path.join(VEL_ROOT, "08", "velodyne")
    # scaffold variants x input modalities.
    # v2 GT typical extent ~ (15m x 19m x 5m), so the input scaffold must be
    # range-cropped or the CD will be dominated by scaffold-bbox / GT-bbox
    # mismatch (sparse LiDAR extends ~80m x 80m x 20m).
    # Range values: xy = half-extent, z = half-extent around LiDAR origin.
    CROP_XY, CROP_Z = 20.0, 5.0
    variants = [
        ("voxel_0p2_r20",  "lidar",  dict(mode="voxel", voxel_size=0.2,
                                          range_xy=CROP_XY, range_z=CROP_Z)),
        ("voxel_0p1_r20",  "lidar",  dict(mode="voxel", voxel_size=0.1,
                                          range_xy=CROP_XY, range_z=CROP_Z)),
        ("voxel_0p05_r20", "lidar",  dict(mode="voxel", voxel_size=0.05,
                                          range_xy=CROP_XY, range_z=CROP_Z)),
        ("voxel_0p2_full", "lidar",  dict(mode="voxel", voxel_size=0.2)),  # no crop
        ("gt_ref",         "lidar",  dict(mode="gt")),                     # reference
        ("voxel_0p2_r20",  "da2",    dict(mode="voxel", voxel_size=0.2,
                                          range_xy=CROP_XY, range_z=CROP_Z)),
        ("voxel_0p1_r20",  "da2",    dict(mode="voxel", voxel_size=0.1,
                                          range_xy=CROP_XY, range_z=CROP_Z)),
        ("voxel_0p05_r20", "da2",    dict(mode="voxel", voxel_size=0.05,
                                          range_xy=CROP_XY, range_z=CROP_Z)),
        ("gt_ref",         "da2",    dict(mode="gt")),
    ]

    per_variant = {f"{name}__{mod}": [] for name, mod, _ in variants}

    for i, fid in enumerate(frames):
        gt_v2_path = os.path.join(GT_V2_SYMLINK, "08", f"{fid}.npz")
        if not os.path.exists(gt_v2_path):
            continue
        lidar = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        gt_v2 = load_gt(gt_v2_path)

        da2_path = os.path.join(DA2_DIR_SEQ08, f"{fid}.bin")
        has_da2 = os.path.exists(da2_path)
        da2 = load_bin(da2_path) if has_da2 else None

        for name, mod, cfg in variants:
            if mod == "da2" and not has_da2:
                continue
            raw = lidar if mod == "lidar" else da2
            input_dict, center = prepare_scan(raw)

            if cfg["mode"] == "voxel":
                scaffold = voxelize_scaffold(
                    raw, center, cfg["voxel_size"],
                    range_xy=cfg.get("range_xy"), range_z=cfg.get("range_z"),
                )
            elif cfg["mode"] == "bbox":
                scaffold = bbox_grid_scaffold(raw, center, cfg["voxel_size"])
            elif cfg["mode"] == "gt":
                scaffold = make_gt_target(gt_v2, center)
            else:
                raise ValueError(cfg)

            pred, _ = run_completion_x0(teacher, input_dict, target_coords=scaffold)
            pred_raw = pred + center
            cd = compute_cd(pred_raw, gt_v2)
            per_variant[f"{name}__{mod}"].append({"frame": fid, "cd": cd})

        if (i + 1) % max(1, len(frames) // 10) == 0 or i < 2:
            log(f"  progress: frame {fid} ({i+1}/{len(frames)})")

    summary = {
        "experiment": "2_scaffold_from_voxelization",
        "num_frames": len(frames),
        "cd_target": "v2_GT",
        "variants": {},
    }
    for key, per_f in per_variant.items():
        cds = [r["cd"] for r in per_f]
        m, s = mean_std(cds)
        summary["variants"][key] = {
            "n": len(cds), "cd_mean": m, "cd_std": s,
            "cd_median": float(np.median(cds)) if cds else None,
            "per_frame": per_f,
        }
        log(f"  {key:25s}: CD={m:.4f} +/- {s:.4f}  (n={len(cds)})")

    save_json(summary, out_path)
    del teacher; torch.cuda.empty_cache()
    return summary


# -----------------------------------------------------------------------------
# EXPERIMENT 3: Multi-seed robustness + paired Wilcoxon
# -----------------------------------------------------------------------------
def experiment_3(num_frames: int, out_path: str):
    log("=" * 70)
    log(f"EXPERIMENT 3: multi-seed robustness ({num_frames} frames, seeds 42/123/7)")
    log("=" * 70)

    seeds = [42, 123, 7]
    # (config_name, ckpt, builder, input_mode)
    configs = [
        ("teacher_lidar",      TEACHER_CKPT,      build_teacher,       "lidar"),
        ("teacher_da2",        TEACHER_CKPT,      build_teacher,       "da2"),
        ("random_ptv3_lidar",  RANDOM_LIDAR_CKPT, build_random_ptv3,   "lidar"),
        ("random_ptv3_da2",    RANDOM_DA2_CKPT,   build_random_ptv3,   "da2"),
    ]

    frames_all = seq08_full_frames()
    # Pre-select frames once (deterministic) so all seeds/configs see the same frames.
    rng = np.random.default_rng(0)
    if num_frames >= len(frames_all):
        sampled = frames_all
    else:
        sampled = sorted(rng.choice(frames_all, size=num_frames, replace=False).tolist())

    vel_dir = os.path.join(VEL_ROOT, "08", "velodyne")

    # result cache: config -> seed -> list[{frame, cd}]
    results = {c[0]: {s: [] for s in seeds} for c in configs}

    for cfg_name, ckpt_path, builder, mod in configs:
        log(f"--- {cfg_name} (ckpt={os.path.basename(ckpt_path)}) ---")
        model = load_ckpt(builder(), ckpt_path)

        for seed in seeds:
            set_seed(seed)
            per_frame = []
            for i, fid in enumerate(sampled):
                gt_v2_path = os.path.join(GT_V2_SYMLINK, "08", f"{fid}.npz")
                if not os.path.exists(gt_v2_path):
                    continue
                gt_v2 = load_gt(gt_v2_path)

                if mod == "lidar":
                    raw = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
                else:
                    da2_path = os.path.join(DA2_DIR_SEQ08, f"{fid}.bin")
                    if not os.path.exists(da2_path):
                        continue
                    raw = load_bin(da2_path)

                input_dict, center = prepare_scan(raw)
                gt_target = make_gt_target(gt_v2, center)
                pred, _ = run_completion_x0(model, input_dict, target_coords=gt_target)
                pred_raw = pred + center
                cd = compute_cd(pred_raw, gt_v2)
                per_frame.append({"frame": fid, "cd": cd})

            cds = [r["cd"] for r in per_frame]
            m, s = mean_std(cds)
            results[cfg_name][seed] = per_frame
            log(f"  seed={seed:>3d}: CD={m:.4f} +/- {s:.4f}  (n={len(cds)})")

        del model; torch.cuda.empty_cache()

    # Aggregate per-config (pool all seeds)
    summary = {
        "experiment": "3_multiseed_robustness",
        "num_frames": len(sampled),
        "seeds": seeds,
        "frames": sampled,
        "per_config": {},
        "paired_wilcoxon": {},
    }
    for cfg_name in results:
        all_cds = []
        per_seed_means = {}
        for seed in seeds:
            cds = [r["cd"] for r in results[cfg_name][seed]]
            all_cds.extend(cds)
            per_seed_means[seed] = mean_std(cds)
        m, s = mean_std(all_cds)
        summary["per_config"][cfg_name] = {
            "pooled_cd_mean": m,
            "pooled_cd_std": s,
            "per_seed_mean_std": {str(k): v for k, v in per_seed_means.items()},
            "per_seed_results": {str(k): v for k, v in results[cfg_name].items()},
        }

    # Paired Wilcoxon for LiDAR vs DA2 per model (on seed=42 per-frame CDs)
    def by_frame(items):
        return {r["frame"]: r["cd"] for r in items}

    for model_name, lidar_key, da2_key in [
        ("teacher",     "teacher_lidar",     "teacher_da2"),
        ("random_ptv3", "random_ptv3_lidar", "random_ptv3_da2"),
    ]:
        try:
            lid = by_frame(results[lidar_key][42])
            d2  = by_frame(results[da2_key][42])
            common = sorted(set(lid.keys()) & set(d2.keys()))
            x = [lid[f] for f in common]
            y = [d2[f]  for f in common]
            if len(x) >= 2:
                stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                summary["paired_wilcoxon"][model_name] = {
                    "n_pairs": len(x),
                    "statistic": float(stat),
                    "p_value": float(p),
                    "mean_lidar": float(np.mean(x)),
                    "mean_da2": float(np.mean(y)),
                    "mean_diff": float(np.mean(np.array(x) - np.array(y))),
                    "seed_used": 42,
                }
                log(f"  Wilcoxon {model_name}: n={len(x)} p={p:.4g} "
                    f"mean_lidar={np.mean(x):.4f} mean_da2={np.mean(y):.4f}")
            else:
                summary["paired_wilcoxon"][model_name] = {"error": "insufficient pairs"}
        except Exception as e:
            summary["paired_wilcoxon"][model_name] = {"error": str(e)}
            log(f"  Wilcoxon {model_name}: ERROR {e}")

    save_json(summary, out_path)
    return summary


# -----------------------------------------------------------------------------
# EXPERIMENT 4: Cross-sequence generalization
# -----------------------------------------------------------------------------
def experiment_4(num_frames_per_seq: int, out_path: str):
    log("=" * 70)
    log(f"EXPERIMENT 4: cross-sequence generalization ({num_frames_per_seq} frames/seq)")
    log("=" * 70)

    set_seed(42)
    teacher = load_ckpt(build_teacher(), TEACHER_CKPT)

    sequences = ["00", "02", "05", "06", "07", "09", "10"]
    rng = np.random.default_rng(42)

    summary = {
        "experiment": "4_cross_sequence",
        "sequences": sequences,
        "num_frames_per_seq": num_frames_per_seq,
        "per_seq": {},
    }

    for seq in sequences:
        log(f"--- seq {seq} ---")
        frames_all = sequence_frames(seq)
        if num_frames_per_seq >= len(frames_all):
            sampled = frames_all
        else:
            sampled = sorted(rng.choice(frames_all, size=num_frames_per_seq, replace=False).tolist())

        vel_dir = os.path.join(VEL_ROOT, seq, "velodyne")
        gt_dir = os.path.join(GT_V2_SYMLINK, seq)
        per_frame = []

        for i, fid in enumerate(sampled):
            gt_p = os.path.join(gt_dir, f"{fid}.npz")
            if not os.path.exists(gt_p):
                continue
            lidar = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
            gt = load_gt(gt_p)
            input_dict, center = prepare_scan(lidar)
            gt_target = make_gt_target(gt, center)
            pred, _ = run_completion_x0(teacher, input_dict, target_coords=gt_target)
            pred_raw = pred + center
            cd = compute_cd(pred_raw, gt)
            per_frame.append({"frame": fid, "cd": cd})

        cds = [r["cd"] for r in per_frame]
        m, s = mean_std(cds)
        summary["per_seq"][seq] = {
            "n": len(cds), "cd_mean": m, "cd_std": s,
            "cd_median": float(np.median(cds)) if cds else None,
            "per_frame": per_frame,
        }
        log(f"  seq {seq}: CD={m:.4f} +/- {s:.4f}  (n={len(cds)})")

    # Pool everything
    all_cds = [pf["cd"] for seq in summary["per_seq"].values() for pf in seq["per_frame"]]
    m, s = mean_std(all_cds)
    summary["pooled_cd_mean"] = m
    summary["pooled_cd_std"] = s
    log(f"  POOLED across seqs: CD={m:.4f} +/- {s:.4f}  (n={len(all_cds)})")

    save_json(summary, out_path)
    del teacher; torch.cuda.empty_cache()
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="5-frame smoke run (verification)")
    ap.add_argument("--only", type=str, default=None,
                    help="Run only one experiment: 1|2|3|4")
    args = ap.parse_args()

    if args.smoke:
        # Smoke test parameters (small, fast — verify each experiment produces
        # numeric CDs and doesn't crash)
        params = dict(e1=5, e2=5, e3=5, e4_per_seq=3)
        suffix = "_smoke"
    else:
        # Full-run parameters
        params = dict(
            e1=300,           # v1 GT eval, seq 08
            e2=150,           # scaffold-from-voxel (8 variants each frame => 8x work)
            e3=200,           # multiseed (4 configs x 3 seeds => 12x work)
            e4_per_seq=200,   # 7 seqs => 1400 frames total
        )
        suffix = ""

    log(f"=== run_paper_experiments.py  smoke={args.smoke} ===")
    log(f"params: {params}")
    log(f"results dir: {RESULTS_DIR}")

    only = args.only

    t_start = time.time()
    if only in (None, "1"):
        t = time.time()
        experiment_1(params["e1"], os.path.join(RESULTS_DIR, f"experiment_1{suffix}.json"))
        log(f"Experiment 1 done in {(time.time()-t)/60:.1f} min")
    if only in (None, "2"):
        t = time.time()
        experiment_2(params["e2"], os.path.join(RESULTS_DIR, f"experiment_2{suffix}.json"))
        log(f"Experiment 2 done in {(time.time()-t)/60:.1f} min")
    if only in (None, "3"):
        t = time.time()
        experiment_3(params["e3"], os.path.join(RESULTS_DIR, f"experiment_3{suffix}.json"))
        log(f"Experiment 3 done in {(time.time()-t)/60:.1f} min")
    if only in (None, "4"):
        t = time.time()
        experiment_4(params["e4_per_seq"], os.path.join(RESULTS_DIR, f"experiment_4{suffix}.json"))
        log(f"Experiment 4 done in {(time.time()-t)/60:.1f} min")

    log(f"=== ALL DONE in {(time.time()-t_start)/60:.1f} min ===")


if __name__ == "__main__":
    main()
