#!/usr/bin/env python3
"""
FAIR Scaffold-Free Teacher Evaluation (apr17 morning experiment, fair variant).

Fixes the GT-bbox information leak in run_scaffoldfree.py variant A:
  - Original variant A cropped the scaffold input to a bbox derived from the GT
    (gt_centered.min() .. gt_centered.max() + 0.5m). This gave the teacher
    privileged GT extent info that LiDiff does NOT have.
  - LiDiff generates freely in the full extent, then CD is computed after
    cropping the generated cloud to GT +/- 1.0m (see
    /home/anywherevla/GitHub/LiDiff/lidiff/tools/lidiff_eval_v2gt.py lines 121-133).

This script re-runs scaffold-free variants with NO GT-derived info on the
scaffold input. Instead:

  A_fair_ego_bbox : duplicate input 10x with jitter, crop to a FIXED ego-vehicle
                    bbox +/-40m xy, +/-5m z (NOT GT-derived), subsample to 20k,
                    single-step x0 at t=200, then crop prediction to GT+/-1m
                    for CD (same as LiDiff eval).
  A_fair_full     : same but no crop at all on the scaffold input (full
                    LiDAR extent after duplication+jitter), subsample to 20k.
                    CD computed with GT+/-1m crop (LiDiff protocol).
  A_fair_no_crop  : same as A_fair_full but NO crop on CD either (maximally
                    honest; may be dominated by extent mismatch).
  A_fair_lidiff_match : A_fair_ego_bbox, CD computed with GT+/-1m crop AND
                        with bidirectional squared-L2 CD implemented the SAME
                        way as lidiff_eval_v2gt.py (symmetric mean squared
                        distances with scipy KDTree). Direct apples-to-apples
                        vs LiDiff 3.41.

Reference numbers to compare against:
  Old UNFAIR variant A (GT-bbox leak):           2.37
  LiDiff (published on our v2 GT):               3.41 (cd_diff_mean)
  ScoreLiDAR (LiDiff refinement on v2 GT):       3.50 (cd_refine_mean)

Results -> results/apr17_morning/teacher_without_scaffold_fair.json
"""
import os, sys, time, json
from pathlib import Path

import torch
import numpy as np

WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR))

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance

# ---- config ----
PREVOX_DIR = Path("/home/anywherevla/sonata_ws/prevoxelized_seq08")
CKPT_PATH = WORK_DIR / "checkpoints" / "diffusion_v2gt" / "best_model.pth"
RESULTS_DIR = WORK_DIR / "results" / "apr17_morning"
RESULTS_PATH = RESULTS_DIR / "teacher_without_scaffold_fair.json"
NUM_FRAMES = 50
DEVICE = "cuda"
MAX_PTS = 20000
SEED = 42

# Fixed ego-vehicle bbox (NOT GT-derived). Values typical for SemanticKITTI
# autonomous-driving scene extents.
EGO_BBOX_MIN = np.array([-40.0, -40.0, -5.0], dtype=np.float32)
EGO_BBOX_MAX = np.array([ 40.0,  40.0,  5.0], dtype=np.float32)

# LiDiff-style post-hoc CD crop margin (applied around GT after prediction).
LIDIFF_MARGIN = 1.0


# ---- model / data helpers (shared with scaffold_sweep) ----
def build_model(device=DEVICE):
    encoder = SonataEncoder(
        pretrained="facebook/sonata", freeze=True,
        enable_flash=False, feature_levels=[0],
    )
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    model = SceneCompletionDiffusion(
        encoder=encoder, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50,
    )
    return model.to(device)


def load_ckpt(model, path, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[ckpt] loaded {path}  (epoch {ckpt.get('epoch', '?')})")
    return model


def make_point_dict(coords, device=DEVICE):
    pts = coords if isinstance(coords, np.ndarray) else coords.cpu().numpy()
    z = pts[:, 2]
    zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1)
    return {
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(pts.shape[0], 3, dtype=torch.float32, device=device),
        "batch": torch.zeros(pts.shape[0], dtype=torch.long, device=device),
    }


def compute_cd_torch(pred, gt, max_pts=10000):
    """Symmetric squared-L2 CD via refinement_net (matches LiDiff's cd_mean convention:
    0.5*(mean_sq_ab + mean_sq_ba))."""
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return float("nan")
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


def compute_cd_lidiff_kdtree(pred, gt, max_pts=60000):
    """Exact match to lidiff_eval_v2gt.py:chamfer_distance_bidirectional.
    Symmetric squared-L2 CD computed with scipy KDTree (deterministic,
    no chunking differences)."""
    from scipy.spatial import cKDTree
    a = np.asarray(pred, dtype=np.float32)
    b = np.asarray(gt, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    if a.shape[0] > max_pts:
        idx = np.random.default_rng(0).choice(a.shape[0], max_pts, replace=False)
        a = a[idx]
    if b.shape[0] > max_pts:
        idx = np.random.default_rng(0).choice(b.shape[0], max_pts, replace=False)
        b = b[idx]
    tree_b = cKDTree(b)
    d_ab, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    d_ba, _ = tree_a.query(b, k=1)
    return 0.5 * (float(np.mean(d_ab ** 2)) + float(np.mean(d_ba ** 2)))


def load_frames(n=NUM_FRAMES):
    files = sorted(PREVOX_DIR.glob("*.npz"))[:n]
    frames = []
    for f in files:
        d = np.load(f)
        frames.append({
            "name": f.stem,
            "lidar_coords": d["lidar_coords"],      # centered on lidar mean
            "lidar_center": d["lidar_center"],
            "gt_coords_lidar": d["gt_coords_lidar"], # centered on lidar mean
            "gt_raw": d["gt_raw"],                   # world-frame GT
        })
    print(f"[data] loaded {len(frames)} frames from {PREVOX_DIR}")
    return frames


# ---- FAIR scaffold constructors (no GT info) ----
def scaffold_input_duplicated_fair(fr, n_dup=10, jitter=0.05, max_pts=MAX_PTS,
                                    crop_mode="ego"):
    """FAIR version: duplicate partial scan N times with jitter, optionally crop to
    a FIXED ego-vehicle bbox (NOT GT-derived). Subsample to max_pts.

    crop_mode:
        "ego"  -> crop to EGO_BBOX_MIN/MAX (fixed, not GT-derived)
        "none" -> no crop at all, use full extent of duplicated+jittered cloud
    """
    lidar = fr["lidar_coords"]  # already centered on lidar_center

    dups = []
    for i in range(n_dup):
        if i == 0:
            dups.append(lidar)
        else:
            dups.append(lidar + np.random.normal(0.0, jitter, size=lidar.shape).astype(np.float32))
    cloud = np.concatenate(dups, axis=0)

    if crop_mode == "ego":
        # Fixed ego bbox (centered frame). NOT GT-derived.
        mask = np.all((cloud >= EGO_BBOX_MIN) & (cloud <= EGO_BBOX_MAX), axis=1)
        cloud = cloud[mask]
    elif crop_mode == "none":
        pass
    else:
        raise ValueError(f"unknown crop_mode: {crop_mode}")

    if cloud.shape[0] > max_pts:
        idx = np.random.choice(cloud.shape[0], max_pts, replace=False)
        cloud = cloud[idx]
    return cloud.astype(np.float32)


# ---- inference modes ----
@torch.no_grad()
def run_x0_single_step(model, point_dict, target_coords, t_val=200):
    """Single-step x0 prediction at t_val. Same as normal teacher inference."""
    model.eval()
    device = point_dict["coord"].device
    model.scheduler._to_device(device)

    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], target_coords)

    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(target_coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * target_coords + som * noise

    pred_noise = model.denoiser(noisy, target_coords, t_tensor, {"features": cond_features})
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0


# ---- variants ----
def variant_A_fair_ego_bbox(model, fr):
    """Duplicate 10x + jitter, crop to FIXED ego bbox (not GT), subsample 20k,
    single-step x0 at t=200. CD post-crop = GT+/-1m."""
    scaffold = scaffold_input_duplicated_fair(fr, n_dup=10, jitter=0.05, crop_mode="ego")
    if scaffold.shape[0] < 64:
        return None, None
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_x0_single_step(model, point_dict, target, t_val=200)
    return pred.cpu().numpy(), scaffold


def variant_A_fair_full(model, fr):
    """Duplicate 10x + jitter, NO crop on scaffold, subsample 20k,
    single-step x0 at t=200. CD post-crop = GT+/-1m."""
    scaffold = scaffold_input_duplicated_fair(fr, n_dup=10, jitter=0.05, crop_mode="none")
    if scaffold.shape[0] < 64:
        return None, None
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_x0_single_step(model, point_dict, target, t_val=200)
    return pred.cpu().numpy(), scaffold


def variant_A_fair_lidiff_match(model, fr):
    """Same as A_fair_ego_bbox, but CD is computed with the LiDiff KDTree
    implementation (direct apples-to-apples with 3.41)."""
    return variant_A_fair_ego_bbox(model, fr)


# CD POST-PROCESSING MODES per variant
VARIANTS = [
    # (name, inference_fn, cd_crop_mode, cd_impl)
    # cd_crop_mode:
    #   "lidiff"  -> crop prediction to gt_raw +/- 1.0m (world frame) before CD
    #   "none"    -> no crop (full prediction vs full GT)
    # cd_impl:
    #   "torch"   -> existing chunked torch CD (matches our past evaluations)
    #   "kdtree"  -> LiDiff-style scipy KDTree CD (direct match to 3.41)
    ("A_fair_ego_bbox_lidiff_crop",
     variant_A_fair_ego_bbox, "lidiff", "torch"),
    ("A_fair_full_scaffold_lidiff_crop",
     variant_A_fair_full, "lidiff", "torch"),
    ("A_fair_full_scaffold_no_crop",
     variant_A_fair_full, "none", "torch"),
    ("A_fair_lidiff_match_kdtree",
     variant_A_fair_ego_bbox, "lidiff", "kdtree"),
]


def cd_with_crop(pred_world, gt_world, crop_mode, cd_impl):
    """Compute CD with optional LiDiff-style GT-bbox post-hoc crop."""
    if crop_mode == "lidiff":
        bbox_min = gt_world.min(axis=0) - LIDIFF_MARGIN
        bbox_max = gt_world.max(axis=0) + LIDIFF_MARGIN
        mask = np.all((pred_world >= bbox_min) & (pred_world <= bbox_max), axis=1)
        pred_cropped = pred_world[mask]
        n_cropped = int(pred_cropped.shape[0])
    elif crop_mode == "none":
        pred_cropped = pred_world
        n_cropped = int(pred_cropped.shape[0])
    else:
        raise ValueError(f"unknown crop_mode: {crop_mode}")

    if cd_impl == "torch":
        cd = compute_cd_torch(pred_cropped, gt_world, max_pts=60000)
    elif cd_impl == "kdtree":
        cd = compute_cd_lidiff_kdtree(pred_cropped, gt_world, max_pts=60000)
    else:
        raise ValueError(f"unknown cd_impl: {cd_impl}")
    return cd, n_cropped


def run_variant_all_frames(model, frames, name, fn, crop_mode, cd_impl):
    print(f"\n{'='*70}\n  Variant: {name}  (crop={crop_mode}, cd={cd_impl})\n{'='*70}")
    t0 = time.time()
    pred_cds = []
    scaffold_cds = []
    n_cropped_list = []
    per_frame_times = []
    for i, fr in enumerate(frames):
        tf0 = time.time()
        try:
            pred_centered, scaffold_centered = fn(model, fr)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [warn] frame {fr['name']}: inference failed: {e}")
            continue
        if pred_centered is None:
            print(f"  [warn] frame {fr['name']}: invalid scaffold")
            continue

        per_frame_times.append(time.time() - tf0)
        pred_world = pred_centered + fr["lidar_center"]
        scaffold_world = scaffold_centered + fr["lidar_center"]
        gt_raw = fr["gt_raw"]

        try:
            cd_pred, n_cropped = cd_with_crop(pred_world, gt_raw, crop_mode, cd_impl)
            cd_scaffold, _ = cd_with_crop(scaffold_world, gt_raw, crop_mode, cd_impl)
        except Exception as e:
            print(f"  [warn] frame {fr['name']}: CD failed: {e}")
            continue

        if not np.isfinite(cd_pred):
            cd_pred = float("inf")
        pred_cds.append(cd_pred)
        scaffold_cds.append(cd_scaffold)
        n_cropped_list.append(n_cropped)

        if (i + 1) % 10 == 0 or (i + 1) == len(frames):
            finite_so_far = [c for c in pred_cds if np.isfinite(c)]
            mean_cd = np.mean(finite_so_far) if finite_so_far else float("nan")
            print(f"  [{i+1}/{len(frames)}] pred_cd={mean_cd:.4f}  "
                  f"scaffold_cd={np.mean(scaffold_cds):.4f}  "
                  f"n_after_crop={np.mean(n_cropped_list):.0f}  "
                  f"per-frame={np.mean(per_frame_times):.2f}s")

    elapsed = time.time() - t0
    finite = [c for c in pred_cds if np.isfinite(c)]
    if not finite:
        result = {"error": "no finite CDs", "n_frames": len(pred_cds), "time_s": round(elapsed, 1)}
    else:
        result = {
            "pred_cd_mean": float(np.mean(finite)),
            "pred_cd_std": float(np.std(finite)),
            "pred_cd_median": float(np.median(finite)),
            "pred_cd_min": float(np.min(finite)),
            "pred_cd_max": float(np.max(finite)),
            "scaffold_cd_mean": float(np.mean(scaffold_cds)),
            "scaffold_cd_std": float(np.std(scaffold_cds)),
            "n_frames": len(finite),
            "n_inf": len(pred_cds) - len(finite),
            "n_after_crop_mean": float(np.mean(n_cropped_list)) if n_cropped_list else 0.0,
            "per_frame_time_s_mean": float(np.mean(per_frame_times)),
            "total_time_s": round(elapsed, 1),
            "crop_mode": crop_mode,
            "cd_impl": cd_impl,
        }
    print(f"  -> {result}")
    return result


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  FAIR SCAFFOLD-FREE TEACHER EVALUATION (apr17_morning)")
    print("  No GT-bbox info leak on scaffold input.")
    print("=" * 70)
    print(f"  results -> {RESULTS_PATH}")
    print(f"  ckpt     -> {CKPT_PATH}")
    print(f"  frames   -> {NUM_FRAMES}")
    print(f"  ego bbox -> {EGO_BBOX_MIN} .. {EGO_BBOX_MAX}")
    print(f"  CD crop  -> GT +/- {LIDIFF_MARGIN}m (LiDiff protocol)")
    print()

    model = build_model()
    model = load_ckpt(model, CKPT_PATH)
    model.eval()
    frames = load_frames(NUM_FRAMES)

    results = {"variants": {}}

    for name, fn, crop_mode, cd_impl in VARIANTS:
        # reseed per-variant so the duplications/jitter are reproducible
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        results["variants"][name] = run_variant_all_frames(
            model, frames, name, fn, crop_mode, cd_impl,
        )
        payload = {
            **results,
            "num_frames": NUM_FRAMES,
            "checkpoint": str(CKPT_PATH),
            "seed": SEED,
            "partial": True,
            "ego_bbox_min": EGO_BBOX_MIN.tolist(),
            "ego_bbox_max": EGO_BBOX_MAX.tolist(),
            "lidiff_margin_m": LIDIFF_MARGIN,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  [persist] {RESULTS_PATH}")

    # final summary
    print("\n" + "=" * 70)
    print(f"{'Variant':<52} {'pred_cd':<22} {'scaffold_cd':<22}")
    print("-" * 100)
    for name, r in results["variants"].items():
        if "error" in r:
            print(f"{name:<52} ERROR: {r['error']}")
            continue
        pc = f"{r['pred_cd_mean']:.4f} +/- {r['pred_cd_std']:.4f}"
        sc = f"{r['scaffold_cd_mean']:.4f} +/- {r['scaffold_cd_std']:.4f}"
        print(f"{name:<52} {pc:<22} {sc:<22}")
    print()
    print("Reference: LiDiff on v2GT = 3.41, ScoreLiDAR on v2GT = 3.50")
    print("Old UNFAIR variant A (GT-bbox leak) = 2.37")

    results["partial"] = False
    results["num_frames"] = NUM_FRAMES
    results["checkpoint"] = str(CKPT_PATH)
    results["seed"] = SEED
    results["ego_bbox_min"] = EGO_BBOX_MIN.tolist()
    results["ego_bbox_max"] = EGO_BBOX_MAX.tolist()
    results["lidiff_margin_m"] = LIDIFF_MARGIN
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
