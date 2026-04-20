#!/usr/bin/env python3
"""
Iterative Self-Scaffolding V2 — diagnostic for apr17 iterative scaffolding failure.

The v1 experiment (results/apr17_morning/iterative_scaffolding.json) diverged
from a voxel-grid starting scaffold (CD 844 -> 1067 over 10 iters). Hypothesis:
the voxel grid's spatial extent (~85x75x30m) vastly exceeds the v2 GT bbox
(~16x19x4.5m), so the denoiser's predictions also span the full extent and
never converge onto the GT region.

This script probes four starting scaffolds on the same teacher checkpoint:
  1) GT + N(0, sigma=0.20m)      -- noisy but bbox-matched (from sweep: scaffold CD 0.035, pred CD 0.027)
  2) GT with 50% dropout          -- sparse but bbox-matched (scaffold CD 0.018, pred CD 0.022)
  3) GT + N(0, sigma=0.50m)      -- heavy noise, bbox-matched (stress test)
  4) Voxel grid CROPPED to v2 GT bbox -- tests if extent mismatch is the killer

For each start, run 5 iterations of Variant A (use pred as next scaffold, voxelize at 0.05m).

Results -> results/apr17_morning/iterative_scaffolding_v2.json
"""

import os, sys, time, json
from pathlib import Path
from collections import OrderedDict

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
RESULTS_PATH = RESULTS_DIR / "iterative_scaffolding_v2.json"
NUM_FRAMES = 20
DEVICE = "cuda"
MAX_PTS = 20000
MAX_ITER = 5
REVOX_RES = 0.05
SEED = 42


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


def load_frames(n=NUM_FRAMES):
    files = sorted(PREVOX_DIR.glob("*.npz"))[:n]
    frames = []
    for f in files:
        d = np.load(f)
        frames.append({
            "name": f.stem,
            "lidar_coords": d["lidar_coords"],
            "lidar_center": d["lidar_center"],
            "gt_raw": d["gt_raw"],
        })
    print(f"[data] loaded {len(frames)} frames")
    return frames


@torch.no_grad()
def run_x0(model, point_dict, target_coords, t_val=200):
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


def voxelize(points_np, res, max_pts=MAX_PTS):
    vc = np.floor(points_np / res).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    out = points_np[idx]
    if out.shape[0] > max_pts:
        sel = np.random.choice(out.shape[0], max_pts, replace=False)
        out = out[sel]
    return out.astype(np.float32)


# ---- starting scaffold builders ----

def make_start_gt_jitter(fr, sigma, rng):
    """GT in centered coords + Gaussian jitter."""
    gt_c = fr["gt_raw"] - fr["lidar_center"]
    scaffold = gt_c + rng.normal(0, sigma, gt_c.shape).astype(np.float32)
    if scaffold.shape[0] > MAX_PTS:
        idx = rng.choice(scaffold.shape[0], MAX_PTS, replace=False)
        scaffold = scaffold[idx]
    return scaffold.astype(np.float32)


def make_start_gt_dropout(fr, keep_frac, rng):
    """Keep a fraction of GT points in centered coords."""
    gt_c = fr["gt_raw"] - fr["lidar_center"]
    n_keep = int(gt_c.shape[0] * keep_frac)
    idx = rng.choice(gt_c.shape[0], n_keep, replace=False)
    scaffold = gt_c[idx]
    if scaffold.shape[0] > MAX_PTS:
        idx2 = rng.choice(scaffold.shape[0], MAX_PTS, replace=False)
        scaffold = scaffold[idx2]
    return scaffold.astype(np.float32)


def make_start_voxel_cropped(fr, res, rng, margin=1.0):
    """Voxel grid cropped to GT bbox (in centered coords) with small margin."""
    gt_c = fr["gt_raw"] - fr["lidar_center"]
    bbox_min = gt_c.min(axis=0) - margin
    bbox_max = gt_c.max(axis=0) + margin
    xs = np.arange(bbox_min[0], bbox_max[0], res)
    ys = np.arange(bbox_min[1], bbox_max[1], res)
    zs = np.arange(bbox_min[2], bbox_max[2], res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    if grid.shape[0] > MAX_PTS:
        idx = rng.choice(grid.shape[0], MAX_PTS, replace=False)
        grid = grid[idx]
    return grid.astype(np.float32)


def iterate_frame(model, fr, start_fn, rng, max_iter=MAX_ITER):
    """Run iterative self-scaffolding variant A from a custom start scaffold."""
    point_dict = make_point_dict(fr["lidar_coords"])
    center = fr["lidar_center"]
    gt_raw = fr["gt_raw"]

    scaffold = start_fn(fr, rng)
    scaffold_cd_0 = compute_cd(scaffold + center, gt_raw)

    cds = []
    for it in range(1, max_iter + 1):
        target = torch.from_numpy(scaffold).float().to(DEVICE)
        try:
            pred = run_x0(model, point_dict, target)
        except Exception as e:
            print(f"  [warn] {fr['name']} iter {it}: {e}")
            while len(cds) < max_iter:
                cds.append(float("nan"))
            break

        pred_np = pred.cpu().numpy()
        cd = compute_cd(pred_np + center, gt_raw)
        cds.append(cd)

        # Variant A: voxelize at 0.05m
        new_scaffold = voxelize(pred_np, REVOX_RES, MAX_PTS)
        scaffold = new_scaffold.astype(np.float32)

    return scaffold_cd_0, cds


def run_start_variant(model, frames, name, start_fn):
    print(f"\n{'='*70}\n  Start variant: {name}\n{'='*70}")
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    init_cds = []
    iter_cds = []
    for fi, fr in enumerate(frames):
        init_cd, cds = iterate_frame(model, fr, start_fn, rng, MAX_ITER)
        init_cds.append(init_cd)
        iter_cds.append(cds)

    elapsed = time.time() - t0
    arr = np.array([c for c in iter_cds if len(c) == MAX_ITER and not any(np.isnan(c))])
    result = {
        "name": name,
        "n_frames": len(iter_cds),
        "n_frames_valid": int(len(arr)),
        "init_scaffold_cd_mean": float(np.mean(init_cds)),
        "init_scaffold_cd_std": float(np.std(init_cds)),
        "time_s": round(elapsed, 1),
    }
    if len(arr):
        result["per_iter_mean"] = [float(arr[:, i].mean()) for i in range(MAX_ITER)]
        result["per_iter_std"] = [float(arr[:, i].std()) for i in range(MAX_ITER)]

    print(f"  init_cd = {result['init_scaffold_cd_mean']:.4f}")
    if "per_iter_mean" in result:
        for i, (m, s) in enumerate(zip(result["per_iter_mean"], result["per_iter_std"])):
            print(f"  iter {i+1:2d}: CD = {m:.4f} +/- {s:.4f}")
    print(f"  time: {elapsed:.1f}s")
    return result


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ITERATIVE SELF-SCAFFOLDING V2")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Frames: {NUM_FRAMES}, Max iters: {MAX_ITER}")
    print("=" * 70)

    model = load_ckpt(build_model(DEVICE), str(CKPT_PATH), DEVICE)
    model.eval()
    frames = load_frames(NUM_FRAMES)

    start_variants = OrderedDict([
        ("gt_jitter_sigma_0.20", lambda fr, rng: make_start_gt_jitter(fr, 0.20, rng)),
        ("gt_dropout_50pct",    lambda fr, rng: make_start_gt_dropout(fr, 0.50, rng)),
        ("gt_jitter_sigma_0.50", lambda fr, rng: make_start_gt_jitter(fr, 0.50, rng)),
        ("voxel_cropped_0.2m_bbox_margin_1.0m",
            lambda fr, rng: make_start_voxel_cropped(fr, 0.2, rng, margin=1.0)),
    ])

    all_results = OrderedDict()
    total_t0 = time.time()
    for name, fn in start_variants.items():
        try:
            all_results[name] = run_start_variant(model, frames, name, fn)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "results": all_results,
                "num_frames": NUM_FRAMES,
                "max_iter": MAX_ITER,
                "revox_res": REVOX_RES,
                "partial": True,
            }, f, indent=2)

    total_elapsed = time.time() - total_t0
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "results": all_results,
            "num_frames": NUM_FRAMES,
            "max_iter": MAX_ITER,
            "revox_res": REVOX_RES,
            "total_time_s": round(total_elapsed, 1),
            "partial": False,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for name, r in all_results.items():
        if "error" in r:
            print(f"{name}: ERROR")
            continue
        pm = r.get("per_iter_mean", [])
        trace = " -> ".join(f"{v:.4f}" for v in pm)
        print(f"{name}: init={r['init_scaffold_cd_mean']:.4f} | {trace}")
    print(f"\n[saved] {RESULTS_PATH}")


if __name__ == "__main__":
    main()
