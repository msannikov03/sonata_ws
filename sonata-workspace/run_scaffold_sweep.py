#!/usr/bin/env python3
"""
Scaffold-Quality Sweep — Step 2 of apr17_morning experiment queue.

Tests teacher v2GT across many scaffold qualities on 50 frames of seq 08.

Scaffold variants:
  - GT scaffold (baseline)
  - GT + Gaussian jitter (sigma = 0.01, 0.02, 0.05, 0.10, 0.20 m)
  - GT + random dropout (10%, 25%, 50%, 75%)
  - Single-scan input as scaffold (LiDAR)
  - 5-frame aggregated scan scaffold (SKIPPED — using single-scan variant)
  - Voxel grid scaffold (res 0.1m, 0.2m)

For each scaffold variant we report:
  - CD of model output vs v2 GT
  - CD of the scaffold itself vs v2 GT (as an "X-axis" of scaffold quality)

Results -> results/apr17_morning/scaffold_quality_sweep.json
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
RESULTS_PATH = RESULTS_DIR / "scaffold_quality_sweep.json"
NUM_FRAMES = 50
DEVICE = "cuda"
MAX_PTS = 20000
SEED = 42


# ---- helpers ----
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
            "gt_coords_lidar": d["gt_coords_lidar"],
            "gt_raw": d["gt_raw"],
        })
    print(f"[data] loaded {len(frames)} frames from {PREVOX_DIR}")
    return frames


@torch.no_grad()
def run_x0_single_step(model, point_dict, target_coords, t_val=200):
    """Single-step x0 prediction using scaffold = target_coords."""
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


def to_target(coords_np):
    return torch.from_numpy(coords_np.astype(np.float32)).to(DEVICE)


# ---- scaffold constructors (all in centered coord frame) ----
def scaffold_gt(fr):
    return fr["gt_coords_lidar"].copy()


def scaffold_gt_jitter(fr, sigma):
    s = fr["gt_coords_lidar"].copy()
    s = s + np.random.normal(0.0, sigma, size=s.shape).astype(np.float32)
    return s


def scaffold_gt_dropout(fr, drop_frac):
    s = fr["gt_coords_lidar"].copy()
    n = s.shape[0]
    keep = int(n * (1.0 - drop_frac))
    keep = max(keep, 64)
    idx = np.random.choice(n, keep, replace=False)
    return s[idx]


def scaffold_input_lidar(fr):
    # lidar_coords is already centered on lidar mean, same frame as gt_coords_lidar
    return fr["lidar_coords"].copy()


def scaffold_voxel_grid(fr, res, max_pts=MAX_PTS):
    lidar = fr["lidar_coords"]
    bbox_min = lidar.min(axis=0) - 5.0
    bbox_max = lidar.max(axis=0) + 5.0
    xs = np.arange(bbox_min[0], bbox_max[0], res)
    ys = np.arange(bbox_min[1], bbox_max[1], res)
    zs = np.arange(bbox_min[2], bbox_max[2], res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    if grid.shape[0] > max_pts:
        idx = np.random.choice(grid.shape[0], max_pts, replace=False)
        grid = grid[idx]
    return grid


# ---- main sweep ----
def run_variant(model, frames, name, scaffold_fn):
    """Run one scaffold variant across all frames. Returns stats dict."""
    print(f"\n{'='*70}\n  Variant: {name}\n{'='*70}")
    t0 = time.time()
    pred_cds = []
    scaffold_cds = []
    for i, fr in enumerate(frames):
        try:
            scaffold_centered = scaffold_fn(fr)
        except Exception as e:
            print(f"  [warn] frame {fr['name']}: scaffold build failed: {e}")
            continue

        if scaffold_centered.shape[0] < 64:
            print(f"  [warn] frame {fr['name']}: scaffold too small ({scaffold_centered.shape[0]})")
            continue

        # subsample scaffold if too large (encoder can't handle too many target pts)
        if scaffold_centered.shape[0] > MAX_PTS:
            idx = np.random.choice(scaffold_centered.shape[0], MAX_PTS, replace=False)
            scaffold_centered = scaffold_centered[idx]

        point_dict = make_point_dict(fr["lidar_coords"])
        target = to_target(scaffold_centered)

        try:
            pred = run_x0_single_step(model, point_dict, target, t_val=200)
        except Exception as e:
            print(f"  [warn] frame {fr['name']}: inference failed: {e}")
            continue

        pred_world = pred.cpu().numpy() + fr["lidar_center"]
        gt_raw = fr["gt_raw"]

        # scaffold CD (scaffold needs to be uncentered to compare against gt_raw)
        scaffold_world = scaffold_centered + fr["lidar_center"]

        try:
            cd_pred = compute_cd(pred_world, gt_raw)
            cd_scaffold = compute_cd(scaffold_world, gt_raw)
        except Exception as e:
            print(f"  [warn] frame {fr['name']}: CD failed: {e}")
            continue

        pred_cds.append(cd_pred)
        scaffold_cds.append(cd_scaffold)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] pred_cd={np.mean(pred_cds):.4f}  scaffold_cd={np.mean(scaffold_cds):.4f}")

    elapsed = time.time() - t0
    if not pred_cds:
        return {"error": "no valid frames", "time_s": round(elapsed, 1)}

    result = {
        "pred_cd_mean": float(np.mean(pred_cds)),
        "pred_cd_std": float(np.std(pred_cds)),
        "scaffold_cd_mean": float(np.mean(scaffold_cds)),
        "scaffold_cd_std": float(np.std(scaffold_cds)),
        "n_frames": len(pred_cds),
        "time_s": round(elapsed, 1),
    }
    print(f"  -> pred_cd = {result['pred_cd_mean']:.4f} +/- {result['pred_cd_std']:.4f}")
    print(f"     scaffold_cd = {result['scaffold_cd_mean']:.4f} +/- {result['scaffold_cd_std']:.4f}")
    print(f"     time: {elapsed:.1f}s")
    return result


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SCAFFOLD-QUALITY SWEEP")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Frames: {NUM_FRAMES} from seq 08 (prevoxelized)")
    print(f"  Results: {RESULTS_PATH}")
    print("=" * 70)

    model = load_ckpt(build_model(DEVICE), str(CKPT_PATH), DEVICE)
    model.eval()

    frames = load_frames(NUM_FRAMES)

    variants = OrderedDict()
    variants["gt"] = lambda fr: scaffold_gt(fr)
    # gaussian jitter
    for sigma in [0.01, 0.02, 0.05, 0.10, 0.20]:
        variants[f"gt_jitter_sigma_{sigma:.2f}"] = (lambda s: lambda fr: scaffold_gt_jitter(fr, s))(sigma)
    # dropout
    for drop in [0.10, 0.25, 0.50, 0.75]:
        variants[f"gt_dropout_{int(drop*100):02d}pct"] = (lambda d: lambda fr: scaffold_gt_dropout(fr, d))(drop)
    # single-scan lidar
    variants["single_scan_lidar"] = lambda fr: scaffold_input_lidar(fr)
    # voxel grid
    for res in [0.1, 0.2]:
        variants[f"voxel_grid_{res:.1f}m"] = (lambda r: lambda fr: scaffold_voxel_grid(fr, r))(res)

    all_results = OrderedDict()
    total_t0 = time.time()
    for name, fn in variants.items():
        try:
            all_results[name] = run_variant(model, frames, name, fn)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}
        # save incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "variants": all_results,
                "num_frames": len(frames),
                "checkpoint": str(CKPT_PATH),
                "seed": SEED,
                "partial": True,
            }, f, indent=2)

    total_elapsed = time.time() - total_t0

    # final table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<32} {'pred_cd':<18} {'scaffold_cd':<18}")
    print("-" * 70)
    for name, r in all_results.items():
        if "error" in r:
            print(f"{name:<32} ERROR: {r['error']}")
            continue
        pcd = f"{r['pred_cd_mean']:.4f} +/- {r['pred_cd_std']:.4f}"
        scd = f"{r['scaffold_cd_mean']:.4f} +/- {r['scaffold_cd_std']:.4f}"
        print(f"{name:<32} {pcd:<18} {scd:<18}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "variants": all_results,
            "num_frames": len(frames),
            "checkpoint": str(CKPT_PATH),
            "seed": SEED,
            "total_time_s": round(total_elapsed, 1),
            "partial": False,
        }, f, indent=2)
    print(f"\n[saved] {RESULTS_PATH}")


if __name__ == "__main__":
    main()
