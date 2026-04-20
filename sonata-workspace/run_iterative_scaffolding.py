#!/usr/bin/env python3
"""
Iterative Self-Scaffolding — Step 4 of apr17_morning experiment queue.

Start with a bad scaffold (voxel grid 0.2m) and iteratively use the model
output as the new scaffold, tracking CD convergence.

Two variants:
  A: scaffold_{k+1} = voxelize(pred_k, 0.05m)
  B: scaffold_{k+1} = 0.7 * pred_k + 0.3 * scaffold_k (point-aligned)

Iteration counts probed: 1, 2, 3, 5, 10

Results -> results/apr17_morning/iterative_scaffolding.json
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
RESULTS_PATH = RESULTS_DIR / "iterative_scaffolding.json"
NUM_FRAMES = 30
DEVICE = "cuda"
MAX_PTS = 20000
MAX_ITER = 10
TRACK_ITERS = [1, 2, 3, 5, 10]
INIT_VOXEL_RES = 0.2
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


def init_voxel_grid(lidar_coords, res, max_pts=MAX_PTS):
    bbox_min = lidar_coords.min(axis=0) - 5.0
    bbox_max = lidar_coords.max(axis=0) + 5.0
    xs = np.arange(bbox_min[0], bbox_max[0], res)
    ys = np.arange(bbox_min[1], bbox_max[1], res)
    zs = np.arange(bbox_min[2], bbox_max[2], res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    if grid.shape[0] > max_pts:
        idx = np.random.choice(grid.shape[0], max_pts, replace=False)
        grid = grid[idx]
    return grid


def iterate_frame(model, fr, variant, max_iter=MAX_ITER):
    """Run iterative self-scaffolding, return CD at every iteration."""
    point_dict = make_point_dict(fr["lidar_coords"])
    center = fr["lidar_center"]
    gt_raw = fr["gt_raw"]

    # initial bad scaffold: voxel grid 0.2m
    scaffold = init_voxel_grid(fr["lidar_coords"], INIT_VOXEL_RES, MAX_PTS)

    cds = []
    # scaffold CD for iteration 0 (before any model step)
    scaffold_cd_0 = compute_cd(scaffold + center, gt_raw)

    prev_scaffold = scaffold.copy()
    for it in range(1, max_iter + 1):
        target = torch.from_numpy(scaffold).float().to(DEVICE)
        try:
            pred = run_x0(model, point_dict, target)
        except Exception as e:
            print(f"  [warn] {fr['name']} iter {it}: {e}")
            # fill remaining with nan
            while len(cds) < max_iter:
                cds.append(float("nan"))
            break

        pred_np = pred.cpu().numpy()
        pred_world = pred_np + center
        cd = compute_cd(pred_world, gt_raw)
        cds.append(cd)

        # form next scaffold
        if variant == "A":
            # voxelize output at REVOX_RES
            new_scaffold = voxelize(pred_np, REVOX_RES, MAX_PTS)
        else:  # variant B
            # 0.7 pred + 0.3 prev_scaffold (point-aligned because same size)
            if pred_np.shape == prev_scaffold.shape:
                new_scaffold = 0.7 * pred_np + 0.3 * prev_scaffold
            else:
                # different sizes from voxelize; just use pred
                new_scaffold = pred_np

        prev_scaffold = scaffold.copy()
        scaffold = new_scaffold.astype(np.float32)
        if scaffold.shape[0] > MAX_PTS:
            idx = np.random.choice(scaffold.shape[0], MAX_PTS, replace=False)
            scaffold = scaffold[idx]

    return scaffold_cd_0, cds


def run_variant(model, frames, variant):
    print(f"\n{'='*70}\n  Variant {variant}: {'voxelize output at 0.05m' if variant=='A' else '0.7 pred + 0.3 prev_scaffold'}\n{'='*70}")
    t0 = time.time()
    init_cds = []
    iter_cds = []
    for fi, fr in enumerate(frames):
        init_cd, cds = iterate_frame(model, fr, variant, MAX_ITER)
        init_cds.append(init_cd)
        iter_cds.append(cds)
        if (fi + 1) % 5 == 0:
            arr = np.array([c for c in iter_cds if len(c) == MAX_ITER])
            if len(arr):
                tail = arr.mean(axis=0)
                print(f"  [{fi+1}/{len(frames)}] init_cd={np.mean(init_cds):.3f}  iter1={tail[0]:.3f}  iter3={tail[2]:.3f}  iter10={tail[-1]:.3f}")

    elapsed = time.time() - t0
    arr = np.array([c for c in iter_cds if len(c) == MAX_ITER])
    result = {
        "variant": variant,
        "n_frames": len(iter_cds),
        "n_frames_valid": int(len(arr)),
        "init_scaffold_cd_mean": float(np.mean(init_cds)),
        "init_scaffold_cd_std": float(np.std(init_cds)),
        "time_s": round(elapsed, 1),
    }
    if len(arr):
        for k in TRACK_ITERS:
            result[f"iter{k}_cd_mean"] = float(arr[:, k - 1].mean())
            result[f"iter{k}_cd_std"] = float(arr[:, k - 1].std())
        # full per-iter trace
        result["per_iter_mean"] = [float(arr[:, i].mean()) for i in range(MAX_ITER)]
        result["per_iter_std"] = [float(arr[:, i].std()) for i in range(MAX_ITER)]

    print(f"  init_cd = {result['init_scaffold_cd_mean']:.4f}")
    for k in TRACK_ITERS:
        if f"iter{k}_cd_mean" in result:
            print(f"  iter {k:2d}: CD = {result[f'iter{k}_cd_mean']:.4f} +/- {result[f'iter{k}_cd_std']:.4f}")
    print(f"  time: {elapsed:.1f}s")
    return result


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ITERATIVE SELF-SCAFFOLDING")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Frames: {NUM_FRAMES}, Max iters: {MAX_ITER}")
    print("=" * 70)

    model = load_ckpt(build_model(DEVICE), str(CKPT_PATH), DEVICE)
    model.eval()
    frames = load_frames(NUM_FRAMES)

    all_results = OrderedDict()
    total_t0 = time.time()
    for variant in ["A", "B"]:
        try:
            all_results[f"variant_{variant}"] = run_variant(model, frames, variant)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[f"variant_{variant}"] = {"error": str(e)}
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "results": all_results,
                "num_frames": NUM_FRAMES,
                "max_iter": MAX_ITER,
                "track_iters": TRACK_ITERS,
                "init_voxel_res": INIT_VOXEL_RES,
                "revox_res": REVOX_RES,
                "partial": True,
            }, f, indent=2)

    total_elapsed = time.time() - total_t0
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "results": all_results,
            "num_frames": NUM_FRAMES,
            "max_iter": MAX_ITER,
            "track_iters": TRACK_ITERS,
            "init_voxel_res": INIT_VOXEL_RES,
            "revox_res": REVOX_RES,
            "total_time_s": round(total_elapsed, 1),
            "partial": False,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for variant, r in all_results.items():
        if "error" in r:
            print(f"{variant}: ERROR")
            continue
        print(f"{variant}: init={r['init_scaffold_cd_mean']:.3f} " +
              " ".join(f"iter{k}={r.get(f'iter{k}_cd_mean', float('nan')):.3f}" for k in TRACK_ITERS))
    print(f"\n[saved] {RESULTS_PATH}")


if __name__ == "__main__":
    main()
