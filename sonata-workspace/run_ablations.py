#!/usr/bin/env python3
"""
Ablation evaluation script for scene completion diffusion model.
Runs 6 ablation experiments sequentially on 50 frames from seq 08.

Ablations:
  1. Conditioning zero-out (does the encoder matter?)
  2. Timestep sweep (optimal noise level for single-step x0)
  3. Sparse scaffold (minimum GT density needed)
  4. Random scaffold (sanity check -- scaffold structure matters)
  5. Input density (robustness to input sparsity)
  6. Voxel grid scaffold (works without GT coords at inference?)

Usage:
  python run_ablations.py
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict

# ---- paths ----
WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR))

PREVOX_DIR = Path("/home/anywherevla/sonata_ws/prevoxelized_seq08")
CKPT_PATH = WORK_DIR / "checkpoints" / "diffusion_v2gt" / "best_model.pth"
RESULTS_PATH = WORK_DIR / "ablation_results.json"
NUM_FRAMES = 50
DEVICE = "cuda"

# ---- imports from codebase ----
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
    """Build encoder-compatible point_dict from (N, 3) centered coords."""
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
    """Symmetric mean-squared-L2 chamfer distance (same as evaluate_fixed.py)."""
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
    """Load first n prevoxelized frames, return list of dicts."""
    files = sorted(PREVOX_DIR.glob("*.npz"))[:n]
    frames = []
    for f in files:
        d = np.load(f)
        frames.append({
            "name": f.stem,
            "lidar_coords": d["lidar_coords"],      # (20k, 3) centered
            "lidar_center": d["lidar_center"],       # (3,)
            "gt_coords_lidar": d["gt_coords_lidar"], # (20k, 3) centered on lidar mean
            "gt_raw": d["gt_raw"],                   # (M, 3) world coords
        })
    print(f"[data] loaded {len(frames)} frames from {PREVOX_DIR}")
    return frames


@torch.no_grad()
def run_x0_single_step(model, point_dict, target_coords, t_val=200,
                        cond_features_override=None):
    """
    Single-step x0 prediction.

    If cond_features_override is provided, skip encoder + knn_interpolate
    and use those features directly (must already be mapped to target_coords).
    """
    model.eval()
    device = point_dict["coord"].device
    model.scheduler._to_device(device)

    if cond_features_override is not None:
        cond_features = cond_features_override
    else:
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


@torch.no_grad()
def get_cond_features(model, point_dict, target_coords):
    """Extract and interpolate conditioning features (cacheable)."""
    model.eval()
    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], target_coords)
    return cond_features


def print_table(rows, headers, title=""):
    """Pretty-print a results table."""
    col_widths = [max(len(str(r[i])) for r in rows + [headers]) + 2 for i in range(len(headers))]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
    print(sep)
    print("|" + "|".join(str(headers[i]).center(col_widths[i]) for i in range(len(headers))) + "|")
    print(sep)
    for r in rows:
        print("|" + "|".join(str(r[i]).center(col_widths[i]) for i in range(len(r))) + "|")
    print(sep)


# ---------------------------------------------------------------------------
# ablation runners
# ---------------------------------------------------------------------------

def ablation_1_conditioning_zeroout(model, frames):
    """Zero out conditioning features after knn_interpolate."""
    print("\n" + "=" * 70)
    print("ABLATION 1: Conditioning Zero-Out")
    print("=" * 70)
    t0 = time.time()
    cds_baseline = []
    cds_zeroed = []

    for i, fr in enumerate(frames):
        point_dict = make_point_dict(fr["lidar_coords"])
        gt_target = torch.from_numpy(fr["gt_coords_lidar"]).float().to(DEVICE)
        center = fr["lidar_center"]

        # baseline
        pred = run_x0_single_step(model, point_dict, gt_target, t_val=200)
        pred_np = pred.cpu().numpy() + center
        cd_base = compute_cd(pred_np, fr["gt_raw"])
        cds_baseline.append(cd_base)

        # zeroed conditioning
        cond = get_cond_features(model, point_dict, gt_target)
        zero_cond = torch.zeros_like(cond)
        pred_z = run_x0_single_step(model, point_dict, gt_target, t_val=200,
                                     cond_features_override=zero_cond)
        pred_z_np = pred_z.cpu().numpy() + center
        cd_zero = compute_cd(pred_z_np, fr["gt_raw"])
        cds_zeroed.append(cd_zero)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] baseline={np.mean(cds_baseline):.4f}  zeroed={np.mean(cds_zeroed):.4f}")

    elapsed = time.time() - t0
    result = {
        "baseline_cd_mean": float(np.mean(cds_baseline)),
        "baseline_cd_std": float(np.std(cds_baseline)),
        "zeroed_cd_mean": float(np.mean(cds_zeroed)),
        "zeroed_cd_std": float(np.std(cds_zeroed)),
        "time_s": round(elapsed, 1),
    }
    rows = [
        ["Baseline (normal)", f"{result['baseline_cd_mean']:.4f} +/- {result['baseline_cd_std']:.4f}"],
        ["Zero conditioning", f"{result['zeroed_cd_mean']:.4f} +/- {result['zeroed_cd_std']:.4f}"],
    ]
    print_table(rows, ["Condition", "CD"], "Ablation 1: Conditioning Zero-Out")
    print(f"  Time: {elapsed:.1f}s")
    return result


def ablation_2_timestep_sweep(model, frames):
    """Sweep over timesteps for single-step x0."""
    print("\n" + "=" * 70)
    print("ABLATION 2: Timestep Sweep")
    print("=" * 70)
    t0 = time.time()
    timesteps = [1, 10, 50, 100, 150, 200, 250, 300, 400, 500, 700, 999]
    all_cds = {t: [] for t in timesteps}

    for i, fr in enumerate(frames):
        point_dict = make_point_dict(fr["lidar_coords"])
        gt_target = torch.from_numpy(fr["gt_coords_lidar"]).float().to(DEVICE)
        center = fr["lidar_center"]

        # cache encoder features once per frame
        cond = get_cond_features(model, point_dict, gt_target)

        for t_val in timesteps:
            pred = run_x0_single_step(model, point_dict, gt_target, t_val=t_val,
                                       cond_features_override=cond)
            pred_np = pred.cpu().numpy() + center
            cd = compute_cd(pred_np, fr["gt_raw"])
            all_cds[t_val].append(cd)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] t=200 mean CD so far: {np.mean(all_cds[200]):.4f}")

    elapsed = time.time() - t0
    result = {}
    rows = []
    for t_val in timesteps:
        m = float(np.mean(all_cds[t_val]))
        s = float(np.std(all_cds[t_val]))
        result[f"t={t_val}"] = {"mean": m, "std": s}
        rows.append([str(t_val), f"{m:.4f}", f"{s:.4f}"])

    result["time_s"] = round(elapsed, 1)
    print_table(rows, ["Timestep", "CD Mean", "CD Std"], "Ablation 2: Timestep Sweep")
    print(f"  Time: {elapsed:.1f}s")
    return result


def ablation_3_sparse_scaffold(model, frames):
    """Subsample GT target coordinates."""
    print("\n" + "=" * 70)
    print("ABLATION 3: Sparse Scaffold")
    print("=" * 70)
    t0 = time.time()
    fractions = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    all_cds = {f: [] for f in fractions}

    for i, fr in enumerate(frames):
        point_dict = make_point_dict(fr["lidar_coords"])
        gt_target_full = torch.from_numpy(fr["gt_coords_lidar"]).float().to(DEVICE)
        center = fr["lidar_center"]
        gt_raw = fr["gt_raw"]
        n_full = gt_target_full.shape[0]

        for frac in fractions:
            n_keep = max(1, int(n_full * frac))
            if frac < 1.0:
                idx = np.random.choice(n_full, n_keep, replace=False)
                gt_sub = gt_target_full[idx]
            else:
                gt_sub = gt_target_full

            # must recompute cond features for subsampled coords
            cond = get_cond_features(model, point_dict, gt_sub)
            pred = run_x0_single_step(model, point_dict, gt_sub, t_val=200,
                                       cond_features_override=cond)
            pred_np = pred.cpu().numpy() + center
            cd = compute_cd(pred_np, gt_raw)
            all_cds[frac].append(cd)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] 100%={np.mean(all_cds[1.0]):.4f}  10%={np.mean(all_cds[0.1]):.4f}")

    elapsed = time.time() - t0
    result = {}
    rows = []
    for frac in fractions:
        n_pts = int(frames[0]["gt_coords_lidar"].shape[0] * frac)
        m = float(np.mean(all_cds[frac]))
        s = float(np.std(all_cds[frac]))
        result[f"{int(frac*100)}%"] = {"mean": m, "std": s, "n_points": n_pts}
        rows.append([f"{int(frac*100)}%", str(n_pts), f"{m:.4f}", f"{s:.4f}"])

    result["time_s"] = round(elapsed, 1)
    print_table(rows, ["Fraction", "Points", "CD Mean", "CD Std"], "Ablation 3: Sparse Scaffold")
    print(f"  Time: {elapsed:.1f}s")
    return result


def ablation_4_random_scaffold(model, frames):
    """Replace GT scaffold with uniform random points in scene bbox."""
    print("\n" + "=" * 70)
    print("ABLATION 4: Random Scaffold (sanity check)")
    print("=" * 70)
    t0 = time.time()
    cds_gt = []
    cds_random = []

    for i, fr in enumerate(frames):
        point_dict = make_point_dict(fr["lidar_coords"])
        gt_target = torch.from_numpy(fr["gt_coords_lidar"]).float().to(DEVICE)
        center = fr["lidar_center"]
        gt_raw = fr["gt_raw"]
        n_pts = gt_target.shape[0]

        # baseline with GT scaffold
        cond = get_cond_features(model, point_dict, gt_target)
        pred = run_x0_single_step(model, point_dict, gt_target, t_val=200,
                                   cond_features_override=cond)
        pred_np = pred.cpu().numpy() + center
        cd_gt = compute_cd(pred_np, gt_raw)
        cds_gt.append(cd_gt)

        # random scaffold: uniform in lidar bbox + 10m margin (centered coords)
        lidar_coords = fr["lidar_coords"]
        bbox_min = lidar_coords.min(axis=0) - 10.0
        bbox_max = lidar_coords.max(axis=0) + 10.0
        rand_coords = np.random.uniform(bbox_min, bbox_max, size=(n_pts, 3)).astype(np.float32)
        rand_target = torch.from_numpy(rand_coords).float().to(DEVICE)

        cond_rand = get_cond_features(model, point_dict, rand_target)
        pred_rand = run_x0_single_step(model, point_dict, rand_target, t_val=200,
                                        cond_features_override=cond_rand)
        pred_rand_np = pred_rand.cpu().numpy() + center
        cd_rand = compute_cd(pred_rand_np, gt_raw)
        cds_random.append(cd_rand)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] GT={np.mean(cds_gt):.4f}  random={np.mean(cds_random):.4f}")

    elapsed = time.time() - t0
    result = {
        "gt_scaffold_cd_mean": float(np.mean(cds_gt)),
        "gt_scaffold_cd_std": float(np.std(cds_gt)),
        "random_scaffold_cd_mean": float(np.mean(cds_random)),
        "random_scaffold_cd_std": float(np.std(cds_random)),
        "time_s": round(elapsed, 1),
    }
    rows = [
        ["GT scaffold", f"{result['gt_scaffold_cd_mean']:.4f} +/- {result['gt_scaffold_cd_std']:.4f}"],
        ["Random scaffold", f"{result['random_scaffold_cd_mean']:.4f} +/- {result['random_scaffold_cd_std']:.4f}"],
    ]
    print_table(rows, ["Scaffold", "CD"], "Ablation 4: Random Scaffold")
    print(f"  Time: {elapsed:.1f}s")
    return result


def ablation_5_input_density(model, frames):
    """Subsample input LiDAR scan before encoding."""
    print("\n" + "=" * 70)
    print("ABLATION 5: Input Density")
    print("=" * 70)
    t0 = time.time()
    fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
    all_cds = {f: [] for f in fractions}

    for i, fr in enumerate(frames):
        gt_target = torch.from_numpy(fr["gt_coords_lidar"]).float().to(DEVICE)
        center = fr["lidar_center"]
        gt_raw = fr["gt_raw"]
        lidar_full = fr["lidar_coords"]
        n_full = lidar_full.shape[0]

        for frac in fractions:
            n_keep = max(100, int(n_full * frac))
            if frac < 1.0:
                idx = np.random.choice(n_full, n_keep, replace=False)
                lidar_sub = lidar_full[idx]
            else:
                lidar_sub = lidar_full

            point_dict = make_point_dict(lidar_sub)
            cond = get_cond_features(model, point_dict, gt_target)
            pred = run_x0_single_step(model, point_dict, gt_target, t_val=200,
                                       cond_features_override=cond)
            pred_np = pred.cpu().numpy() + center
            cd = compute_cd(pred_np, gt_raw)
            all_cds[frac].append(cd)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] 100%={np.mean(all_cds[1.0]):.4f}  25%={np.mean(all_cds[0.25]):.4f}")

    elapsed = time.time() - t0
    result = {}
    rows = []
    for frac in fractions:
        n_pts = int(frames[0]["lidar_coords"].shape[0] * frac)
        m = float(np.mean(all_cds[frac]))
        s = float(np.std(all_cds[frac]))
        result[f"{int(frac*100)}%"] = {"mean": m, "std": s, "n_points": n_pts}
        rows.append([f"{int(frac*100)}%", str(n_pts), f"{m:.4f}", f"{s:.4f}"])

    result["time_s"] = round(elapsed, 1)
    print_table(rows, ["Input %", "Points", "CD Mean", "CD Std"], "Ablation 5: Input Density")
    print(f"  Time: {elapsed:.1f}s")
    return result


def ablation_6_voxel_grid_scaffold(model, frames):
    """Replace GT scaffold with regular 3D voxel grid."""
    print("\n" + "=" * 70)
    print("ABLATION 6: Voxel Grid Scaffold")
    print("=" * 70)
    t0 = time.time()
    resolutions = [0.5, 1.0, 2.0]
    max_grid_pts = 20000
    all_cds = {r: [] for r in resolutions}

    for i, fr in enumerate(frames):
        point_dict = make_point_dict(fr["lidar_coords"])
        center = fr["lidar_center"]
        gt_raw = fr["gt_raw"]

        # bbox from lidar scan (centered) + 5m margin
        lidar_coords = fr["lidar_coords"]
        bbox_min = lidar_coords.min(axis=0) - 5.0
        bbox_max = lidar_coords.max(axis=0) + 5.0

        for res in resolutions:
            # build regular grid
            xs = np.arange(bbox_min[0], bbox_max[0], res)
            ys = np.arange(bbox_min[1], bbox_max[1], res)
            zs = np.arange(bbox_min[2], bbox_max[2], res)
            grid = np.stack(
                np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1
            ).reshape(-1, 3).astype(np.float32)

            # subsample if too many points
            if grid.shape[0] > max_grid_pts:
                idx = np.random.choice(grid.shape[0], max_grid_pts, replace=False)
                grid = grid[idx]

            grid_target = torch.from_numpy(grid).float().to(DEVICE)

            try:
                cond = get_cond_features(model, point_dict, grid_target)
                pred = run_x0_single_step(model, point_dict, grid_target, t_val=200,
                                           cond_features_override=cond)
                pred_np = pred.cpu().numpy() + center
                cd = compute_cd(pred_np, gt_raw)
            except Exception as e:
                print(f"    [warn] frame {fr['name']} res={res}: {e}")
                cd = float("nan")

            all_cds[res].append(cd)

        if (i + 1) % 10 == 0:
            cds_1 = [c for c in all_cds[1.0] if not np.isnan(c)]
            if cds_1:
                print(f"  [{i+1}/{len(frames)}] res=1.0m mean CD: {np.mean(cds_1):.4f}")
            else:
                print(f"  [{i+1}/{len(frames)}]")

    elapsed = time.time() - t0
    result = {}
    rows = []
    for res in resolutions:
        valid = [c for c in all_cds[res] if not np.isnan(c)]
        if valid:
            m = float(np.mean(valid))
            s = float(np.std(valid))
        else:
            m = s = float("nan")
        # estimate grid size from first frame
        lidar_coords_f0 = frames[0]["lidar_coords"]
        bbox_min_f0 = lidar_coords_f0.min(axis=0) - 5.0
        bbox_max_f0 = lidar_coords_f0.max(axis=0) + 5.0
        dims = ((bbox_max_f0 - bbox_min_f0) / res).astype(int)
        total = int(np.prod(dims))
        n_used = min(total, max_grid_pts)
        result[f"res={res}m"] = {"mean": m, "std": s, "grid_total": total, "grid_used": n_used}
        rows.append([f"{res}m", str(total), str(n_used), f"{m:.4f}", f"{s:.4f}"])

    result["time_s"] = round(elapsed, 1)
    print_table(rows, ["Resolution", "Grid Total", "Used", "CD Mean", "CD Std"],
                "Ablation 6: Voxel Grid Scaffold")
    print(f"  Time: {elapsed:.1f}s")
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("  ABLATION EVALUATION SUITE")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Frames: {NUM_FRAMES} from seq 08 (prevoxelized)")
    print("=" * 70)

    # load model once
    print("\n[init] building model ...")
    model = load_ckpt(build_model(DEVICE), str(CKPT_PATH), DEVICE)
    model.eval()

    # load data once
    frames = load_frames(NUM_FRAMES)

    all_results = OrderedDict()
    total_t0 = time.time()

    # run all ablations
    ablation_fns = [
        ("ablation_1_conditioning_zeroout", ablation_1_conditioning_zeroout),
        ("ablation_2_timestep_sweep", ablation_2_timestep_sweep),
        ("ablation_3_sparse_scaffold", ablation_3_sparse_scaffold),
        ("ablation_4_random_scaffold", ablation_4_random_scaffold),
        ("ablation_5_input_density", ablation_5_input_density),
        ("ablation_6_voxel_grid_scaffold", ablation_6_voxel_grid_scaffold),
    ]

    for name, fn in ablation_fns:
        try:
            result = fn(model, frames)
            all_results[name] = result
        except Exception as e:
            print(f"\n[ERROR] {name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    total_elapsed = time.time() - total_t0

    # save results
    all_results["total_time_s"] = round(total_elapsed, 1)
    all_results["num_frames"] = NUM_FRAMES
    all_results["checkpoint"] = str(CKPT_PATH)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] results -> {RESULTS_PATH}")

    # final summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, res in all_results.items():
        if isinstance(res, dict) and "error" not in res and "time_s" in res:
            print(f"  {name}: {res['time_s']}s")
        elif isinstance(res, dict) and "error" in res:
            print(f"  {name}: FAILED - {res['error']}")
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
