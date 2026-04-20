#!/usr/bin/env python3
"""
Scaffold-Free Teacher Evaluation (apr17 morning experiment).

Goal: Measure teacher v2GT CD when the GT coordinate scaffold is removed --
for fair comparison with LiDiff / ScoreLiDAR / LiNeXt which do not use scaffolds.

Note on architecture:
  Our SceneCompletionDiffusion predicts noise AT A SET OF COORDINATES (scaffold).
  The coords serve as the spatial positions that the denoiser uses for attention.
  "Scaffold-free" in our framework means: replace GT coords with something not-GT
  (input scan duplicated + noised, voxel grid cropped to bbox, etc.),
  then start the diffusion process either from x_0=noise (pure scaffold-free DDPM/DDIM)
  or from noised GT-scaffold as our normal single-step.

Configurations tested:
  A  input_dup10_single_step  : Input scan duplicated 10x + jitter,
                                bbox-cropped to v2 GT extent, single-step x0 at t=200
  A2 input_dup5_single_step   : Same, 5x duplication (closer to LiDiff's multiplier)
  B  bbox_voxel_grid          : Voxel grid scaffold cropped to v2 GT bbox,
                                single-step x0 at t=200 (already ablated in run_scaffold_sweep
                                but NOT bbox-cropped; we crop here)
  E  ddpm_pure_noise          : Full 1000-step DDPM from pure noise, scaffold=input LiDAR
  E2 ddpm_pure_noise_gt_coords: Full 1000-step DDPM from pure noise, scaffold=GT coords
                                (isolates "scaffold vs full sampling" regardless of coords)
  F  ddim_50steps_pure_noise  : DDIM 50 steps from pure noise (start_t=999),
                                scaffold=input LiDAR
  F2 ddim_20steps_pure_noise  : DDIM 20 steps from pure noise, scaffold=input LiDAR

We evaluate on 50 frames of seq 08 using prevoxelized data.
Metric: Chamfer Distance vs v2 ground truth (world-frame).

Results -> results/apr17_morning/teacher_without_scaffold.json
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
RESULTS_PATH = RESULTS_DIR / "teacher_without_scaffold.json"
NUM_FRAMES = 50
DEVICE = "cuda"
MAX_PTS = 20000
SEED = 42


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
            "lidar_coords": d["lidar_coords"],      # centered on lidar mean
            "lidar_center": d["lidar_center"],
            "gt_coords_lidar": d["gt_coords_lidar"], # centered on lidar mean
            "gt_raw": d["gt_raw"],                   # world-frame GT
        })
    print(f"[data] loaded {len(frames)} frames from {PREVOX_DIR}")
    return frames


# ---- scaffold constructors ----
def scaffold_input_duplicated(fr, n_dup=10, jitter=0.05, max_pts=MAX_PTS, bbox_crop=True):
    """LiDiff-style: duplicate the partial scan N times with jitter, crop to v2 GT bbox."""
    lidar = fr["lidar_coords"]  # already centered on lidar_center
    gt_centered = fr["gt_raw"] - fr["lidar_center"]

    dups = []
    for i in range(n_dup):
        if i == 0:
            dups.append(lidar)
        else:
            dups.append(lidar + np.random.normal(0.0, jitter, size=lidar.shape).astype(np.float32))
    cloud = np.concatenate(dups, axis=0)

    if bbox_crop:
        # crop to v2 GT bounding box (centered frame) + small margin
        bmin = gt_centered.min(axis=0) - 0.5
        bmax = gt_centered.max(axis=0) + 0.5
        mask = np.all((cloud >= bmin) & (cloud <= bmax), axis=1)
        cloud = cloud[mask]

    if cloud.shape[0] > max_pts:
        idx = np.random.choice(cloud.shape[0], max_pts, replace=False)
        cloud = cloud[idx]
    return cloud.astype(np.float32)


def scaffold_bbox_voxel_grid(fr, res=0.2, max_pts=MAX_PTS):
    """Voxel grid cropped to v2 GT bounding box (not full LiDAR range)."""
    gt_centered = fr["gt_raw"] - fr["lidar_center"]
    bmin = gt_centered.min(axis=0)
    bmax = gt_centered.max(axis=0)
    xs = np.arange(bmin[0], bmax[0], res)
    ys = np.arange(bmin[1], bmax[1], res)
    zs = np.arange(bmin[2], bmax[2], res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    if grid.shape[0] > max_pts:
        idx = np.random.choice(grid.shape[0], max_pts, replace=False)
        grid = grid[idx]
    return grid


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


@torch.no_grad()
def run_ddpm_pure_noise(model, point_dict, scaffold_coords, num_steps=1000):
    """Full reverse DDPM from pure noise at t=T-1, scaffold=scaffold_coords (fixed)."""
    model.eval()
    device = point_dict["coord"].device
    model.scheduler._to_device(device)

    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], scaffold_coords)

    # x_T = pure noise (same shape as scaffold)
    x_t = torch.randn_like(scaffold_coords)

    for t in range(num_steps - 1, -1, -1):
        x_t = model.scheduler.p_sample_step(
            model.denoiser, x_t, scaffold_coords, t,
            {"features": cond_features},
        )
    return x_t


@torch.no_grad()
def run_ddim_pure_noise(model, point_dict, scaffold_coords, num_steps=50, start_t=999, eta=0.0):
    """DDIM reverse process from pure noise at start_t -> 0, scaffold=scaffold_coords."""
    model.eval()
    device = point_dict["coord"].device
    model.scheduler._to_device(device)

    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], scaffold_coords)
    condition = {"features": cond_features}

    # timesteps: [start_t, ..., 0]
    timesteps = list(reversed(
        [int(x) for x in torch.linspace(0, start_t, num_steps + 1)]
    ))

    # Initialize from pure noise at start_t
    noise = torch.randn_like(scaffold_coords)
    sa = model.scheduler.sqrt_alphas_cumprod[start_t]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[start_t]
    # Pure noise at t=999 means x_T ~ noise; we set x_t = noise (equivalent when alpha_bar ~ 0)
    x_t = noise.clone() if start_t >= 990 else sa * scaffold_coords + som * noise

    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_prev = timesteps[i + 1]
        if t_cur <= 0:
            break
        x_t = model.scheduler.ddim_sample_step(
            model.denoiser, x_t, scaffold_coords, t_cur, t_prev,
            condition, eta=eta,
        )
    return x_t


# ---- variants ----
def variant_A_input_dup10(model, fr):
    scaffold = scaffold_input_duplicated(fr, n_dup=10, jitter=0.05, bbox_crop=True)
    if scaffold.shape[0] < 64:
        return None, None
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_x0_single_step(model, point_dict, target, t_val=200)
    return pred.cpu().numpy(), scaffold


def variant_A2_input_dup5(model, fr):
    scaffold = scaffold_input_duplicated(fr, n_dup=5, jitter=0.05, bbox_crop=True)
    if scaffold.shape[0] < 64:
        return None, None
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_x0_single_step(model, point_dict, target, t_val=200)
    return pred.cpu().numpy(), scaffold


def variant_B_bbox_voxel(model, fr):
    scaffold = scaffold_bbox_voxel_grid(fr, res=0.2)
    if scaffold.shape[0] < 64:
        return None, None
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_x0_single_step(model, point_dict, target, t_val=200)
    return pred.cpu().numpy(), scaffold


def variant_E_ddpm_pure_noise(model, fr):
    """Full 1000-step DDPM from pure noise, scaffold = input LiDAR (LiDiff-style regime)."""
    scaffold = fr["lidar_coords"].copy()
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_ddpm_pure_noise(model, point_dict, target, num_steps=1000)
    return pred.cpu().numpy(), scaffold


def variant_E2_ddpm_pure_noise_gt_coords(model, fr):
    """Full 1000-step DDPM from pure noise, scaffold = GT coords.
    Isolates 'full sampling vs single-step' -- scaffold is still GT to keep coords perfect."""
    scaffold = fr["gt_coords_lidar"].copy()
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_ddpm_pure_noise(model, point_dict, target, num_steps=1000)
    return pred.cpu().numpy(), scaffold


def variant_F_ddim50_pure_noise(model, fr):
    """DDIM 50 steps from pure noise, scaffold = input LiDAR."""
    scaffold = fr["lidar_coords"].copy()
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_ddim_pure_noise(model, point_dict, target, num_steps=50, start_t=999, eta=0.0)
    return pred.cpu().numpy(), scaffold


def variant_F2_ddim20_pure_noise(model, fr):
    scaffold = fr["lidar_coords"].copy()
    point_dict = make_point_dict(fr["lidar_coords"])
    target = torch.from_numpy(scaffold).float().to(DEVICE)
    pred = run_ddim_pure_noise(model, point_dict, target, num_steps=20, start_t=999, eta=0.0)
    return pred.cpu().numpy(), scaffold


VARIANTS = [
    ("A_input_dup10_single_step_bbox", variant_A_input_dup10),
    ("A2_input_dup5_single_step_bbox", variant_A2_input_dup5),
    ("B_bbox_voxel_grid_0.2m_single_step", variant_B_bbox_voxel),
    ("E_ddpm_pure_noise_1000steps_lidar_scaffold", variant_E_ddpm_pure_noise),
    ("E2_ddpm_pure_noise_1000steps_gt_scaffold", variant_E2_ddpm_pure_noise_gt_coords),
    ("F_ddim_pure_noise_50steps_lidar_scaffold", variant_F_ddim50_pure_noise),
    ("F2_ddim_pure_noise_20steps_lidar_scaffold", variant_F2_ddim20_pure_noise),
]


def run_variant_all_frames(model, frames, name, fn, max_frames_for_slow=None):
    print(f"\n{'='*70}\n  Variant: {name}\n{'='*70}")
    fs = frames if max_frames_for_slow is None else frames[:max_frames_for_slow]
    t0 = time.time()
    pred_cds = []
    scaffold_cds = []
    per_frame_times = []
    for i, fr in enumerate(fs):
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
            cd_pred = compute_cd(pred_world, gt_raw)
            cd_scaffold = compute_cd(scaffold_world, gt_raw)
        except Exception as e:
            print(f"  [warn] frame {fr['name']}: CD failed: {e}")
            continue

        # Guard rail: if a variant blows up to >10^3 we still record it (honest failure)
        if not np.isfinite(cd_pred):
            cd_pred = float("inf")
        pred_cds.append(cd_pred)
        scaffold_cds.append(cd_scaffold)

        if (i + 1) % 10 == 0 or (i + 1) == len(fs):
            print(f"  [{i+1}/{len(fs)}] pred_cd={np.mean([c for c in pred_cds if np.isfinite(c)]):.4f}  "
                  f"scaffold_cd={np.mean(scaffold_cds):.4f}  "
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
            "per_frame_time_s_mean": float(np.mean(per_frame_times)),
            "total_time_s": round(elapsed, 1),
        }
    print(f"  -> {result}")
    return result


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  SCAFFOLD-FREE TEACHER EVALUATION (apr17_morning)")
    print("=" * 70)
    print(f"  results -> {RESULTS_PATH}")
    print(f"  ckpt     -> {CKPT_PATH}")
    print(f"  frames   -> {NUM_FRAMES}")
    print()

    model = build_model()
    model = load_ckpt(model, CKPT_PATH)
    model.eval()
    frames = load_frames(NUM_FRAMES)

    # For E/E2 full DDPM (1000 steps) we cap at fewer frames since it's ~50x slower.
    # Single DDPM step ~ same cost as single-step x0 (~300ms). So 1000 steps ~ 5 min / frame.
    # Restrict E/E2 to 3 frames to keep total runtime reasonable.
    slow_cap = 3

    results = {"variants": {}}

    for name, fn in VARIANTS:
        slow = name.startswith("E_") or name.startswith("E2_")
        cap = slow_cap if slow else None
        results["variants"][name] = run_variant_all_frames(
            model, frames, name, fn, max_frames_for_slow=cap,
        )
        # persist after each variant so partial results survive crashes
        payload = {
            **results,
            "num_frames": NUM_FRAMES,
            "checkpoint": str(CKPT_PATH),
            "seed": SEED,
            "partial": True,
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

    results["partial"] = False
    results["num_frames"] = NUM_FRAMES
    results["checkpoint"] = str(CKPT_PATH)
    results["seed"] = SEED
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
