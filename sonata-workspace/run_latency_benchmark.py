#!/usr/bin/env python3
"""
Hardware Latency Benchmark — Step 5 of apr17_morning experiment queue.

Measure per-frame inference time on RTX 4090 for teacher v2GT at:
  - Input sizes: 1k, 5k, 10k, 20k points
  - Target scaffold sizes: 5k, 10k, 20k points

Compute FPS, compare to LiDiff (30s), ScoreLiDAR (5.37s), LiNeXt (0.167s).

Results -> results/apr17_morning/latency_benchmark.json
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
RESULTS_PATH = RESULTS_DIR / "latency_benchmark.json"
DEVICE = "cuda"
N_FRAMES = 20
N_WARMUP = 3
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


def load_frames(n=N_FRAMES + N_WARMUP):
    files = sorted(PREVOX_DIR.glob("*.npz"))[:n]
    frames = []
    for f in files:
        d = np.load(f)
        frames.append({
            "name": f.stem,
            "lidar_coords": d["lidar_coords"],
            "lidar_center": d["lidar_center"],
            "gt_coords_lidar": d["gt_coords_lidar"],
        })
    return frames


@torch.no_grad()
def run_once_timed(model, point_dict, target_coords, sync=True):
    """Time the full single-step inference (encoder + knn + denoiser)."""
    model.eval()
    device = point_dict["coord"].device
    model.scheduler._to_device(device)

    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], target_coords)

    t_val = 200
    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(target_coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * target_coords + som * noise
    pred_noise = model.denoiser(noisy, target_coords, t_tensor, {"features": cond_features})
    _ = (noisy - som * pred_noise) / sa

    if sync:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def run_encoder_only_timed(model, point_dict, sync=True):
    model.eval()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model.condition_extractor(point_dict)
    if sync:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def run_denoiser_only_timed(model, point_dict, target_coords, sync=True):
    """Only the denoiser step, pre-computing cond features (cached)."""
    model.eval()
    device = point_dict["coord"].device
    model.scheduler._to_device(device)
    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], target_coords)
    t_val = 200
    t_tensor = torch.full((1,), t_val, device=device)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]

    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    noise = torch.randn_like(target_coords)
    noisy = sa * target_coords + som * noise
    pred_noise = model.denoiser(noisy, target_coords, t_tensor, {"features": cond_features})
    _ = (noisy - som * pred_noise) / sa

    if sync:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def subsample(points, n):
    if points.shape[0] <= n:
        return points
    idx = np.random.choice(points.shape[0], n, replace=False)
    return points[idx]


def benchmark_config(model, frames, input_size, target_size, label):
    """Benchmark a (input_size, target_size) config across N_FRAMES frames + warmup."""
    full_times = []
    enc_times = []
    den_times = []

    # warmup
    for wi in range(N_WARMUP):
        fr = frames[wi]
        pts = subsample(fr["lidar_coords"], input_size)
        tgt = subsample(fr["gt_coords_lidar"], target_size)
        pd = make_point_dict(pts)
        target = torch.from_numpy(tgt).float().to(DEVICE)
        try:
            _ = run_once_timed(model, pd, target)
        except Exception as e:
            print(f"  [warn] warmup failed for {label}: {e}")

    # timed
    for fi in range(N_WARMUP, N_WARMUP + N_FRAMES):
        fr = frames[fi]
        pts = subsample(fr["lidar_coords"], input_size)
        tgt = subsample(fr["gt_coords_lidar"], target_size)
        pd = make_point_dict(pts)
        target = torch.from_numpy(tgt).float().to(DEVICE)
        try:
            full_t = run_once_timed(model, pd, target)
            enc_t = run_encoder_only_timed(model, pd)
            den_t = run_denoiser_only_timed(model, pd, target)
        except Exception as e:
            print(f"  [warn] timing failed for {label}: {e}")
            continue
        full_times.append(full_t)
        enc_times.append(enc_t)
        den_times.append(den_t)

    if not full_times:
        return {"error": "no valid timings", "label": label}

    return {
        "label": label,
        "input_size": input_size,
        "target_size": target_size,
        "n_samples": len(full_times),
        "full_mean_s": float(np.mean(full_times)),
        "full_std_s": float(np.std(full_times)),
        "full_median_s": float(np.median(full_times)),
        "full_min_s": float(np.min(full_times)),
        "full_fps": float(1.0 / np.mean(full_times)),
        "encoder_mean_s": float(np.mean(enc_times)),
        "encoder_fps": float(1.0 / np.mean(enc_times)),
        "denoiser_mean_s": float(np.mean(den_times)),
        "denoiser_fps": float(1.0 / np.mean(den_times)),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  LATENCY BENCHMARK")
    print(f"  N_FRAMES={N_FRAMES} warmup={N_WARMUP}")
    print("=" * 70)

    model = load_ckpt(build_model(DEVICE), str(CKPT_PATH), DEVICE)
    model.eval()
    frames = load_frames(N_FRAMES + N_WARMUP)
    print(f"[data] loaded {len(frames)} frames")

    # GPU info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    print(f"GPU: {gpu_name}")

    configs = [
        # input size sweep (fixed target=20k)
        (1000, 20000, "input_1k"),
        (5000, 20000, "input_5k"),
        (10000, 20000, "input_10k"),
        (20000, 20000, "input_20k"),
        # target size sweep (fixed input=20k)
        (20000, 5000, "target_5k"),
        (20000, 10000, "target_10k"),
        # default config (matches evaluation)
        (20000, 20000, "default"),
    ]

    results = OrderedDict()
    t0 = time.time()
    for input_size, target_size, label in configs:
        print(f"\n[bench] {label}: input={input_size} target={target_size}")
        try:
            r = benchmark_config(model, frames, input_size, target_size, label)
            results[label] = r
            if "error" not in r:
                print(f"   full = {r['full_mean_s']*1000:.2f} ms  -> {r['full_fps']:.1f} FPS")
                print(f"   encoder = {r['encoder_mean_s']*1000:.2f} ms   denoiser = {r['denoiser_mean_s']*1000:.2f} ms")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[label] = {"error": str(e)}

    total_elapsed = time.time() - t0

    # Published-method comparison
    published = {
        "LiDiff_CVPR2024": 30.0,
        "ScoreLiDAR_ICCV2025": 5.37,
        "LiNeXt_2025": 0.167,
    }
    default = results.get("default", {})
    ours_default_s = default.get("full_mean_s", None)
    comparison = {}
    if ours_default_s is not None:
        comparison["ours_default_s"] = ours_default_s
        comparison["ours_default_fps"] = 1.0 / ours_default_s
        for name, t_pub in published.items():
            comparison[f"speedup_vs_{name}"] = t_pub / ours_default_s

    output = {
        "gpu": gpu_name,
        "n_frames": N_FRAMES,
        "n_warmup": N_WARMUP,
        "configs": results,
        "published_comparison": comparison,
        "published_times_s": published,
        "total_time_s": round(total_elapsed, 1),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Config':<15} {'Full (ms)':<15} {'FPS':<10} {'Encoder':<12} {'Denoiser':<12}")
    for label, r in results.items():
        if "error" in r:
            print(f"{label:<15} ERROR")
            continue
        print(f"{label:<15} {r['full_mean_s']*1000:<15.2f} {r['full_fps']:<10.1f} {r['encoder_mean_s']*1000:<12.2f} {r['denoiser_mean_s']*1000:<12.2f}")
    if comparison:
        print(f"\nOurs default: {comparison['ours_default_s']*1000:.2f}ms ({comparison['ours_default_fps']:.1f} FPS)")
        for k, v in comparison.items():
            if k.startswith("speedup_vs_"):
                print(f"  {k}: {v:.1f}x")


if __name__ == "__main__":
    main()
