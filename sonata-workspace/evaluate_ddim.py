#!/usr/bin/env python3
"""
DDIM Multi-Step Inference Evaluation for Scene Completion

Tests whether DDIM multi-step sampling improves over single-step x0 prediction.
Sweeps: num_steps x start_t configurations.
"""

import os, sys, torch, numpy as np, time, json
from pathlib import Path

WORKSPACE = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
sys.path.insert(0, WORKSPACE)

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance


def build_model(device="cuda"):
    encoder = SonataEncoder(
        pretrained="facebook/sonata", freeze=True,
        enable_flash=False, feature_levels=[0]
    )
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    model = SceneCompletionDiffusion(
        encoder=encoder, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50
    )
    return model.to(device)


def load_ckpt(model, path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {path} (epoch {ckpt.get('epoch', '?')})")
    return model


def prepare_scan(pts_raw, device="cuda", max_points=20000, voxel_size=0.05):
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
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(pts.shape[0], 3).float().to(device),
        "batch": torch.zeros(pts.shape[0], dtype=torch.long).to(device),
    }, center


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path):
    return np.load(path)["points"]


@torch.no_grad()
def run_single_step_x0(model, point_dict, target_coords, t_val=200):
    """Baseline: single-step x0 prediction at t=t_val."""
    model.eval()
    device = point_dict["coord"].device

    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], target_coords)
    model.scheduler._to_device(device)

    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(target_coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * target_coords + som * noise

    pred_noise = model.denoiser(noisy, target_coords, t_tensor, {"features": cond_features})
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0.cpu().numpy()


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


def main():
    device = "cuda"
    np.random.seed(42)
    torch.manual_seed(42)

    data_path = "/home/anywherevla/sonata_ws/dataset/sonata_depth_pro"
    ckpt_path = os.path.join(WORKSPACE, "checkpoints/diffusion_v2gt/best_model.pth")

    print("Building model...")
    model = load_ckpt(build_model(device), ckpt_path, device)

    # Load frames
    seq = "08"
    vel_dir = os.path.join(data_path, "sequences", seq, "velodyne")
    gt_dir = os.path.join(data_path, "ground_truth", seq)

    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])
    # Use 30 evenly spaced frames for quick evaluation
    num_samples = 30
    step = max(1, len(frames) // num_samples)
    sample_frames = frames[::step][:num_samples]
    print(f"Evaluating {len(sample_frames)} frames from seq {seq}")

    # Pre-load data
    print("Pre-loading frame data...")
    frame_data = []
    for fid in sample_frames:
        lidar_path = os.path.join(vel_dir, f"{fid}.bin")
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        if not os.path.exists(gt_path):
            continue
        lidar = load_bin(lidar_path)
        gt = load_gt(gt_path)
        frame_data.append({"fid": fid, "lidar": lidar, "gt": gt})
    print(f"  loaded {len(frame_data)} frames")

    # Pre-compute GT targets and inputs
    print("Pre-computing inputs and GT targets...")
    inputs = []
    gt_targets = []
    centers = []
    for fd in frame_data:
        pt_dict, center = prepare_scan(fd["lidar"], device)
        inputs.append(pt_dict)
        centers.append(center)

        gt_centered = fd["gt"] - center
        vc = np.floor(gt_centered / 0.05).astype(np.int32)
        _, idx = np.unique(vc, axis=0, return_index=True)
        gt_sub = gt_centered[idx]
        if gt_sub.shape[0] > 20000:
            sel = np.random.choice(gt_sub.shape[0], 20000, replace=False)
            gt_sub = gt_sub[sel]
        gt_targets.append(torch.from_numpy(gt_sub).float().to(device))

    # Configuration sweep
    num_steps_list = [1, 2, 5, 10, 20]
    start_t_list = [200, 300, 500]

    all_results = {}

    # First: baseline single-step x0 at t=200
    print("\n" + "=" * 70)
    print("BASELINE: Single-step x0 at t=200")
    print("=" * 70)
    baseline_cds = []
    t0 = time.time()
    for i, fd in enumerate(frame_data):
        pred = run_single_step_x0(model, inputs[i], gt_targets[i], t_val=200)
        pred += centers[i]
        cd = compute_cd(pred, fd["gt"])
        baseline_cds.append(cd)
    baseline_time = (time.time() - t0) / len(frame_data)
    baseline_mean = np.mean(baseline_cds)
    baseline_std = np.std(baseline_cds)
    print(f"  CD: {baseline_mean:.4f} +/- {baseline_std:.4f}  ({baseline_time:.3f}s/frame)")
    all_results["baseline_t200_1step"] = {
        "cd_mean": float(baseline_mean),
        "cd_std": float(baseline_std),
        "time_per_frame": float(baseline_time),
        "start_t": 200,
        "num_steps": 1,
        "method": "single_step_x0",
    }

    # DDIM sweep
    for start_t in start_t_list:
        for num_steps in num_steps_list:
            if num_steps == 1 and start_t == 200:
                config_key = f"ddim_t{start_t}_{num_steps}steps"
                all_results[config_key] = {
                    "cd_mean": float(baseline_mean),
                    "cd_std": float(baseline_std),
                    "time_per_frame": float(baseline_time),
                    "start_t": start_t,
                    "num_steps": num_steps,
                    "method": "ddim",
                    "note": "equivalent to single-step x0 baseline",
                }
                continue

            config_key = f"ddim_t{start_t}_{num_steps}steps"
            print(f"\n--- DDIM start_t={start_t}, num_steps={num_steps} ---")

            cds = []
            t0 = time.time()
            for i, fd in enumerate(frame_data):
                pred = model.complete_scene_ddim(
                    inputs[i], gt_targets[i],
                    num_steps=num_steps, start_t=start_t, eta=0.0
                )
                pred_np = pred.cpu().numpy() + centers[i]
                cd = compute_cd(pred_np, fd["gt"])
                cds.append(cd)
            elapsed = (time.time() - t0) / len(frame_data)

            cd_mean = np.mean(cds)
            cd_std = np.std(cds)
            delta = cd_mean - baseline_mean
            pct = (delta / baseline_mean) * 100
            better = "BETTER" if delta < 0 else "WORSE"
            print(f"  CD: {cd_mean:.4f} +/- {cd_std:.4f}  ({elapsed:.3f}s/frame)  [{better}: {delta:+.4f} ({pct:+.1f}%)]")

            all_results[config_key] = {
                "cd_mean": float(cd_mean),
                "cd_std": float(cd_std),
                "time_per_frame": float(elapsed),
                "start_t": start_t,
                "num_steps": num_steps,
                "method": "ddim",
                "delta_vs_baseline": float(delta),
                "pct_vs_baseline": float(pct),
            }

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<30} {'CD Mean':>10} {'CD Std':>10} {'Time/F':>10} {'vs Baseline':>15}")
    print("=" * 90)

    sorted_keys = sorted(all_results.keys(), key=lambda k: all_results[k]["cd_mean"])
    for key in sorted_keys:
        r = all_results[key]
        delta = r.get("delta_vs_baseline", 0.0)
        pct = r.get("pct_vs_baseline", 0.0)
        if key == "baseline_t200_1step":
            delta_str = "(baseline)"
        else:
            delta_str = f"{delta:+.4f} ({pct:+.1f}%)"
        print(f"  {key:<28} {r['cd_mean']:>10.4f} {r['cd_std']:>10.4f} {r['time_per_frame']:>10.3f}s {delta_str:>15}")

    print("=" * 90)

    # Save results
    output_path = os.path.join(WORKSPACE, "evaluation_ddim_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
