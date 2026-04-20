#!/usr/bin/env python3
"""
Generate predictions for Figure 1 (2x2 grid):
  1. Teacher v2GT on LiDAR input  (CD ~0.039)
  2. Teacher v2GT on DA2 input    (CD ~0.040)
  3. Random PTv3 on LiDAR input   (CD ~0.076)
  4. Random PTv3 on DA2 input     (CD ~0.071)

For a list of candidate frames from seq 08. Saves all 4 predictions + inputs + GT as .npz.

This script runs on compute (has the checkpoints + prevoxelized seq08 data).

Outputs:
  <output_dir>/frame_<fid>/teacher_lidar.npz  (pred, input, gt)
  <output_dir>/frame_<fid>/teacher_da2.npz
  <output_dir>/frame_<fid>/random_lidar.npz
  <output_dir>/frame_<fid>/random_da2.npz
"""
import os, sys, argparse, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance


def build_model(encoder_mode="pretrained_frozen", device="cuda"):
    if encoder_mode == "pretrained_frozen":
        encoder = SonataEncoder(
            pretrained="facebook/sonata", freeze=True,
            enable_flash=False, feature_levels=[0],
        )
    elif encoder_mode == "random_unfrozen":
        encoder = SonataEncoder(
            pretrained="random", freeze=False,
            enable_flash=False, feature_levels=[0],
        )
    else:
        raise ValueError(encoder_mode)
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    model = SceneCompletionDiffusion(
        encoder=encoder, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50
    )
    return model.to(device)


def load_ckpt(model, path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Some ckpts use model_state_dict, random_ptv3 uses 'model' key
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  MISSING keys: {len(missing)} (first 3): {missing[:3]}")
    if unexpected:
        print(f"  UNEXPECTED keys: {len(unexpected)} (first 3): {unexpected[:3]}")
    print(f"Loaded {path}")
    return model


def prepare_input(coords_centered, device="cuda"):
    """Build point dict from already-centered coords."""
    pts = coords_centered.astype(np.float32)
    z = pts[:, 2]
    zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1).astype(np.float32)
    return {
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(pts.shape[0], 3, dtype=torch.float32).to(device),
        "batch": torch.zeros(pts.shape[0], dtype=torch.long).to(device),
    }


@torch.no_grad()
def run_completion(model, point_dict, target_coords, device="cuda"):
    model.eval()
    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], target_coords)

    model.scheduler._to_device(device)
    t_val = 200
    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(target_coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * target_coords + som * noise

    pred_noise = model.denoiser(noisy, target_coords, t_tensor, {"features": cond_features})
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/home/anywherevla/sonata_ws/prevoxelized_seq08")
    p.add_argument("--teacher_ckpt", default="checkpoints/diffusion_v2gt/best_model.pth")
    p.add_argument("--random_lidar_ckpt", default="checkpoints/random_ptv3_lidar/random_unfrozen_lidar/best.pth")
    p.add_argument("--random_da2_ckpt", default="checkpoints/random_ptv3_da2/random_unfrozen_da2/best.pth")
    p.add_argument("--frames", nargs="+", type=int, default=[0, 500, 1000, 2000, 3000])
    p.add_argument("--output_dir", default="fig1_predictions")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    print("=" * 60)
    print("Building TEACHER (pretrained frozen)...")
    teacher = load_ckpt(build_model("pretrained_frozen", device), args.teacher_ckpt, device)

    print("\nBuilding RANDOM PTv3 (LiDAR-trained)...")
    random_lidar = load_ckpt(build_model("random_unfrozen", device), args.random_lidar_ckpt, device)

    print("\nBuilding RANDOM PTv3 (DA2-trained)...")
    random_da2 = load_ckpt(build_model("random_unfrozen", device), args.random_da2_ckpt, device)
    print("=" * 60)

    all_results = {}

    for fid in args.frames:
        print(f"\n--- Frame {fid:06d} ---")
        path = os.path.join(args.data_dir, f"{fid:06d}.npz")
        if not os.path.exists(path):
            print(f"  SKIP (file not found): {path}")
            continue
        data = np.load(path)

        # Shapes: (N, 3). Already voxelized + centered per-modality by preprocess.
        lidar_coords = data["lidar_coords"].astype(np.float32)  # centered on lidar mean
        da2_coords = data["da2_coords"].astype(np.float32)      # centered on da2 mean
        gt_lidar = data["gt_coords_lidar"].astype(np.float32)    # GT centered on lidar mean
        gt_da2 = data["gt_coords_da2"].astype(np.float32)        # GT centered on da2 mean
        gt_raw = data["gt_raw"].astype(np.float32)              # World-frame GT
        lidar_center = data["lidar_center"].astype(np.float32)
        da2_center = data["da2_center"].astype(np.float32)

        # Put back in world frame for rendering
        def to_world(coords, center):
            return coords + center[None, :]

        frame_dir = os.path.join(args.output_dir, f"frame_{fid:06d}")
        os.makedirs(frame_dir, exist_ok=True)

        # --- 1. Teacher on LiDAR input ---
        print("  Teacher on LiDAR ...", end=" ")
        pd = prepare_input(lidar_coords, device)
        target = torch.from_numpy(gt_lidar).float().to(device)
        pred = run_completion(teacher, pd, target, device)
        pred_world = to_world(pred, lidar_center)
        cd = chamfer_distance(
            torch.from_numpy(pred_world).float().cuda(),
            torch.from_numpy(gt_raw).float().cuda(),
            chunk_size=512,
        ).item()
        np.savez(
            os.path.join(frame_dir, "teacher_lidar.npz"),
            pred=pred_world,
            input=to_world(lidar_coords, lidar_center),
            gt=gt_raw,
            cd=cd,
            label="Pretrained PTv3 + LiDAR",
        )
        print(f"CD={cd:.4f}")

        # --- 2. Teacher on DA2 input ---
        print("  Teacher on DA2   ...", end=" ")
        pd = prepare_input(da2_coords, device)
        target = torch.from_numpy(gt_da2).float().to(device)
        pred = run_completion(teacher, pd, target, device)
        pred_world = to_world(pred, da2_center)
        cd = chamfer_distance(
            torch.from_numpy(pred_world).float().cuda(),
            torch.from_numpy(gt_raw).float().cuda(),
            chunk_size=512,
        ).item()
        np.savez(
            os.path.join(frame_dir, "teacher_da2.npz"),
            pred=pred_world,
            input=to_world(da2_coords, da2_center),
            gt=gt_raw,
            cd=cd,
            label="Pretrained PTv3 + DA2",
        )
        print(f"CD={cd:.4f}")

        # --- 3. Random on LiDAR input ---
        print("  Random on LiDAR  ...", end=" ")
        pd = prepare_input(lidar_coords, device)
        target = torch.from_numpy(gt_lidar).float().to(device)
        pred = run_completion(random_lidar, pd, target, device)
        pred_world = to_world(pred, lidar_center)
        cd = chamfer_distance(
            torch.from_numpy(pred_world).float().cuda(),
            torch.from_numpy(gt_raw).float().cuda(),
            chunk_size=512,
        ).item()
        np.savez(
            os.path.join(frame_dir, "random_lidar.npz"),
            pred=pred_world,
            input=to_world(lidar_coords, lidar_center),
            gt=gt_raw,
            cd=cd,
            label="Random PTv3 + LiDAR",
        )
        print(f"CD={cd:.4f}")

        # --- 4. Random on DA2 input ---
        print("  Random on DA2    ...", end=" ")
        pd = prepare_input(da2_coords, device)
        target = torch.from_numpy(gt_da2).float().to(device)
        pred = run_completion(random_da2, pd, target, device)
        pred_world = to_world(pred, da2_center)
        cd = chamfer_distance(
            torch.from_numpy(pred_world).float().cuda(),
            torch.from_numpy(gt_raw).float().cuda(),
            chunk_size=512,
        ).item()
        np.savez(
            os.path.join(frame_dir, "random_da2.npz"),
            pred=pred_world,
            input=to_world(da2_coords, da2_center),
            gt=gt_raw,
            cd=cd,
            label="Random PTv3 + DA2",
        )
        print(f"CD={cd:.4f}")

        # Summary for this frame
        all_results[f"frame_{fid:06d}"] = {
            "teacher_lidar": float(np.load(os.path.join(frame_dir, "teacher_lidar.npz"))["cd"]),
            "teacher_da2": float(np.load(os.path.join(frame_dir, "teacher_da2.npz"))["cd"]),
            "random_lidar": float(np.load(os.path.join(frame_dir, "random_lidar.npz"))["cd"]),
            "random_da2": float(np.load(os.path.join(frame_dir, "random_da2.npz"))["cd"]),
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
