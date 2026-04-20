#!/usr/bin/env python3
"""
Evaluate teacher model with DA2 input (camera pseudo-LiDAR).
Uses the same evaluation logic as evaluate.py but feeds DA2 clouds
to the teacher instead of LiDAR.
"""
import os, sys, torch, numpy as np, argparse, time, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion
from models.refinement_net import chamfer_distance

# Reproducibility seeds
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {path} (epoch {epoch})")
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


def bev_plot(pts_dict, title, save_path):
    n = len(pts_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, pts) in zip(axes, pts_dict.items()):
        if pts is not None and len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.1, cmap="viridis", vmin=-2, vmax=4)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_title(name)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def run_completion_x0(model, point_dict, target_coords=None):
    model.eval()
    device = point_dict['coord'].device
    cond_features, _ = model.condition_extractor(point_dict)
    if target_coords is not None:
        coords = target_coords
        from models.diffusion_module import knn_interpolate
        cond_features = knn_interpolate(cond_features, point_dict['coord'], coords)
    else:
        coords = point_dict['coord']
    model.scheduler._to_device(device)
    t_val = 200
    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * coords + som * noise
    t0 = time.time()
    pred_noise = model.denoiser(noisy, coords, t_tensor, {'features': cond_features})
    elapsed = time.time() - t0
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0.cpu().numpy(), elapsed


def compute_cd(pred, gt, max_pts=10000):
    if pred.shape[0] > max_pts:
        pred = pred[np.random.choice(pred.shape[0], max_pts, replace=False)]
    if gt.shape[0] > max_pts:
        gt = gt[np.random.choice(gt.shape[0], max_pts, replace=False)]
    cd = chamfer_distance(
        torch.from_numpy(pred).float().cuda(),
        torch.from_numpy(gt).float().cuda(),
        chunk_size=512
    )
    return cd.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--da2_cloud_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="evaluation_teacher_on_da2")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--sequence", type=str, default="08")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    print("Building teacher model...")
    model = load_ckpt(build_model(device), args.teacher_ckpt, device)

    seq_dir = os.path.join(args.data_path, "sequences", args.sequence)
    vel_dir = os.path.join(seq_dir, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", args.sequence)
    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])

    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][:args.num_samples]
    print(f"Evaluating {len(sample_frames)} frames from seq {args.sequence}")
    print("Teacher model on DA2 (camera) input\n")

    results = []
    for i, fid in enumerate(sample_frames):
        print(f"\n--- Frame {fid} ({i+1}/{len(sample_frames)}) ---")

        # Load DA2 pseudo-cloud as input (instead of LiDAR)
        da2_path = os.path.join(args.da2_cloud_dir, fid + ".bin")
        if not os.path.exists(da2_path):
            print(f"  SKIP: no DA2 cloud at {da2_path}")
            continue

        da2 = load_bin(da2_path)

        gt_path = os.path.join(gt_dir, fid + ".npz")
        gt_raw = load_gt(gt_path) if os.path.exists(gt_path) else None

        # Prepare GT coords as target (same as in evaluate.py)
        gt_target = None
        if gt_raw is not None:
            gt_center = da2.mean(axis=0)  # center using DA2 input (as student does)
            gt_centered = gt_raw - gt_center
            vc = np.floor(gt_centered / 0.05).astype(np.int32)
            _, idx = np.unique(vc, axis=0, return_index=True)
            gt_sub = gt_centered[idx]
            if gt_sub.shape[0] > 20000:
                sel = np.random.choice(gt_sub.shape[0], 20000, replace=False)
                gt_sub = gt_sub[sel]
            gt_target = torch.from_numpy(gt_sub).float().to(device)

        # Run teacher on DA2 input
        input_dict, center = prepare_scan(da2, device)
        comp, comp_time = run_completion_x0(model, input_dict, target_coords=gt_target)
        comp += center

        r = {"frame": fid, "time": comp_time}
        if gt_raw is not None:
            cd_val = compute_cd(comp, gt_raw)
            r["cd"] = cd_val
            print(f"  Teacher+DA2: CD={cd_val:.4f}, time={comp_time:.2f}s")
        else:
            print(f"  Teacher+DA2: time={comp_time:.2f}s (no GT)")

        # Viz: DA2 input, teacher completion, GT
        viz = {"Input (DA2)": da2, "Teacher+DA2": comp}
        if gt_raw is not None:
            viz["GT"] = gt_raw
        bev_plot(viz, "Teacher on DA2 - Frame " + fid, os.path.join(args.output_dir, "bev_" + fid + ".png"))
        results.append(r)

    # Summary
    cds = [r["cd"] for r in results if "cd" in r]
    print("\n" + "=" * 60)
    header = "{:<10} {:<15} {:<12}".format("Frame", "CD", "Time")
    print(header)
    print("=" * 60)
    for r in results:
        cd_str = "{:.4f}".format(r["cd"]) if "cd" in r else "-"
        t_str = "{:.2f}s".format(r["time"])
        print("{:<10} {:<15} {:<12}".format(r["frame"], cd_str, t_str))
    print("\nTeacher on DA2 input:")
    if cds:
        print("  CD: {:.4f} +/- {:.4f}".format(np.mean(cds), np.std(cds)))
    times = [r["time"] for r in results]
    print("  Avg time: {:.2f}s".format(np.mean(times)))
    print("  N samples: {}".format(len(results)))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to " + args.output_dir + "/")


if __name__ == "__main__":
    main()
