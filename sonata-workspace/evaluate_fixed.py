#!/usr/bin/env python3
"""
Evaluate scene completion: teacher vs student vs GT.
Produces metrics + BEV visualizations.
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
    """Single-step x_0 prediction at a low noise level to test model quality."""
    model.eval()
    device = point_dict['coord'].device

    # Get conditioning
    cond_features, _ = model.condition_extractor(point_dict)

    if target_coords is not None:
        coords = target_coords
        from models.diffusion_module import knn_interpolate
        cond_features = knn_interpolate(cond_features, point_dict['coord'], coords)
    else:
        coords = point_dict['coord']

    # Move scheduler to device
    model.scheduler._to_device(device)

    # Add noise at a moderate timestep (t=200) to GT coords, then predict x_0
    t_val = 200
    t_tensor = torch.full((1,), t_val, device=device)
    noise = torch.randn_like(coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * coords + som * noise

    t0 = time.time()
    pred_noise = model.denoiser(noisy, coords, t_tensor, {'features': cond_features})
    elapsed = time.time() - t0

    # Reconstruct x_0
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
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--da2_cloud_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sequence", type=str, default="08")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    print("Building teacher model...")
    teacher = load_ckpt(build_model(device), args.teacher_ckpt, device)

    student = None
    if args.student_ckpt:
        print("Building student model...")
        student = load_ckpt(build_model(device), args.student_ckpt, device)

    seq_dir = os.path.join(args.data_path, "sequences", args.sequence)
    vel_dir = os.path.join(seq_dir, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", args.sequence)
    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])

    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][:args.num_samples]
    print(f"Evaluating {len(sample_frames)} frames from seq {args.sequence}")

    results = []
    for i, fid in enumerate(sample_frames):
        print(f"\n--- Frame {fid} ({i+1}/{len(sample_frames)}) ---")

        lidar = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        gt_raw = load_gt(gt_path) if os.path.exists(gt_path) else None

        # Prepare GT coords as target (matches training spatial structure)
        gt_target = None
        gt_center = None
        if gt_raw is not None:
            gt_center = lidar.mean(axis=0)  # same center as input
            gt_centered = gt_raw - gt_center
            # Voxelize and subsample GT to match training
            vc = np.floor(gt_centered / 0.05).astype(np.int32)
            _, idx = np.unique(vc, axis=0, return_index=True)
            gt_sub = gt_centered[idx]
            if gt_sub.shape[0] > 20000:
                sel = np.random.choice(gt_sub.shape[0], 20000, replace=False)
                gt_sub = gt_sub[sel]
            gt_target = torch.from_numpy(gt_sub).float().to(device)

        # Teacher
        t_input, center = prepare_scan(lidar, device)
        t_comp, t_time = run_completion_x0(teacher, t_input, target_coords=gt_target)
        t_comp += center
        r = {"frame": fid, "teacher_time": t_time}
        if gt_raw is not None:
            r["teacher_cd"] = compute_cd(t_comp, gt_raw)
            print(f"  Teacher: CD={r['teacher_cd']:.4f}, time={t_time:.2f}s")
        else:
            print(f"  Teacher: time={t_time:.2f}s (no GT)")

        # Student
        s_comp = None
        if student and args.da2_cloud_dir:
            da2_path = os.path.join(args.da2_cloud_dir, f"{fid}.bin")
            if os.path.exists(da2_path):
                da2 = load_bin(da2_path)
                s_input, center_s = prepare_scan(da2, device)

                # FIX: Recompute gt_target centered on DA2 mean, not LiDAR mean.
                # The student's input coords are centered on da2.mean() (center_s),
                # so gt_target must be in the same coordinate frame for knn_interpolate
                # and the diffusion forward pass to be spatially consistent.
                gt_target_student = None
                if gt_raw is not None:
                    gt_centered_s = gt_raw - center_s  # center on DA2 mean
                    vc_s = np.floor(gt_centered_s / 0.05).astype(np.int32)
                    _, idx_s = np.unique(vc_s, axis=0, return_index=True)
                    gt_sub_s = gt_centered_s[idx_s]
                    if gt_sub_s.shape[0] > 20000:
                        sel_s = np.random.choice(gt_sub_s.shape[0], 20000, replace=False)
                        gt_sub_s = gt_sub_s[sel_s]
                    gt_target_student = torch.from_numpy(gt_sub_s).float().to(device)

                s_comp, s_time = run_completion_x0(student, s_input, target_coords=gt_target_student)
                s_comp += center_s
                r["student_time"] = s_time
                if gt_raw is not None:
                    r["student_cd"] = compute_cd(s_comp, gt_raw)
                    print(f"  Student: CD={r['student_cd']:.4f}, time={s_time:.2f}s")

        # Viz
        viz = {"Input (LiDAR)": lidar, "Teacher": t_comp}
        if s_comp is not None:
            viz["Student"] = s_comp
        if gt_raw is not None:
            viz["GT"] = gt_raw
        bev_plot(viz, f"Frame {fid}", os.path.join(args.output_dir, f"bev_{fid}.png"))
        results.append(r)

    # Summary
    print("\n" + "=" * 60)
    header = f"{'Frame':<10} {'Teacher CD':<15} {'Student CD':<15} {'T time':<12} {'S time':<12}"
    print(header)
    print("=" * 60)
    for r in results:
        tcd = f"{r['teacher_cd']:.4f}" if "teacher_cd" in r else "-"
        scd = f"{r['student_cd']:.4f}" if "student_cd" in r else "-"
        tt = f"{r['teacher_time']:.2f}s"
        st = f"{r['student_time']:.2f}s" if "student_time" in r else "-"
        print(f"{r['frame']:<10} {tcd:<15} {scd:<15} {tt:<12} {st:<12}")

    tcds = [r["teacher_cd"] for r in results if "teacher_cd" in r]
    scds = [r["student_cd"] for r in results if "student_cd" in r]
    print(f"\nAverages:")
    if tcds:
        print(f"  Teacher CD: {np.mean(tcds):.4f} +/- {np.std(tcds):.4f}")
    if scds:
        print(f"  Student CD: {np.mean(scds):.4f} +/- {np.std(scds):.4f}")
    print(f"  Teacher time: {np.mean([r['teacher_time'] for r in results]):.2f}s")
    if any("student_time" in r for r in results):
        print(f"  Student time: {np.mean([r['student_time'] for r in results if 'student_time' in r]):.2f}s")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
