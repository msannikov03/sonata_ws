#!/usr/bin/env python3
"""
Evaluate latent-diffusion scene completion (VAE + latent diffusion).
Produces Chamfer distance metrics + BEV plots, comparable to evaluate.py.
"""
import os, sys, torch, numpy as np, argparse, time, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.latent_diffusion import SceneCompletionLatentDiffusion
from models.point_cloud_vae import PointCloudVAE
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.refinement_net import chamfer_distance


def infer_architecture_from_state_dict(sd):
    """Infer latent_dim and num_decoded_points from checkpoint weights."""
    w = sd["vae.fc_mu.weight"]
    latent_dim = w.shape[0]
    dec_w = sd["vae.decoder_out.weight"]
    k = dec_w.shape[0] // 3
    return latent_dim, k


def build_latent_model(ckpt_path, device="cuda"):
    """Build SceneCompletionLatentDiffusion from checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    latent_dim, k = infer_architecture_from_state_dict(sd)
    num_t = ck.get("num_timesteps", 1000)
    sched = ck.get("schedule", "cosine")

    vae = PointCloudVAE(latent_dim=latent_dim, num_decoded_points=k)
    encoder = SonataEncoder(
        pretrained="facebook/sonata",
        freeze=True,
        enable_flash=False,
        feature_levels=[0],
    )
    cond = ConditionalFeatureExtractor(
        encoder, feature_levels=[0], fusion_type="concat"
    )
    model = SceneCompletionLatentDiffusion(
        vae=vae,
        condition_extractor=cond,
        num_timesteps=num_t,
        schedule=sched,
        denoising_steps=50,
    )
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    epoch = ck.get("epoch", "?")
    print(f"Loaded latent diffusion checkpoint: {ckpt_path} (epoch {epoch})")
    print(f"  Architecture: latent_dim={latent_dim}, K={k}, T={num_t}, schedule={sched}")
    return model


def prepare_scan(pts_raw, device="cuda", max_points=20000, voxel_size=0.05):
    """Prepare partial scan dict for Sonata encoder (same as evaluate.py)."""
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


def bev_plot(pts_dict, title, save_path):
    n = len(pts_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, pts) in zip(axes, pts_dict.items()):
        if pts is not None and len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.1,
                       cmap="viridis", vmin=-2, vmax=4)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
        ax.set_title(name)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def run_latent_completion(model, point_dict, denoising_steps=50):
    """Run full DDPM reverse process in latent space, decode via VAE."""
    t0 = time.time()
    pts = model.complete_scene(point_dict, num_steps=denoising_steps)
    elapsed = time.time() - t0
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    return pts, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root of dataset (contains sequences/ and ground_truth/)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_latent_diffusion.pth")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_latent")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sequence", type=str, default="08")
    parser.add_argument("--denoising_steps", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # Build model
    print("=" * 60)
    print("LATENT DIFFUSION (VAE + Latent DDPM) EVALUATION")
    print("=" * 60)
    model = build_latent_model(args.checkpoint, device)

    # Find frames
    seq_dir = os.path.join(args.data_path, "sequences", args.sequence)
    vel_dir = os.path.join(seq_dir, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", args.sequence)
    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])

    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][:args.num_samples]
    print(f"\nEvaluating {len(sample_frames)} frames from seq {args.sequence}")
    print(f"Denoising steps: {args.denoising_steps}")
    print()

    results = []
    for i, fid in enumerate(sample_frames):
        print(f"--- Frame {fid} ({i+1}/{len(sample_frames)}) ---")

        lidar = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        gt_raw = load_gt(gt_path) if os.path.exists(gt_path) else None

        # Run latent diffusion completion
        scan_dict, center = prepare_scan(lidar, device)
        completed, elapsed = run_latent_completion(model, scan_dict, args.denoising_steps)

        # The completed points are in centered coords; add center back
        completed = completed + center

        r = {"frame": fid, "latent_time": elapsed, "num_output_points": int(completed.shape[0])}

        if gt_raw is not None:
            r["latent_cd"] = compute_cd(completed, gt_raw)
            print(f"  Latent diffusion: CD={r['latent_cd']:.4f}, "
                  f"time={elapsed:.2f}s, output_pts={completed.shape[0]}")
        else:
            print(f"  Latent diffusion: time={elapsed:.2f}s, "
                  f"output_pts={completed.shape[0]} (no GT)")

        # BEV visualization
        viz = {"Input (LiDAR)": lidar, "Latent Diffusion": completed}
        if gt_raw is not None:
            viz["GT"] = gt_raw
        bev_plot(viz, f"Latent Diffusion - Frame {fid}",
                 os.path.join(args.output_dir, f"bev_{fid}.png"))
        results.append(r)

    # Summary table
    print("\n" + "=" * 70)
    print("LATENT DIFFUSION EVALUATION RESULTS")
    print("=" * 70)
    header = f"{'Frame':<10} {'CD':<15} {'Time':<12} {'Output Pts':<12}"
    print(header)
    print("-" * 70)
    for r in results:
        cd = f"{r['latent_cd']:.4f}" if "latent_cd" in r else "-"
        tt = f"{r['latent_time']:.2f}s"
        npts = str(r["num_output_points"])
        print(f"{r['frame']:<10} {cd:<15} {tt:<12} {npts:<12}")

    cds = [r["latent_cd"] for r in results if "latent_cd" in r]
    times = [r["latent_time"] for r in results]
    print("-" * 70)
    if cds:
        print(f"Mean CD:   {np.mean(cds):.4f} +/- {np.std(cds):.4f}")
        print(f"Median CD: {np.median(cds):.4f}")
        print(f"Min CD:    {np.min(cds):.4f}")
        print(f"Max CD:    {np.max(cds):.4f}")
    print(f"Mean time: {np.mean(times):.2f}s")
    print(f"Total time: {np.sum(times):.1f}s for {len(results)} frames")

    # Save metrics
    summary = {
        "model": "latent_diffusion",
        "checkpoint": args.checkpoint,
        "denoising_steps": args.denoising_steps,
        "num_samples": len(results),
        "mean_cd": float(np.mean(cds)) if cds else None,
        "std_cd": float(np.std(cds)) if cds else None,
        "median_cd": float(np.median(cds)) if cds else None,
        "mean_time": float(np.mean(times)),
        "per_frame": results,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
