#!/usr/bin/env python3
"""
Evaluate latent-diffusion (VAE + DDPM) vs old direct-diffusion pipeline.
Fair comparison: uses same frames, reports multiple metrics including:
- Latent diffusion full pipeline (condition -> denoise latent -> VAE decode)
- VAE reconstruction baseline (encode GT -> decode, upper bound for pipeline)
- Old pipeline numbers (loaded from saved metrics.json)
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
    w = sd["vae.fc_mu.weight"]
    latent_dim = w.shape[0]
    dec_w = sd["vae.decoder_out.weight"]
    k = dec_w.shape[0] // 3
    return latent_dim, k


def build_latent_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    latent_dim, k = infer_architecture_from_state_dict(sd)
    num_t = ck.get("num_timesteps", 1000)
    sched = ck.get("schedule", "cosine")

    vae = PointCloudVAE(latent_dim=latent_dim, num_decoded_points=k)
    encoder = SonataEncoder(
        pretrained="facebook/sonata", freeze=True,
        enable_flash=False, feature_levels=[0],
    )
    cond = ConditionalFeatureExtractor(
        encoder, feature_levels=[0], fusion_type="concat"
    )
    model = SceneCompletionLatentDiffusion(
        vae=vae, condition_extractor=cond,
        num_timesteps=num_t, schedule=sched, denoising_steps=50,
    )
    model.load_state_dict(sd)
    model = model.to(device).eval()
    epoch = ck.get("epoch", "?")
    print(f"Loaded latent diffusion: {ckpt_path} (epoch {epoch})")
    print(f"  latent_dim={latent_dim}, K={k}, T={num_t}, schedule={sched}")
    return model, k


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


def compute_cd(pred, gt, max_pts=10000):
    """Chamfer distance, subsampling both to max_pts."""
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


def compute_cd_matched(pred, gt, n_pts=2048):
    """Chamfer distance with both point clouds subsampled to same count for fairness."""
    if pred.shape[0] > n_pts:
        pred = pred[np.random.choice(pred.shape[0], n_pts, replace=False)]
    if gt.shape[0] > n_pts:
        gt = gt[np.random.choice(gt.shape[0], n_pts, replace=False)]
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
            ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3,
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
    t0 = time.time()
    pts = model.complete_scene(point_dict, num_steps=denoising_steps)
    elapsed = time.time() - t0
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    return pts, elapsed


@torch.no_grad()
def vae_reconstruct_gt(model, gt_pts_centered, device="cuda", n_input=8000):
    """Upper bound: encode GT with VAE, decode. Shows best possible output."""
    if gt_pts_centered.shape[0] > n_input:
        sel = np.random.choice(gt_pts_centered.shape[0], n_input, replace=False)
        gt_sub = gt_pts_centered[sel]
    else:
        gt_sub = gt_pts_centered
    mu, _ = model.vae.encode(torch.from_numpy(gt_sub).float().to(device))
    recon = model.vae.decode(mu)
    if isinstance(recon, torch.Tensor):
        recon = recon.detach().cpu().numpy()
    return recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--old_metrics", type=str, default=None,
                        help="metrics.json from old pipeline evaluation")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_latent")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sequence", type=str, default="08")
    parser.add_argument("--denoising_steps", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    print("=" * 70)
    print("LATENT DIFFUSION vs DIRECT DIFFUSION COMPARISON")
    print("=" * 70)

    model, K = build_latent_model(args.checkpoint, device)

    # Load old pipeline results if available
    old_results = {}
    if args.old_metrics and os.path.exists(args.old_metrics):
        with open(args.old_metrics) as f:
            old_data = json.load(f)
        for r in old_data:
            old_results[r["frame"]] = r
        print(f"Loaded {len(old_results)} old pipeline results")

    # Find frames
    vel_dir = os.path.join(args.data_path, "sequences", args.sequence, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", args.sequence)
    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])

    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][:args.num_samples]
    print(f"\nEvaluating {len(sample_frames)} frames from seq {args.sequence}")
    print(f"Denoising steps: {args.denoising_steps}, VAE output K={K}\n")

    results = []
    for i, fid in enumerate(sample_frames):
        print(f"--- Frame {fid} ({i+1}/{len(sample_frames)}) ---")

        lidar = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        gt_raw = load_gt(gt_path) if os.path.exists(gt_path) else None

        scan_dict, center = prepare_scan(lidar, device)

        # 1) Latent diffusion completion
        completed_centered, elapsed = run_latent_completion(
            model, scan_dict, args.denoising_steps
        )
        completed_world = completed_centered + center

        r = {
            "frame": fid,
            "latent_time": elapsed,
            "num_output_points": int(completed_centered.shape[0]),
        }

        if gt_raw is not None:
            gt_centered = gt_raw - center

            # CD: latent output (2048 pts) vs GT subsampled to 2048 (fair)
            r["latent_cd_matched"] = compute_cd_matched(
                completed_world, gt_raw, n_pts=K
            )
            # CD: latent output vs GT 10k (comparable to old pipeline format)
            r["latent_cd_10k"] = compute_cd(completed_world, gt_raw, max_pts=10000)

            # 2) VAE reconstruction baseline (upper bound)
            vae_recon = vae_reconstruct_gt(model, gt_centered, device, n_input=8000)
            vae_recon_world = vae_recon + center
            r["vae_recon_cd_matched"] = compute_cd_matched(
                vae_recon_world, gt_raw, n_pts=K
            )

            # 3) Old pipeline result (if available)
            if fid in old_results:
                r["old_teacher_cd"] = old_results[fid]["teacher_cd"]

            print(f"  Latent diffusion CD (matched {K}): {r['latent_cd_matched']:.4f}")
            print(f"  Latent diffusion CD (vs 10k GT):   {r['latent_cd_10k']:.4f}")
            print(f"  VAE recon baseline CD (matched):    {r['vae_recon_cd_matched']:.4f}")
            if "old_teacher_cd" in r:
                print(f"  Old direct-diffusion teacher CD:    {r['old_teacher_cd']:.4f}")
            print(f"  Time: {elapsed:.2f}s")

            # BEV plot
            viz = {
                "Input": lidar,
                "Latent Diffusion": completed_world,
                "VAE Recon (GT)": vae_recon_world,
                "GT": gt_raw,
            }
        else:
            viz = {"Input": lidar, "Latent Diffusion": completed_world}
            print(f"  Time: {elapsed:.2f}s (no GT)")

        bev_plot(viz, f"Frame {fid}", os.path.join(args.output_dir, f"bev_{fid}.png"))
        results.append(r)

    # Summary
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    header = (f"{'Frame':<8} {'Latent CD':<12} {'Latent CD':<12} "
              f"{'VAE Recon':<12} {'Old Direct':<12} {'Time':<8}")
    subhdr = (f"{'':8} {'(matched)':12} {'(vs 10k)':12} "
              f"{'(baseline)':12} {'(teacher)':12} {'':8}")
    print(header)
    print(subhdr)
    print("-" * 90)
    for r in results:
        lcd_m = f"{r['latent_cd_matched']:.4f}" if "latent_cd_matched" in r else "-"
        lcd_10 = f"{r['latent_cd_10k']:.4f}" if "latent_cd_10k" in r else "-"
        vae_cd = f"{r['vae_recon_cd_matched']:.4f}" if "vae_recon_cd_matched" in r else "-"
        old_cd = f"{r['old_teacher_cd']:.4f}" if "old_teacher_cd" in r else "-"
        tt = f"{r['latent_time']:.2f}s"
        print(f"{r['frame']:<8} {lcd_m:<12} {lcd_10:<12} {vae_cd:<12} {old_cd:<12} {tt:<8}")

    # Averages
    latent_matched = [r["latent_cd_matched"] for r in results if "latent_cd_matched" in r]
    latent_10k = [r["latent_cd_10k"] for r in results if "latent_cd_10k" in r]
    vae_recon = [r["vae_recon_cd_matched"] for r in results if "vae_recon_cd_matched" in r]
    old_cds = [r["old_teacher_cd"] for r in results if "old_teacher_cd" in r]
    times = [r["latent_time"] for r in results]

    print("-" * 90)
    if latent_matched:
        print(f"{'MEAN':<8} {np.mean(latent_matched):<12.4f} {np.mean(latent_10k):<12.4f} "
              f"{np.mean(vae_recon):<12.4f} "
              f"{np.mean(old_cds) if old_cds else '-':<12} {np.mean(times):.2f}s")
        print(f"{'STD':<8} {np.std(latent_matched):<12.4f} {np.std(latent_10k):<12.4f} "
              f"{np.std(vae_recon):<12.4f}")

    print(f"\n--- Analysis ---")
    if latent_matched and vae_recon:
        print(f"VAE reconstruction CD (upper bound):   {np.mean(vae_recon):.4f}")
        print(f"Latent diffusion CD (matched):         {np.mean(latent_matched):.4f}")
        gap = np.mean(latent_matched) - np.mean(vae_recon)
        print(f"Gap (diffusion overhead):               {gap:.4f}")
        if old_cds:
            print(f"Old direct diffusion CD:               {np.mean(old_cds):.4f}")
            print(f"\nNOTE: Old pipeline uses single-step x0 prediction at t=200 with GT target coords.")
            print(f"      Latent pipeline uses full {args.denoising_steps}-step DDPM reverse from noise.")
            print(f"      These are different evaluation protocols.")

    # Save
    summary = {
        "model": "latent_diffusion_comparison",
        "checkpoint": args.checkpoint,
        "denoising_steps": args.denoising_steps,
        "vae_output_K": K,
        "num_samples": len(results),
        "mean_latent_cd_matched": float(np.mean(latent_matched)) if latent_matched else None,
        "mean_latent_cd_10k": float(np.mean(latent_10k)) if latent_10k else None,
        "mean_vae_recon_cd": float(np.mean(vae_recon)) if vae_recon else None,
        "mean_old_teacher_cd": float(np.mean(old_cds)) if old_cds else None,
        "mean_time": float(np.mean(times)),
        "per_frame": results,
    }
    with open(os.path.join(args.output_dir, "metrics_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
