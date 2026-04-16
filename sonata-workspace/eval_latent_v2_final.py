#!/usr/bin/env python3
"""Evaluate latent diffusion v2 (DiT denoiser) vs old direct diffusion."""
import os, sys, torch, numpy as np, argparse, time, json, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]

def load_gt(path):
    return np.load(path)["points"]

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

def compute_cd(pred, gt, max_pts=10000):
    from models.refinement_net import chamfer_distance
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
    parser.add_argument("--latent_ckpt", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--da2_cloud_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="evaluation_results_latent_v2_final")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sequence", type=str, default="08")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # Load latent diffusion model
    print("Loading latent diffusion v2 model...")
    from models.latent_diffusion import SceneCompletionLatentDiffusion
    from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
    from models.point_cloud_vq_vae import PointCloudVQVAE
    from models.point_cloud_vae import PointCloudVAE

    ck = torch.load(args.latent_ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]

    # Infer VAE type
    dec_w = sd["vae.decoder_out.weight"]
    K = dec_w.shape[0] // 3
    if "vae.codebook.weight" in sd:
        cb = sd["vae.codebook.weight"]
        latent_dim, num_codes = cb.shape[1], cb.shape[0]
        vae = PointCloudVQVAE(latent_dim=latent_dim, num_codes=num_codes, num_decoded_points=K)
        print(f"VQ-VAE: latent_dim={latent_dim}, num_codes={num_codes}, K={K}")
    else:
        latent_dim = sd["vae.fc_mu.weight"].shape[0]
        vae = PointCloudVAE(latent_dim=latent_dim, num_decoded_points=K)
        print(f"Gaussian VAE: latent_dim={latent_dim}, K={K}")

    encoder = SonataEncoder(pretrained="facebook/sonata", freeze=True, enable_flash=False, feature_levels=[0])
    cond_ext = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")

    latent_model = SceneCompletionLatentDiffusion(
        vae=vae, condition_extractor=cond_ext,
        num_timesteps=ck.get("num_timesteps", 1000),
        schedule=ck.get("schedule", "cosine"),
        denoising_steps=50,
        hidden_dim=ck.get("hidden_dim", 1024),
        num_denoiser_blocks=ck.get("num_denoiser_blocks", 8),
        num_latent_tokens=ck.get("num_latent_tokens", 8),
        num_cond_tokens=ck.get("num_cond_tokens", 32),
        num_heads=ck.get("num_heads", 4),
        time_embed_dim=ck.get("time_embed_dim", 256),
    )
    latent_model.load_state_dict(sd)
    latent_model = latent_model.to(device).eval()
    print("Latent diffusion v2 loaded.")

    # Load old direct diffusion teacher if provided
    old_teacher = None
    if args.teacher_ckpt:
        print("Loading old direct diffusion teacher...")
        from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
        from models.diffusion_module import SceneCompletionDiffusion
        enc2 = SonataEncoder(pretrained="facebook/sonata", freeze=True, enable_flash=False, feature_levels=[0])
        cond2 = ConditionalFeatureExtractor(enc2, feature_levels=[0], fusion_type="concat")
        old_teacher = SceneCompletionDiffusion(encoder=enc2, condition_extractor=cond2, num_timesteps=1000, schedule="cosine", denoising_steps=50)
        t_ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
        old_teacher.load_state_dict(t_ckpt["model_state_dict"])
        old_teacher = old_teacher.to(device).eval()
        print("Old teacher loaded.")

    # Get val frames
    vel_dir = os.path.join(args.data_path, "sequences", args.sequence, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", args.sequence)
    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])
    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][:args.num_samples]
    print(f"Evaluating {len(sample_frames)} frames")

    results = []
    for i, fid in enumerate(sample_frames):
        print(f"\n--- Frame {fid} ({i+1}/{len(sample_frames)}) ---")

        lidar = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        gt = load_gt(gt_path) if os.path.exists(gt_path) else None

        r = {"frame": fid}

        # Latent diffusion v2
        with torch.no_grad():
            pt_dict, center = prepare_scan(lidar, device)
            t0 = time.time()
            latent_comp = latent_model.complete_scene(pt_dict)
            latent_time = time.time() - t0
            latent_pts = latent_comp.cpu().numpy() + center

        r["latent_time"] = latent_time
        r["latent_num_pts"] = latent_pts.shape[0]
        if gt is not None:
            r["latent_cd"] = compute_cd(latent_pts, gt)
            print(f"  Latent v2: CD={r['latent_cd']:.4f}, time={latent_time:.2f}s, pts={latent_pts.shape[0]}")

        # Old teacher (single-step x0)
        if old_teacher and gt is not None:
            with torch.no_grad():
                pt_dict_t, center_t = prepare_scan(lidar, device)
                # Prepare GT target coords
                gt_centered = gt - lidar.mean(axis=0)
                vc = np.floor(gt_centered / 0.05).astype(np.int32)
                _, idx = np.unique(vc, axis=0, return_index=True)
                gt_sub = gt_centered[idx]
                if gt_sub.shape[0] > 20000:
                    sel = np.random.choice(gt_sub.shape[0], 20000, replace=False)
                    gt_sub = gt_sub[sel]
                gt_target = torch.from_numpy(gt_sub).float().to(device)

                cond_features, _ = old_teacher.condition_extractor(pt_dict_t)
                from models.diffusion_module import knn_interpolate
                cond_mapped = knn_interpolate(cond_features, pt_dict_t['coord'], gt_target)
                old_teacher.scheduler._to_device(device)

                t_val = 200
                noise = torch.randn_like(gt_target)
                sa = old_teacher.scheduler.sqrt_alphas_cumprod[t_val]
                som = old_teacher.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
                noisy = sa * gt_target + som * noise

                t0 = time.time()
                t_tensor = torch.full((1,), t_val, device=device)
                pred_noise = old_teacher.denoiser(noisy, gt_target, t_tensor, {'features': cond_mapped})
                teacher_time = time.time() - t0
                pred_x0 = (noisy - som * pred_noise) / sa
                teacher_pts = pred_x0.cpu().numpy() + center_t

            r["teacher_time"] = teacher_time
            r["teacher_cd"] = compute_cd(teacher_pts, gt)
            print(f"  Old teacher: CD={r['teacher_cd']:.4f}, time={teacher_time:.2f}s")

        # Viz
        viz = {"Input": lidar, "Latent v2": latent_pts}
        if old_teacher and gt is not None:
            viz["Old Teacher"] = teacher_pts
        if gt is not None:
            viz["GT"] = gt
        bev_plot(viz, f"Frame {fid}", os.path.join(args.output_dir, f"comparison_{fid}.png"))
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Frame':<10} {'Latent v2 CD':<15} {'Old Teacher CD':<15} {'Latent t':<12} {'Teacher t':<12}")
    print("=" * 70)
    for r in results:
        lcd = f"{r['latent_cd']:.4f}" if "latent_cd" in r else "-"
        tcd = f"{r['teacher_cd']:.4f}" if "teacher_cd" in r else "-"
        lt = f"{r['latent_time']:.2f}s"
        tt = f"{r['teacher_time']:.2f}s" if "teacher_time" in r else "-"
        print(f"{r['frame']:<10} {lcd:<15} {tcd:<15} {lt:<12} {tt:<12}")

    lcds = [r["latent_cd"] for r in results if "latent_cd" in r]
    tcds = [r["teacher_cd"] for r in results if "teacher_cd" in r]
    print(f"\nAverages:")
    if lcds:
        print(f"  Latent v2 CD: {np.mean(lcds):.4f} +/- {np.std(lcds):.4f}")
    if tcds:
        print(f"  Old Teacher CD: {np.mean(tcds):.4f} +/- {np.std(tcds):.4f}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
