#!/usr/bin/env python3
"""
Diagnostic: test if the latent diffusion denoiser actually learned,
by doing single-step x_0 prediction at moderate noise level.
"""
import os, sys, torch, numpy as np
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

def build_model(ckpt_path, device="cuda"):
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
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    model = SceneCompletionLatentDiffusion(
        vae=vae, condition_extractor=cond,
        num_timesteps=num_t, schedule=sched, denoising_steps=50,
    )
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded: latent_dim={latent_dim}, K={k}, T={num_t}")
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

@torch.no_grad()
def main():
    device = "cuda"
    base = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
    ckpt = os.path.join(base, "checkpoints/latent_diffusion/best_model.pth")
    
    model, K = build_model(ckpt, device)
    model._move_scheduler(device)
    sched = model.scheduler
    
    # Load a test frame
    data_path = os.path.join(base, "data/semantic_kitti")
    vel_dir = os.path.join(data_path, "sequences/08/velodyne")
    gt_dir = os.path.join(data_path, "ground_truth/08")
    
    frames = sorted(os.listdir(vel_dir))
    test_frame = frames[200].replace(".bin", "")
    print(f"\nTest frame: {test_frame}")
    
    lidar = np.fromfile(os.path.join(vel_dir, f"{test_frame}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]
    gt_raw = np.load(os.path.join(gt_dir, f"{test_frame}.npz"))["points"]
    
    scan_dict, center = prepare_scan(lidar, device)
    gt_centered = gt_raw - center
    
    # 1) VAE reconstruction baseline
    if gt_centered.shape[0] > 8000:
        sel = np.random.choice(gt_centered.shape[0], 8000, replace=False)
        gt_sub = gt_centered[sel]
    else:
        gt_sub = gt_centered
    mu, logvar = model.vae.encode(torch.from_numpy(gt_sub).float().to(device))
    z0_clean = mu  # (1, latent_dim) -- the clean latent
    vae_recon = model.vae.decode(mu).cpu().numpy()
    
    # CD: VAE recon vs GT
    vae_cd = chamfer_distance(
        torch.from_numpy(vae_recon).float().cuda(),
        torch.from_numpy(gt_centered[:2048]).float().cuda(),
        chunk_size=512,
    ).item()
    print(f"\n=== VAE Reconstruction Baseline ===")
    print(f"  CD (VAE encode-decode GT): {vae_cd:.4f}")
    
    # 2) Encode condition from partial scan
    cond = model.encode_condition(scan_dict)  # (1, cond_dim)
    print(f"  Condition shape: {cond.shape}")
    print(f"  z0_clean shape: {z0_clean.shape}, range: [{z0_clean.min():.2f}, {z0_clean.max():.2f}]")
    
    # 3) Single-step denoising test at various noise levels
    print(f"\n=== Single-Step x0 Prediction Test ===")
    print(f"{t:>5} {alpha_bar:>10} {noise_level:>12} {CD_pred:>10} {CD_noisy:>10} {CD_random:>10}")
    print("-" * 60)
    
    for t_val in [50, 100, 200, 300, 500, 700, 999]:
        t = torch.tensor([t_val], device=device, dtype=torch.long)
        alpha_bar = sched.alphas_cumprod[t_val]
        sqrt_ab = sched.sqrt_alphas_cumprod[t_val]
        sqrt_om = sched.sqrt_one_minus_alphas_cumprod[t_val]
        
        # Add noise to z0_clean
        noise = torch.randn_like(z0_clean)
        z_t = sqrt_ab * z0_clean + sqrt_om * noise
        
        # Predict noise
        pred_noise = model.denoiser(z_t, t, cond)
        
        # Reconstruct z0
        z0_pred = (z_t - sqrt_om * pred_noise) / sqrt_ab
        
        # Also try: just decode z_t directly (noisy latent)
        pts_pred = model.vae.decode(z0_pred).cpu().numpy()
        pts_noisy = model.vae.decode(z_t).cpu().numpy()
        
        # Random latent for comparison
        z_rand = torch.randn_like(z0_clean)
        pts_rand = model.vae.decode(z_rand).cpu().numpy()
        
        gt_eval = gt_centered[:2048] if gt_centered.shape[0] > 2048 else gt_centered
        
        cd_pred = chamfer_distance(
            torch.from_numpy(pts_pred).float().cuda(),
            torch.from_numpy(gt_eval).float().cuda(), chunk_size=512).item()
        cd_noisy = chamfer_distance(
            torch.from_numpy(pts_noisy).float().cuda(),
            torch.from_numpy(gt_eval).float().cuda(), chunk_size=512).item()
        cd_rand = chamfer_distance(
            torch.from_numpy(pts_rand).float().cuda(),
            torch.from_numpy(gt_eval).float().cuda(), chunk_size=512).item()
        
        noise_level = sqrt_om.item() / sqrt_ab.item()
        print(f"{t_val:>5} {alpha_bar.item():>10.6f} {noise_level:>12.4f} {cd_pred:>10.4f} {cd_noisy:>10.4f} {cd_rand:>10.4f}")
    
    # 4) Full pipeline with current (buggy) sampling
    print(f"\n=== Full Pipeline (50-step skip sampling) ===")
    completed = model.complete_scene(scan_dict, num_steps=50)
    if isinstance(completed, torch.Tensor):
        completed = completed.cpu().numpy()
    cd_full = chamfer_distance(
        torch.from_numpy(completed).float().cuda(),
        torch.from_numpy(gt_eval).float().cuda(), chunk_size=512).item()
    print(f"  CD (50-step skip): {cd_full:.4f}")
    
    # 5) Full pipeline with ALL 1000 steps (consecutive, should be correct)
    print(f"\n=== Full Pipeline (1000-step consecutive) ===")
    completed_1000 = model.complete_scene(scan_dict, num_steps=1000)
    if isinstance(completed_1000, torch.Tensor):
        completed_1000 = completed_1000.cpu().numpy()
    cd_1000 = chamfer_distance(
        torch.from_numpy(completed_1000).float().cuda(),
        torch.from_numpy(gt_eval).float().cuda(), chunk_size=512).item()
    print(f"  CD (1000-step consecutive): {cd_1000:.4f}")
    
    # 6) Summary
    print(f"\n{=*60}")
    print(f"SUMMARY")
    print(f"{=*60}")
    print(f"VAE recon (upper bound):    {vae_cd:.4f}")
    print(f"Single-step t=200 pred:     (see table above)")
    print(f"50-step skip sampling:      {cd_full:.4f}")
    print(f"1000-step consecutive:      {cd_1000:.4f}")
    print(f"Random latent decode:       (see table above)")
    print()
    if cd_1000 < cd_full * 0.5:
        print("DIAGNOSIS: Skip-step sampling is broken! 1000-step works much better.")
        print("The posterior formula uses consecutive-step coefficients but the timesteps skip ~20 steps.")
    elif cd_1000 > 50 and cd_full > 50:
        print("DIAGNOSIS: Both sampling chains produce blobs. Check if denoiser learned.")
        print("Single-step test results above show whether the denoiser can predict noise.")
    else:
        print("DIAGNOSIS: Sampling seems ok. Check other issues.")

if __name__ == "__main__":
    main()
