import torch, numpy as np, sys
sys.path.insert(0, ".")

device = "cuda"
ck = torch.load("checkpoints/latent_diffusion_gaussian/best_latent_diffusion.pth", map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]

from models.latent_diffusion import SceneCompletionLatentDiffusion
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.point_cloud_vae import PointCloudVAE

dec_w = sd["vae.decoder_out.weight"]
K = dec_w.shape[0] // 3
latent_dim = sd["vae.fc_mu.weight"].shape[0]
vae = PointCloudVAE(latent_dim=latent_dim, num_decoded_points=K)

encoder = SonataEncoder(pretrained="facebook/sonata", freeze=True, enable_flash=False, feature_levels=[0])
cond_ext = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")

model = SceneCompletionLatentDiffusion(
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
).to(device)
model.load_state_dict(sd)
model.eval()

from data.semantickitti import SemanticKITTI, collate_fn
from torch.utils.data import DataLoader
ds = SemanticKITTI(root="/home/anywherevla/sonata_ws/dataset/sonata_depth_pro", split="val",
                   use_ground_truth_maps=True, augmentation=False, use_point_cloud=True,
                   point_max_partial=20000, point_max_complete=8000)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
batch = next(iter(loader))
for k in batch:
    if isinstance(batch[k], torch.Tensor):
        batch[k] = batch[k].to(device)

coord = batch["partial_coord"]
gc = torch.floor(coord / 0.05).long()
bi = batch["partial_batch"]
for b in bi.unique():
    mask = bi == b
    gc[mask] -= gc[mask].min(dim=0)[0]
partial = {"coord": coord, "color": batch["partial_color"], "normal": batch["partial_normal"], "grid_coord": gc, "batch": bi}

gt = batch["complete_coord"]

with torch.no_grad():
    mu, logvar = model.vae.encode_batched(gt, batch["complete_batch"], 1)
    z0 = model.normalizer.normalize(mu)
    print("=== LATENT SPACE DIAGNOSTICS ===")
    print(f"GT z0 normalized: mean={z0.mean():.4f}, std={z0.std():.4f}, norm={z0.norm():.4f}")
    print(f"GT z0 shape: {z0.shape}")
    print(f"Normalizer running_mean norm: {model.normalizer.running_mean.norm():.4f}")
    rmv = model.normalizer.running_var
    print(f"Normalizer running_var: mean={rmv.mean():.6f}, min={rmv.min():.6f}, max={rmv.max():.6f}")
    
    # Use CPU indices for scheduler
    t_cpu = torch.tensor([100])
    t_gpu = t_cpu.to(device)
    noise = torch.randn_like(z0)
    sa = model.scheduler.sqrt_alphas_cumprod[t_cpu].unsqueeze(-1).to(device)
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_cpu].unsqueeze(-1).to(device)
    z_t = sa * z0 + som * noise
    
    cond = model.encode_condition(partial, 1)
    print(f"Condition shape: {cond.shape}, norm: {cond.norm():.4f}")
    eps_pred = model.denoiser(z_t, t_gpu, cond)
    
    ab = model.scheduler.alphas_cumprod[100].to(device)
    z0_pred = (z_t - torch.sqrt(1 - ab) * eps_pred) / torch.sqrt(ab)
    print(f"z0 pred from t=100: mean={z0_pred.mean():.4f}, std={z0_pred.std():.4f}, norm={z0_pred.norm():.4f}")
    print(f"z0 pred error MAE: {(z0_pred - z0).abs().mean():.4f}")
    
    from models.refinement_net import chamfer_distance
    z0_denorm = model.normalizer.denormalize(z0)
    z0_pred_denorm = model.normalizer.denormalize(z0_pred)
    
    pts_gt_recon = model.vae.decode(z0_denorm)
    pts_pred_recon = model.vae.decode(z0_pred_denorm)
    
    if pts_gt_recon.dim() == 3: pts_gt_recon = pts_gt_recon.squeeze(0)
    if pts_pred_recon.dim() == 3: pts_pred_recon = pts_pred_recon.squeeze(0)
    
    cd_gt_recon = chamfer_distance(pts_gt_recon, gt.squeeze() if gt.dim()==3 else gt).item()
    cd_pred_recon = chamfer_distance(pts_pred_recon, gt.squeeze() if gt.dim()==3 else gt).item()
    print(f"CD from GT mu decode - VAE floor: {cd_gt_recon:.4f}")
    print(f"CD from t=100 pred decode: {cd_pred_recon:.4f}")
    
    # Full DDIM from noise
    print("\n=== DDIM SAMPLING ===")
    pts_ddim = model.complete_scene(partial, num_steps=50, sampling="ddim", eta=0.0)
    if pts_ddim.dim() == 3: pts_ddim = pts_ddim.squeeze(0)
    cd_ddim = chamfer_distance(pts_ddim, gt.squeeze() if gt.dim()==3 else gt).item()
    print(f"CD from full DDIM 50 steps: {cd_ddim:.4f}")
    print(f"DDIM output range: [{pts_ddim.min():.2f}, {pts_ddim.max():.2f}]")
    print(f"DDIM output mean: {pts_ddim.mean(dim=0)}")
    
    # Single-step x0 from t=200
    print("\n=== SINGLE-STEP x0 ===")
    t200_cpu = torch.tensor([200])
    t200_gpu = t200_cpu.to(device)
    sa200 = model.scheduler.sqrt_alphas_cumprod[t200_cpu].unsqueeze(-1).to(device)
    som200 = model.scheduler.sqrt_one_minus_alphas_cumprod[t200_cpu].unsqueeze(-1).to(device)
    z_t200 = sa200 * z0 + som200 * noise
    eps200 = model.denoiser(z_t200, t200_gpu, cond)
    ab200 = model.scheduler.alphas_cumprod[200].to(device)
    z0_from_200 = (z_t200 - torch.sqrt(1-ab200) * eps200) / torch.sqrt(ab200)
    z0_200_denorm = model.normalizer.denormalize(z0_from_200)
    pts_200 = model.vae.decode(z0_200_denorm)
    if pts_200.dim() == 3: pts_200 = pts_200.squeeze(0)
    cd_200 = chamfer_distance(pts_200, gt.squeeze() if gt.dim()==3 else gt).item()
    print(f"CD from single-step x0 at t=200: {cd_200:.4f}")
    
    # Pure noise baseline
    z_noise = torch.randn_like(z0)
    z_noise_denorm = model.normalizer.denormalize(z_noise)
    pts_noise = model.vae.decode(z_noise_denorm)
    if pts_noise.dim() == 3: pts_noise = pts_noise.squeeze(0)
    cd_noise = chamfer_distance(pts_noise, gt.squeeze() if gt.dim()==3 else gt).item()
    print(f"CD from pure noise decode baseline: {cd_noise:.4f}")
    
    # Noise prediction quality
    print(f"\nNoise stats: mean={noise.mean():.4f}, std={noise.std():.4f}")
    print(f"eps_pred t=100 stats: mean={eps_pred.mean():.4f}, std={eps_pred.std():.4f}")
    
    # GT point cloud stats
    gt_flat = gt.squeeze() if gt.dim()==3 else gt
    print(f"\nGT point cloud range: [{gt_flat.min():.2f}, {gt_flat.max():.2f}]")
    print(f"GT point cloud mean: {gt_flat.mean(dim=0)}")
    print(f"GT point cloud std per dim: {gt_flat.std(dim=0)}")
