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

model._move_scheduler(torch.device(device))

with torch.no_grad():
    # Get GT z0 for reference
    mu, logvar = model.vae.encode_batched(gt, batch["complete_batch"], 1)
    z0_gt = model.normalizer.normalize(mu)
    
    cond = model.encode_condition(partial, 1)
    
    # Check schedule values
    print("=== COSINE SCHEDULE CHECK ===")
    for t_val in [0, 100, 200, 500, 800, 999]:
        ab = model.scheduler.alphas_cumprod[t_val].item()
        print(f"  t={t_val}: alpha_cumprod={ab:.6f}, sqrt_ac={ab**0.5:.6f}, sqrt_1-ac={(1-ab)**0.5:.6f}")
    
    # Test denoiser at various t levels from pure noise
    print("\n=== DENOISER AT DIFFERENT TIMESTEPS from z_t = sqrt_ac * z0_gt + sqrt_1-ac * noise ===")
    noise = torch.randn(1, latent_dim, device=device)
    for t_val in [999, 800, 500, 200, 100, 50, 10]:
        t_vec = torch.full((1,), t_val, device=device, dtype=torch.long)
        ab = model.scheduler.alphas_cumprod[t_val].to(device)
        z_t = torch.sqrt(ab) * z0_gt + torch.sqrt(1-ab) * noise
        eps_pred = model.denoiser(z_t, t_vec, cond)
        
        # Reconstruct x0
        x0_pred = (z_t - torch.sqrt(1-ab) * eps_pred) / torch.sqrt(ab)
        err = (x0_pred - z0_gt).abs().mean().item()
        eps_err = (eps_pred - noise).abs().mean().item()
        print(f"  t={t_val}: eps_pred std={eps_pred.std():.4f}, eps_err={eps_err:.4f}, x0_err={err:.4f}, x0_pred norm={x0_pred.norm():.2f}")
    
    # Now trace the DDIM loop step by step
    print("\n=== DDIM TRAJECTORY TRACE (50 steps) ===")
    z = torch.randn(1, latent_dim, device=device)
    print(f"Initial z: mean={z.mean():.4f}, std={z.std():.4f}, norm={z.norm():.4f}")
    
    num_steps = 50
    ts = torch.linspace(model.scheduler.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
    ts_list = sorted(set(int(v) for v in ts.tolist()), reverse=True)
    if 0 not in ts_list:
        ts_list.append(0)
    
    for i in range(len(ts_list) - 1):
        t_cur = ts_list[i]
        t_prev = ts_list[i + 1]
        
        t_vec = torch.full((1,), t_cur, device=device, dtype=torch.long)
        eps = model.denoiser(z, t_vec, cond)
        
        ab = model.scheduler.alphas_cumprod[t_cur].to(device)
        ab_prev = model.scheduler.alphas_cumprod[t_prev].to(device)
        
        x0_pred = (z - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)
        z = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps
        
        if i < 5 or i >= len(ts_list) - 6 or i % 10 == 0:
            print(f"  step {i}: t={t_cur}->{t_prev}, z norm={z.norm():.2f}, x0 norm={x0_pred.norm():.2f}, x0 mean={x0_pred.mean():.4f}, eps std={eps.std():.4f}")
    
    # The final z is the result -- denorm and decode
    from models.refinement_net import chamfer_distance
    z_denorm = model.normalizer.denormalize(z)
    pts = model.vae.decode(z_denorm)
    if pts.dim() == 3: pts = pts.squeeze(0)
    cd = chamfer_distance(pts, gt.squeeze() if gt.dim()==3 else gt).item()
    print(f"\nFinal DDIM CD: {cd:.4f}")
    print(f"Final z: mean={z.mean():.4f}, std={z.std():.4f}, norm={z.norm():.4f}")
    print(f"GT z0 for ref: mean={z0_gt.mean():.4f}, std={z0_gt.std():.4f}, norm={z0_gt.norm():.4f}")
    
    # Now try: start from z0_gt + small noise, do a few reverse steps
    # This tests if partial DDIM works
    print("\n=== PARTIAL DDIM: start from t=300 with correct z_t ===")
    t_start = 300
    ab_start = model.scheduler.alphas_cumprod[t_start].to(device)
    z = torch.sqrt(ab_start) * z0_gt + torch.sqrt(1-ab_start) * noise
    
    # Only use timesteps <= t_start
    ts_partial = [t for t in ts_list if t <= t_start]
    for i in range(len(ts_partial) - 1):
        t_cur = ts_partial[i]
        t_prev = ts_partial[i + 1]
        t_vec = torch.full((1,), t_cur, device=device, dtype=torch.long)
        eps = model.denoiser(z, t_vec, cond)
        ab = model.scheduler.alphas_cumprod[t_cur].to(device)
        ab_prev = model.scheduler.alphas_cumprod[t_prev].to(device)
        x0_pred = (z - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)
        z = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps
    
    z_denorm = model.normalizer.denormalize(z)
    pts = model.vae.decode(z_denorm)
    if pts.dim() == 3: pts = pts.squeeze(0)
    cd_partial = chamfer_distance(pts, gt.squeeze() if gt.dim()==3 else gt).item()
    print(f"CD from partial DDIM starting t=300: {cd_partial:.4f}")
