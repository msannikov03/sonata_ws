"""Test DDIM with clipped starting timestep to fix cosine schedule explosion."""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.latent_diffusion import SceneCompletionLatentDiffusion
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.point_cloud_vae import PointCloudVAE
from models.refinement_net import chamfer_distance
from data.semantickitti import SemanticKITTI, collate_fn
from torch.utils.data import DataLoader

device = "cuda"
ck = torch.load("checkpoints/latent_diffusion_gaussian/best_latent_diffusion.pth",
                 map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]

K = sd["vae.decoder_out.weight"].shape[0] // 3
latent_dim = sd["vae.fc_mu.weight"].shape[0]
vae = PointCloudVAE(latent_dim=latent_dim, num_decoded_points=K)
encoder = SonataEncoder(pretrained="facebook/sonata", freeze=True,
                        enable_flash=False, feature_levels=[0])
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

ds = SemanticKITTI(root="/home/anywherevla/sonata_ws/dataset/sonata_depth_pro",
                   split="val", use_ground_truth_maps=True, augmentation=False,
                   use_point_cloud=True, point_max_partial=20000, point_max_complete=8000)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)


@torch.no_grad()
def complete_clipped(model, partial, num_steps=50, max_t=980):
    device = next(model.denoiser.parameters()).device
    model._move_scheduler(device)
    batch = partial["batch"]
    bsz = int(batch.max().item()) + 1
    cond = model.encode_condition(partial, bsz)
    B = cond.size(0)
    z = torch.randn(B, model.latent_dim, device=device)

    ts = torch.linspace(max_t, 0, num_steps, dtype=torch.long, device=device)
    ts_list = sorted(set(int(v) for v in ts.tolist()), reverse=True)
    if 0 not in ts_list:
        ts_list.append(0)

    for i in range(len(ts_list) - 1):
        t_cur, t_prev = ts_list[i], ts_list[i + 1]
        t_vec = torch.full((B,), t_cur, device=device, dtype=torch.long)
        eps = model.denoiser(z, t_vec, cond)
        ab = model.scheduler.alphas_cumprod[t_cur]
        ab_prev = model.scheduler.alphas_cumprod[t_prev]
        x0_pred = (z - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)
        z = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps

    z = model.normalizer.denormalize(z)
    return model.vae.decode(z)


def make_partial(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    coord = batch["partial_coord"]
    gc = torch.floor(coord / 0.05).long()
    bi = batch["partial_batch"]
    for b in bi.unique():
        mask = bi == b
        gc[mask] -= gc[mask].min(dim=0)[0]
    return {"coord": coord, "color": batch["partial_color"],
            "normal": batch["partial_normal"], "grid_coord": gc, "batch": bi}


N = 5
for max_t in [980, 950, 900, 800, 500]:
    cds = []
    for i, batch in enumerate(loader):
        if i >= N:
            break
        partial = make_partial(batch, device)
        gt = batch["complete_coord"].to(device)
        pred = complete_clipped(model, partial, num_steps=50, max_t=max_t)
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        cd = chamfer_distance(pred, gt).item()
        cds.append(cd)
    avg = sum(cds) / len(cds)
    per = ", ".join("{:.2f}".format(c) for c in cds)
    print("max_t={}: mean CD = {:.4f} | {}".format(max_t, avg, per))
