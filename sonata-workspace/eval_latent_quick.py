"""Quick eval: load best latent diffusion checkpoint, run inference on N val samples, compute CD."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.semantickitti import SemanticKITTI, collate_fn
from torch.utils.data import DataLoader
from models.latent_diffusion import SceneCompletionLatentDiffusion
from models.point_cloud_vae import PointCloudVAE
from models.sonata_encoder import ConditionalFeatureExtractor, SonataEncoder
from models.refinement_net import chamfer_distance

CKPT = "checkpoints/latent_diffusion_gaussian/best_latent_diffusion.pth"
DATA = "/home/anywherevla/sonata_ws/dataset/sonata_depth_pro"
N_SAMPLES = 10
DDIM_STEPS = 50

def build_partial_dict(batch, voxel_size=0.05):
    coord = batch["partial_coord"]
    gc = torch.floor(coord / voxel_size).long()
    bi = batch["partial_batch"]
    for b in bi.unique():
        mask = bi == b
        gc[mask] -= gc[mask].min(dim=0)[0]
    return {"coord": coord, "color": batch["partial_color"],
            "normal": batch["partial_normal"], "grid_coord": gc, "batch": bi}

@torch.no_grad()
def main():
    device = torch.device("cuda")
    print("Loading checkpoint...")
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck["model_state_dict"]

    # Infer VAE params
    decoder_w = sd["vae.decoder_out.weight"]
    K = decoder_w.shape[0] // 3
    latent_dim = sd["vae.fc_mu.weight"].shape[0]
    print(f"VAE: latent_dim={latent_dim}, K={K}")

    # Build model
    encoder = SonataEncoder(pretrained="facebook/sonata", freeze=True,
                            enable_flash=False, feature_levels=[0])
    cond_ext = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    vae = PointCloudVAE(latent_dim=latent_dim, num_decoded_points=K)

    # Infer denoiser hparams
    hidden_dim = sd["denoiser.input_proj.weight"].shape[0]
    num_latent_tokens = sd["denoiser.pos_embed"].shape[1]
    time_embed_dim = sd["denoiser.time_proj.0.weight"].shape[1]
    num_blocks = sum(1 for k in sd if k.startswith("denoiser.blocks.") and k.endswith(".adaln_sa.norm.bias"))
    num_cond_tokens = sd["cond_pooler.query"].shape[1]

    model = SceneCompletionLatentDiffusion(
        vae=vae, condition_extractor=cond_ext,
        num_timesteps=ck.get("num_timesteps", 1000),
        schedule=ck.get("schedule", "cosine"),
        denoising_steps=DDIM_STEPS,
        hidden_dim=hidden_dim,
        num_denoiser_blocks=num_blocks,
        num_latent_tokens=num_latent_tokens,
        num_cond_tokens=num_cond_tokens,
        num_heads=4,
        time_embed_dim=time_embed_dim,
    ).to(device)

    model.load_state_dict(sd)
    model.eval()
    print(f"Model loaded. Denoiser: {hidden_dim}d, {num_blocks} blocks, {num_latent_tokens} tokens")

    # Dataset
    ds = SemanticKITTI(root=DATA, split="val", use_ground_truth_maps=True,
                       augmentation=False, use_point_cloud=True,
                       point_max_partial=20000, point_max_complete=8000)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    cds = []
    for i, batch in enumerate(loader):
        if i >= N_SAMPLES:
            break
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        partial = build_partial_dict(batch)
        gt = batch["complete_coord"]

        # Run inference
        pred = model.complete_scene(partial, num_steps=DDIM_STEPS, sampling="ddim", eta=0.0)
        if pred.dim() == 3:
            pred = pred.squeeze(0)

        cd = chamfer_distance(pred, gt).item()
        cds.append(cd)
        print(f"Sample {i}: CD = {cd:.4f}, pred shape = {pred.shape}")

    cds = np.array(cds)
    print(f"\n=== Results ({len(cds)} samples) ===")
    print(f"Mean CD: {cds.mean():.4f} +/- {cds.std():.4f}")
    print(f"Min: {cds.min():.4f}, Max: {cds.max():.4f}")

if __name__ == "__main__":
    main()
