"""Evaluate multi-token VAE: reconstruction CD + BEV plots."""
import torch, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.point_cloud_vae import PointCloudVAE
from models.refinement_net import chamfer_distance
from data.semantickitti import SemanticKITTI, collate_fn
from torch.utils.data import DataLoader

CKPT = "checkpoints/point_vae_multitoken/best_point_vae.pth"
DATA = "/home/anywherevla/sonata_ws/dataset/sonata_depth_pro"
OUT = "evaluation_vae_multitoken"
N = 20

def bev(ax, pts, title):
    if pts is not None and len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3, cmap="viridis", vmin=-3, vmax=5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)

def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]

    # Infer architecture from checkpoint
    latent_dim = ck.get("latent_dim", 1024)
    num_decoded_points = ck.get("num_decoded_points", 8000)
    num_latent_tokens = ck.get("num_latent_tokens", 16)
    internal_dim = ck.get("internal_dim", 256)
    num_heads = ck.get("num_heads", 4)
    num_dec_blocks = ck.get("num_dec_blocks", 3)

    print("Loading VAE: latent_dim={}, K={}, tokens={}, internal={}".format(
        latent_dim, num_decoded_points, num_latent_tokens, internal_dim))

    vae = PointCloudVAE(
        latent_dim=latent_dim,
        num_decoded_points=num_decoded_points,
        num_latent_tokens=num_latent_tokens,
        internal_dim=internal_dim,
        num_heads=num_heads,
        num_dec_blocks=num_dec_blocks,
    )
    vae.load_state_dict(sd)
    vae = vae.to(device).eval()
    print("Loaded (epoch {}, val {:.4f})".format(ck.get("epoch", "?"), ck.get("best_val_loss", -1)))

    ds = SemanticKITTI(root=DATA, split="val", use_ground_truth_maps=True,
                       augmentation=False, use_point_cloud=True,
                       point_max_partial=20000, point_max_complete=8000)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    cds = []
    step = max(1, len(ds) // N)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % step != 0:
                continue
            if len(cds) >= N:
                break

            cc = batch["complete_coord"].to(device)
            cb = batch["complete_batch"].to(device)
            pc = batch["partial_coord"].to(device)

            mu, logvar = vae.encode_batched(cc, cb, 1)
            recon = vae.decode(mu)
            if recon.dim() == 3:
                recon = recon.squeeze(0)

            cd = chamfer_distance(recon, cc).item()
            cds.append(cd)

            # BEV: Input | Reconstruction | GT
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            bev(axes[0], pc.cpu().numpy(), "Input (partial)")
            bev(axes[1], recon.cpu().numpy(), "Multi-token VAE (CD={:.3f})".format(cd))
            bev(axes[2], cc.cpu().numpy(), "Ground Truth ({} pts)".format(cc.shape[0]))
            fid = "{:06d}".format(i)
            plt.suptitle("Frame {} - Multi-token VAE Reconstruction".format(fid), fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, "bev_{}.png".format(fid)), dpi=150, bbox_inches="tight")
            plt.close()
            print("Sample {} (frame {}): CD = {:.4f}".format(len(cds), fid, cd))

    cds = np.array(cds)
    print("\n=== Multi-token VAE Results ({} samples) ===".format(len(cds)))
    print("Mean CD: {:.4f} +/- {:.4f}".format(cds.mean(), cds.std()))
    print("Min: {:.4f}, Max: {:.4f}".format(cds.min(), cds.max()))

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(cds)), cds, color="steelblue")
    ax.axhline(cds.mean(), color="red", linestyle="--", label="Mean={:.3f}".format(cds.mean()))
    ax.axhline(2.99, color="orange", linestyle=":", label="Old VAE=2.99")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Chamfer Distance")
    ax.set_title("Multi-token VAE vs Old VAE ({} val samples)".format(len(cds)))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "summary_cd.png"), dpi=150)
    plt.close()

    # Also check latent diversity
    print("\nLatent diagnostics:")
    mus = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            cc = batch["complete_coord"].to(device)
            cb = batch["complete_batch"].to(device)
            mu, _ = vae.encode_batched(cc, cb, 1)
            mus.append(mu.cpu())
    mus = torch.stack(mus).squeeze(1)
    print("  Latent dim: {}".format(mus.shape[1]))
    print("  Mean norm: {:.4f}".format(mus.norm(dim=1).mean()))
    print("  Std across samples: {:.4f}".format(mus.std(dim=0).mean()))
    pairwise = []
    for i in range(len(mus)):
        for j in range(i+1, len(mus)):
            pairwise.append((mus[i] - mus[j]).norm().item())
    print("  Mean pairwise L2: {:.4f}".format(np.mean(pairwise)))
    print("Saved to {}/".format(OUT))

if __name__ == "__main__":
    main()
