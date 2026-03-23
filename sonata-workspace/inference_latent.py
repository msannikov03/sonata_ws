"""
Inference: latent diffusion + VAE decode -> fixed K point cloud (PLY).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.latent_diffusion import SceneCompletionLatentDiffusion
from models.point_cloud_vae import PointCloudVAE
from models.point_cloud_vq_vae import PointCloudVQVAE
from models.sonata_encoder import ConditionalFeatureExtractor, SonataEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Latent diffusion scene completion")
    p.add_argument("--input", type=str, required=True, help=".bin or .pcd/.ply scan")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="best_latent_diffusion.pth (full SceneCompletionLatentDiffusion)",
    )
    p.add_argument("--output", type=str, default="latent_completion.ply")
    p.add_argument("--denoising_steps", type=int, default=50)
    p.add_argument(
        "--sampling",
        type=str,
        default="ddim",
        choices=["ddim", "ddpm"],
        help="DDIM supports non-adjacent timesteps; DDPM uses t->t-1 steps.",
    )
    p.add_argument(
        "--voxel_size_sonata",
        type=float,
        default=0.05,
        help="grid_coord = floor(coord / this) for Sonata",
    )
    p.add_argument(
        "--max_input_points",
        type=int,
        default=20000,
        help="Random subsample partial scan to this cap (point mode, no voxel merge)",
    )
    p.add_argument("--encoder_ckpt", type=str, default="facebook/sonata")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--enable_flash", action="store_true")
    return p.parse_args()


def load_scan(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        s = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return s[:, :3].copy()
    if ext in (".pcd", ".ply"):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)
    raise ValueError(f"Unsupported format: {ext}")


def prepare_partial(
    scan: np.ndarray,
    voxel_size_sonata: float,
    max_points: int,
    device: torch.device,
) -> tuple[dict, np.ndarray]:
    center = scan.mean(axis=0)
    pts = scan - center
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    grid = np.floor(pts / voxel_size_sonata).astype(np.int64)
    grid -= grid.min(axis=0)  # Shift to non-negative (Sonata serialization requirement)
    z = pts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([z_norm, 1 - z_norm, 0.5 * np.ones_like(z_norm)], axis=1)
    data = {
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(len(pts), 3, device=device),
        "grid_coord": torch.from_numpy(grid).long().to(device),
        "batch": torch.zeros(len(pts), dtype=torch.long, device=device),
    }
    return data, center


def infer_architecture_from_state_dict(sd: dict) -> tuple[str, int, int, int]:
    """
    Returns:
        (vae_kind, latent_dim, num_codes, K)
    """
    dec_w = sd["vae.decoder_out.weight"]
    k = dec_w.shape[0] // 3

    if "vae.fc_mu.weight" in sd:
        latent_dim = sd["vae.fc_mu.weight"].shape[0]
        return "gaussian_vae", latent_dim, 0, k

    if "vae.codebook.weight" in sd:
        codebook_w = sd["vae.codebook.weight"]
        latent_dim = codebook_w.shape[1]
        num_codes = codebook_w.shape[0]
        return "vq_vae", latent_dim, num_codes, k

    raise KeyError(
        "Could not infer VAE kind from latent diffusion checkpoint. "
        "Expected 'vae.fc_mu.weight' or 'vae.codebook.weight'."
    )


def build_model_from_checkpoint(
    ckpt_path: str, args, device: torch.device
) -> SceneCompletionLatentDiffusion:
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model_state_dict"]
    vae_kind, latent_dim, num_codes, k = infer_architecture_from_state_dict(sd)
    num_t = ck.get("num_timesteps", 1000)
    sched = ck.get("schedule", "cosine")

    if vae_kind == "gaussian_vae":
        vae = PointCloudVAE(latent_dim=latent_dim, num_decoded_points=k)
    elif vae_kind == "vq_vae":
        vae = PointCloudVQVAE(
            latent_dim=latent_dim, num_codes=num_codes, num_decoded_points=k
        )
    else:
        raise ValueError(f"Unknown vae_kind: {vae_kind}")
    encoder = SonataEncoder(
        pretrained=args.encoder_ckpt,
        freeze=args.freeze_encoder,
        enable_flash=args.enable_flash,
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
        denoising_steps=args.denoising_steps,
    )
    model.load_state_dict(sd)
    return model.to(device)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan = load_scan(args.input)
    partial, center = prepare_partial(
        scan, args.voxel_size_sonata, args.max_input_points, device
    )

    model = build_model_from_checkpoint(args.checkpoint, args, device)
    model.eval()

    pts = model.complete_scene(
        partial,
        num_steps=args.denoising_steps,
        sampling=args.sampling,
    )
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    pts = pts + center

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"Wrote {pts.shape[0]} points to {args.output}")


if __name__ == "__main__":
    main()
