"""
Train PointCloudVAE v3 — multi-token latent + cross-attention decoder.

Fixes over v2 (val 2.48):
  1. Per-sample GT centering (removes 11.5m x-offset)
  2. Per-sample scale normalization to [-1, 1] (stabilizes gradients)
  3. LR warmup (10 epochs linear ramp, prevents early explosions)
  4. 32 latent tokens (more spatial resolution)
  5. 5 decoder blocks (more refinement capacity)
  6. beta_kl=1e-3 (prevent posterior collapse, was 1e-4)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.semantickitti import SemanticKITTI, collate_fn
from models.point_cloud_vae import PointCloudVAE, vae_reconstruction_chamfer
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Train point cloud VAE v3 (multi-token, centered)")
    p.add_argument("--data_path", type=str,
                    default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=10)

    # --- VAE architecture ---
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--num_latent_tokens", type=int, default=32)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_dec_blocks", type=int, default=5)

    # --- loss ---
    p.add_argument("--beta_kl", type=float, default=1e-3)

    # --- data ---
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--point_max_partial", type=int, default=20000)

    # --- normalization ---
    p.add_argument("--center_gt", action="store_true", default=True,
                    help="Center GT on its own mean before encoding")
    p.add_argument("--scale_gt", action="store_true", default=True,
                    help="Scale GT to [-1,1] before encoding")

    # --- IO ---
    p.add_argument("--output_dir", type=str, default="checkpoints/point_vae_v3")
    p.add_argument("--log_dir", type=str, default="logs/point_vae_v3")
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)

    # --- GT version ---
    p.add_argument("--gt_subdir", type=str, default="ground_truth",
                    help="Subdirectory for GT maps (ground_truth or ground_truth_v2)")
    p.add_argument("--gt_name_suffix", type=str, default="",
                    help="Suffix for GT filenames (_v2 for boost v2)")
    return p.parse_args()


def _arch_extra(args) -> dict:
    return {
        "latent_dim": args.latent_dim,
        "num_decoded_points": args.num_decoded_points,
        "num_latent_tokens": args.num_latent_tokens,
        "internal_dim": args.internal_dim,
        "num_heads": args.num_heads,
        "num_dec_blocks": args.num_dec_blocks,
        "center_gt": args.center_gt,
        "scale_gt": args.scale_gt,
    }


def normalize_points(pts, center=True, scale=True):
    """Center and/or scale a point cloud. Returns (pts_norm, centroid, scale_factor)."""
    centroid = pts.mean(dim=0) if center else torch.zeros(3, device=pts.device)
    pts_c = pts - centroid
    if scale:
        s = pts_c.abs().max().clamp(min=1e-6)
    else:
        s = torch.ones(1, device=pts.device)
    pts_n = pts_c / s
    return pts_n, centroid, s


def train_epoch(model, loader, opt, epoch, args, writer, device):
    model.train()
    tot = 0.0
    n = 0
    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        cc = batch["complete_coord"]
        cb = batch["complete_batch"]
        bsz = int(cb.max().item()) + 1

        opt.zero_grad()
        loss_b = 0.0
        c_b = 0.0
        kl_b = 0.0
        for b in range(bsz):
            m = cb == b
            pts = cc[m]

            # --- Per-sample normalization ---
            pts_norm, _, _ = normalize_points(pts, center=args.center_gt, scale=args.scale_gt)

            recon, mu, logvar = model(pts_norm)
            total, c_term, kl_term = vae_reconstruction_chamfer(
                recon.unsqueeze(0),
                pts_norm.unsqueeze(0),
                mu.unsqueeze(0),
                logvar.unsqueeze(0),
                beta_kl=args.beta_kl,
            )
            (total / bsz).backward()
            loss_b += total.item()
            c_b += c_term.item()
            kl_b += kl_term.item()

        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        opt.step()

        tot += loss_b / bsz
        n += 1
        step = epoch * len(loader) + i
        writer.add_scalar("train/loss", loss_b / bsz, step)
        writer.add_scalar("train/chamfer", c_b / bsz, step)
        writer.add_scalar("train/kl", kl_b / bsz, step)

    return tot / max(n, 1)


@torch.no_grad()
def validate(model, loader, args, device):
    model.eval()
    tot = 0.0
    n = 0
    for batch in tqdm(loader, desc="Val"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc = batch["complete_coord"]
        cb = batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        loss_b = 0.0
        for b in range(bsz):
            m = cb == b
            pts = cc[m]
            pts_norm, _, _ = normalize_points(pts, center=args.center_gt, scale=args.scale_gt)
            recon, mu, logvar = model(pts_norm)
            total, _, _ = vae_reconstruction_chamfer(
                recon.unsqueeze(0),
                pts_norm.unsqueeze(0),
                mu.unsqueeze(0),
                logvar.unsqueeze(0),
                beta_kl=args.beta_kl,
            )
            loss_b += total.item() / bsz
        tot += loss_b
        n += 1
    return tot / max(n, 1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset kwargs for GT version support
    ds_kwargs = {}
    if args.gt_subdir != "ground_truth":
        ds_kwargs["gt_subdir"] = args.gt_subdir
    if args.gt_name_suffix:
        ds_kwargs["gt_name_suffix"] = args.gt_name_suffix

    ds_train = SemanticKITTI(
        root=args.data_path, split="train", use_ground_truth_maps=True,
        augmentation=True, use_point_cloud=True,
        point_max_partial=args.point_max_partial,
        point_max_complete=args.point_max_complete,
        **ds_kwargs,
    )
    ds_val = SemanticKITTI(
        root=args.data_path, split="val", use_ground_truth_maps=True,
        augmentation=False, use_point_cloud=True,
        point_max_partial=args.point_max_partial,
        point_max_complete=args.point_max_complete,
        **ds_kwargs,
    )

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    model = PointCloudVAE(
        latent_dim=args.latent_dim,
        num_decoded_points=args.num_decoded_points,
        num_latent_tokens=args.num_latent_tokens,
        internal_dim=args.internal_dim,
        num_heads=args.num_heads,
        num_dec_blocks=args.num_dec_blocks,
    ).to(device)

    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {nparams:,}")
    logger.info(f"Fixes: center_gt={args.center_gt}, scale_gt={args.scale_gt}, "
                f"warmup={args.warmup_epochs}, tokens={args.num_latent_tokens}, "
                f"dec_blocks={args.num_dec_blocks}, beta_kl={args.beta_kl}")

    opt = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    # Warmup + cosine schedule
    warmup_sch = optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, total_iters=args.warmup_epochs,
    )
    cosine_sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6,
    )
    sch = optim.lr_scheduler.SequentialLR(
        opt, [warmup_sch, cosine_sch], milestones=[args.warmup_epochs],
    )

    start = 0
    best = float("inf")
    if args.resume:
        ck = load_checkpoint(args.resume, model, opt, sch)
        start = ck.get("epoch", 0) + 1
        best = ck.get("best_val_loss", float("inf"))

    writer = SummaryWriter(args.log_dir)
    extra = _arch_extra(args)

    for epoch in range(start, args.num_epochs):
        tr = train_epoch(model, train_loader, opt, epoch, args, writer, device)
        va = validate(model, val_loader, args, device)
        writer.add_scalar("val/loss", va, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
        sch.step()
        logger.info(f"Epoch {epoch} train {tr:.6f} val {va:.6f} lr {opt.param_groups[0]['lr']:.6f}")

        if va < best:
            best = va
            path = os.path.join(args.output_dir, "best_point_vae.pth")
            save_checkpoint(
                path, model, opt, sch, epoch, best_val_loss=best,
                additional_info=extra,
            )
            logger.info(f"Saved {path}")
        if (epoch + 1) % args.save_freq == 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(
                path, model, opt, sch, epoch, best_val_loss=best,
                additional_info=extra,
            )

    writer.close()


if __name__ == "__main__":
    main()
