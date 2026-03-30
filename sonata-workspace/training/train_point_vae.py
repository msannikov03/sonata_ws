"""
Train PointCloudVAE (multi-token latent + cross-attention decoder)
on complete point clouds.
"""

from __future__ import annotations

import argparse
import os
import sys

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
    p = argparse.ArgumentParser(description="Train point cloud VAE (multi-token)")
    p.add_argument(
        "--data_path",
        type=str,
        default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"),
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)

    # --- VAE architecture ---
    p.add_argument("--latent_dim", type=int, default=1024,
                    help="Total latent dim = num_latent_tokens × token_dim")
    p.add_argument("--num_decoded_points", type=int, default=4096,
                    help="Increase toward target count for lower CD")
    p.add_argument("--num_latent_tokens", type=int, default=16)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_dec_blocks", type=int, default=3)

    # --- loss ---
    p.add_argument("--beta_kl", type=float, default=1e-4,
                    help="KL weight; keep low for reconstruction quality")

    # --- data ---
    p.add_argument(
        "--point_max_complete", type=int, default=8000,
        help="Subsample GT complete cloud to this many points (Chamfer target)",
    )
    p.add_argument(
        "--point_max_partial", type=int, default=20000,
        help="Partial scan cap (unused for VAE loss, kept for dataloader parity)",
    )

    # --- IO ---
    p.add_argument("--output_dir", type=str, default="checkpoints/point_vae")
    p.add_argument("--log_dir", type=str, default="logs/point_vae")
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def _arch_extra(args) -> dict:
    """Architecture metadata to save in checkpoints."""
    return {
        "latent_dim": args.latent_dim,
        "num_decoded_points": args.num_decoded_points,
        "num_latent_tokens": args.num_latent_tokens,
        "internal_dim": args.internal_dim,
        "num_heads": args.num_heads,
        "num_dec_blocks": args.num_dec_blocks,
    }


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
            recon, mu, logvar = model(pts)
            total, c_term, kl_term = vae_reconstruction_chamfer(
                recon.unsqueeze(0),
                pts.unsqueeze(0),
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
            recon, mu, logvar = model(pts)
            total, _, _ = vae_reconstruction_chamfer(
                recon.unsqueeze(0),
                pts.unsqueeze(0),
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

    ds_train = SemanticKITTI(
        root=args.data_path,
        split="train",
        use_ground_truth_maps=True,
        augmentation=True,
        use_point_cloud=True,
        point_max_partial=args.point_max_partial,
        point_max_complete=args.point_max_complete,
    )
    ds_val = SemanticKITTI(
        root=args.data_path,
        split="val",
        use_ground_truth_maps=True,
        augmentation=False,
        use_point_cloud=True,
        point_max_partial=args.point_max_partial,
        point_max_complete=args.point_max_complete,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
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

    opt = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)

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
        sch.step()
        logger.info(f"Epoch {epoch} train {tr:.4f} val {va:.4f}")

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
