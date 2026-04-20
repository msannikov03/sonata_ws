"""
Train PointCloudVAEv2 on complete point clouds (Chamfer + KL).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.semantickitti import SemanticKITTI, collate_fn
from models.point_cloud_vae_v2 import (
    PointCloudVAEv2,
    vae_reconstruction_chamfer,
)
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Train PointCloudVAEv2")
    p.add_argument("--data_path", type=str,
                    default=os.path.expanduser("~/sonata_ws/dataset/sonata_depth_pro"))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)

    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--decoder_hidden", type=int, default=512)

    p.add_argument("--beta_kl", type=float, default=1e-3)

    p.add_argument("--max_points", type=int, default=8000,
                    help="Max points per cloud (dataset-level subsampling)")

    p.add_argument("--output_dir", type=str, default="checkpoints/point_vae_v2")
    p.add_argument("--log_dir", type=str, default="logs/point_vae_v2")
    p.add_argument("--save_freq", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def train_epoch(model, loader, opt, scaler, epoch, args, writer, device):
    model.train()
    total_loss = 0.0
    total_cd = 0.0
    total_kl = 0.0
    n = 0

    for i, batch in enumerate(tqdm(loader, desc=f"Train {epoch}")):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        cc = batch["complete_coord"]
        cb = batch["complete_batch"]
        bsz = int(cb.max().item()) + 1

        opt.zero_grad(set_to_none=True)

        loss_accum = 0.0
        cd_accum = 0.0
        kl_accum = 0.0

        for b in range(bsz):
            pts = cc[cb == b]                               # (N_b, 3)

            with autocast(enabled=args.fp16):
                recon, mu, logvar = model(pts)
                loss, cd, kl = vae_reconstruction_chamfer(
                    recon, pts, mu, logvar, beta_kl=args.beta_kl,
                )

            scaler.scale(loss / bsz).backward()
            loss_accum += loss.item()
            cd_accum += cd.item()
            kl_accum += kl.item()

        if args.gradient_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(opt)
        scaler.update()

        total_loss += loss_accum / bsz
        total_cd += cd_accum / bsz
        total_kl += kl_accum / bsz
        n += 1

        step = epoch * len(loader) + i
        writer.add_scalar("train/loss", loss_accum / bsz, step)
        writer.add_scalar("train/cd", cd_accum / bsz, step)
        writer.add_scalar("train/kl", kl_accum / bsz, step)

    return total_loss / max(n, 1), total_cd / max(n, 1), total_kl / max(n, 1)


@torch.no_grad()
def validate(model, loader, args, device):
    model.eval()
    total_loss = 0.0
    total_cd = 0.0
    total_kl = 0.0
    n = 0

    for batch in tqdm(loader, desc="Val"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        cc = batch["complete_coord"]
        cb = batch["complete_batch"]
        bsz = int(cb.max().item()) + 1

        loss_accum = 0.0
        cd_accum = 0.0
        kl_accum = 0.0

        for b in range(bsz):
            pts = cc[cb == b]

            with autocast(enabled=args.fp16):
                recon, mu, logvar = model(pts)
                loss, cd, kl = vae_reconstruction_chamfer(
                    recon, pts, mu, logvar, beta_kl=args.beta_kl,
                )
            loss_accum += loss.item()
            cd_accum += cd.item()
            kl_accum += kl.item()

        total_loss += loss_accum / bsz
        total_cd += cd_accum / bsz
        total_kl += kl_accum / bsz
        n += 1

    avg_loss = total_loss / max(n, 1)
    avg_cd = total_cd / max(n, 1)
    avg_kl = total_kl / max(n, 1)
    return avg_loss, avg_cd, avg_kl


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = setup_logger(args.output_dir)
    logger.info(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    ds_train = SemanticKITTI(
        root=args.data_path,
        split="train",
        use_ground_truth_maps=True,
        augmentation=True,
        max_points=args.max_points,
    )
    ds_val = SemanticKITTI(
        root=args.data_path,
        split="val",
        use_ground_truth_maps=True,
        augmentation=False,
        max_points=args.max_points,
    )

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- Model ----
    model = PointCloudVAEv2(
        latent_dim=args.latent_dim,
        num_decoded_points=args.num_decoded_points,
        decoder_hidden=args.decoder_hidden,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PointCloudVAEv2  trainable params: {n_params:,}")

    # ---- Optimizer ----
    opt = optim.AdamW(model.parameters(), lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.fp16)

    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        ck = load_checkpoint(args.resume, model, opt, sch)
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, best val {best_val:.4f}")

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()
        tr_loss, tr_cd, tr_kl = train_epoch(
            model, train_loader, opt, scaler, epoch, args, writer, device,
        )
        va_loss, va_cd, va_kl = validate(model, val_loader, args, device)
        sch.step()
        elapsed = time.time() - t0

        writer.add_scalar("val/loss", va_loss, epoch)
        writer.add_scalar("val/cd", va_cd, epoch)
        writer.add_scalar("val/kl", va_kl, epoch)
        writer.add_scalar("lr", sch.get_last_lr()[0], epoch)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"train {tr_loss:.4f} (cd {tr_cd:.4f} kl {tr_kl:.4f}) | "
            f"val {va_loss:.4f} (cd {va_cd:.4f} kl {va_kl:.4f}) | "
            f"lr {sch.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
        )

        extra = {
            "latent_dim": args.latent_dim,
            "num_decoded_points": args.num_decoded_points,
            "decoder_hidden": args.decoder_hidden,
        }

        if va_loss < best_val:
            best_val = va_loss
            path = os.path.join(args.output_dir, "best_point_vae_v2.pth")
            save_checkpoint(path, model, opt, sch, epoch,
                            best_val_loss=best_val, additional_info=extra)
            logger.info(f"  -> new best val {best_val:.4f}, saved {path}")

        if (epoch + 1) % args.save_freq == 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(path, model, opt, sch, epoch,
                            best_val_loss=best_val, additional_info=extra)

    writer.close()
    logger.info(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
