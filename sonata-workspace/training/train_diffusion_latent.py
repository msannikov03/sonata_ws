"""
Train latent diffusion (DDPM in VAE latent) with Sonata-pooled conditioning.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.semantickitti import SemanticKITTI, collate_fn
from models.latent_diffusion import SceneCompletionLatentDiffusion
from models.point_cloud_vae import PointCloudVAE
from models.sonata_encoder import ConditionalFeatureExtractor, SonataEncoder
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Train latent diffusion")
    p.add_argument(
        "--data_path",
        type=str,
        default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"),
    )
    p.add_argument(
        "--vae_ckpt",
        type=str,
        default=None,
        help="best_point_vae.pth (not needed if --resume loads full model)",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument(
        "--schedule", type=str, default="cosine",
        choices=["linear", "cosine", "sigmoid"],
    )
    p.add_argument("--denoising_steps", type=int, default=50)
    p.add_argument("--encoder_ckpt", type=str, default="facebook/sonata")
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--enable_flash", action="store_true")
    p.add_argument(
        "--voxel_size_sonata",
        type=float,
        default=0.05,
        help="Grid for Sonata: grid_coord = floor(coord / this)",
    )
    p.add_argument("--point_max_partial", type=int, default=20000)
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--output_dir", type=str, default="checkpoints/latent_diffusion")
    p.add_argument("--log_dir", type=str, default="logs/latent_diffusion")
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--eval_freq", type=int, default=1)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--train_vae",
        action="store_true",
        help="Unfreeze VAE and train jointly with small diffusion loss on mu",
    )
    p.add_argument(
        "--use_posterior_sample",
        action="store_true",
        help="If training VAE jointly, sample z ~ q(z|x) as z0 (noisy targets)",
    )
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML with keys matching argparse dest names",
    )
    args = p.parse_args()
    if args.config is not None:
        cfg_path = os.path.expanduser(args.config)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            for key, value in cfg.items():
                if isinstance(value, str) and value.startswith("~"):
                    value = os.path.expanduser(value)
                setattr(args, key, value)
    if args.vae_ckpt is None and args.resume is None:
        raise SystemExit(
            "Either --vae_ckpt (for new run) or --resume (full latent ckpt) is required."
        )
    return args


def build_partial_dict(batch, voxel_size_sonata: float) -> dict:
    coord = batch["partial_coord"]
    gc = torch.floor(coord / voxel_size_sonata).long()
    # Shift per batch sample so grid_coord is non-negative
    # (required by Sonata serialization / hashing)
    batch_idx = batch["partial_batch"]
    for b in batch_idx.unique():
        mask = batch_idx == b
        gc[mask] -= gc[mask].min(dim=0)[0]
    return {
        "coord": coord,
        "color": batch["partial_color"],
        "normal": batch["partial_normal"],
        "grid_coord": gc,
        "batch": batch_idx,
    }


def _infer_vae_dims_from_state_dict(sd: dict) -> tuple[int, int]:
    w = sd["vae.fc_mu.weight"]
    latent_dim = w.shape[0]
    k = sd["vae.decoder_out.weight"].shape[0] // 3
    return latent_dim, k


def load_vae_from_checkpoint(path: str, device: torch.device) -> PointCloudVAE:
    ck = torch.load(os.path.expanduser(path), map_location="cpu")
    latent_dim = ck.get("latent_dim", 256)
    k = ck.get("num_decoded_points", 2048)
    vae = PointCloudVAE(latent_dim=latent_dim, num_decoded_points=k)
    vae.load_state_dict(ck["model_state_dict"])
    return vae.to(device)


def train_epoch(model, loader, opt, epoch, args, writer, scaler, device, use_amp: bool):
    model.train()
    if not args.train_vae:
        model.vae.eval()
    tot = 0.0
    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        partial = build_partial_dict(batch, args.voxel_size_sonata)
        cc = batch["complete_coord"]
        cb = batch["complete_batch"]

        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model.forward_training(
                partial,
                cc,
                cb,
                freeze_vae=not args.train_vae,
                use_posterior_sample=args.use_posterior_sample,
            )
            loss = out["loss"]

        if use_amp:
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            opt.step()

        lv = loss.item()
        tot += lv
        writer.add_scalar("train/loss", lv, epoch * len(loader) + i)
    return tot / len(loader)


@torch.no_grad()
def validate(model, loader, args, writer, epoch, device, use_amp: bool):
    model.eval()
    tot = 0.0
    for batch in tqdm(loader, desc="Val"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        partial = build_partial_dict(batch, args.voxel_size_sonata)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model.forward_training(
                partial,
                batch["complete_coord"],
                batch["complete_batch"],
                freeze_vae=True,
                use_posterior_sample=False,
            )
            tot += out["loss"].item()
    avg = tot / len(loader)
    writer.add_scalar("val/loss", avg, epoch)
    return avg


def collect_trainable_params(model: SceneCompletionLatentDiffusion, train_vae: bool):
    params = (
        list(model.denoiser.parameters())
        + list(model.condition_extractor.parameters())
    )
    if train_vae:
        params += list(model.vae.parameters())
    return params


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = SonataEncoder(
        pretrained=args.encoder_ckpt,
        freeze=args.freeze_encoder,
        enable_flash=args.enable_flash,
        feature_levels=[0],
    )
    cond_ext = ConditionalFeatureExtractor(
        encoder, feature_levels=[0], fusion_type="concat"
    )

    start = 0
    best = float("inf")
    if args.resume:
        ck0 = torch.load(os.path.expanduser(args.resume), map_location="cpu")
        sd0 = ck0["model_state_dict"]
        ld, nk = _infer_vae_dims_from_state_dict(sd0)
        num_t = ck0.get("num_timesteps", args.num_timesteps)
        sched = ck0.get("schedule", args.schedule)
        vae = PointCloudVAE(latent_dim=ld, num_decoded_points=nk).to(device)
        model = SceneCompletionLatentDiffusion(
            vae=vae,
            condition_extractor=cond_ext,
            num_timesteps=num_t,
            schedule=sched,
            denoising_steps=args.denoising_steps,
        ).to(device)
        opt = optim.AdamW(
            collect_trainable_params(model, args.train_vae),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.num_epochs, eta_min=1e-6
        )
        ck = load_checkpoint(os.path.expanduser(args.resume), model, opt, sch)
        start = ck.get("epoch", 0) + 1
        best = ck.get("best_val_loss", float("inf"))
    else:
        vae = load_vae_from_checkpoint(os.path.expanduser(args.vae_ckpt), device)
        if not args.train_vae:
            for p in vae.parameters():
                p.requires_grad = False
        model = SceneCompletionLatentDiffusion(
            vae=vae,
            condition_extractor=cond_ext,
            num_timesteps=args.num_timesteps,
            schedule=args.schedule,
            denoising_steps=args.denoising_steps,
        ).to(device)
        opt = optim.AdamW(
            collect_trainable_params(model, args.train_vae),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.num_epochs, eta_min=1e-6
        )

    ds_tr = SemanticKITTI(
        root=args.data_path,
        split="train",
        use_ground_truth_maps=True,
        augmentation=True,
        use_point_cloud=True,
        point_max_partial=args.point_max_partial,
        point_max_complete=args.point_max_complete,
    )
    ds_va = SemanticKITTI(
        root=args.data_path,
        split="val",
        use_ground_truth_maps=True,
        augmentation=False,
        use_point_cloud=True,
        point_max_partial=args.point_max_partial,
        point_max_complete=args.point_max_complete,
    )
    tr_loader = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    va_loader = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    writer = SummaryWriter(args.log_dir)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(start, args.num_epochs):
        tr = train_epoch(model, tr_loader, opt, epoch, args, writer, scaler, device, use_amp)
        logger.info(f"Epoch {epoch} train {tr:.6f}")
        if (epoch + 1) % args.eval_freq == 0:
            va = validate(model, va_loader, args, writer, epoch, device, use_amp)
            logger.info(f"Val {va:.6f}")
            if va < best:
                best = va
                path = os.path.join(args.output_dir, "best_latent_diffusion.pth")
                save_checkpoint(
                    path,
                    model,
                    opt,
                    sch,
                    epoch,
                    best_val_loss=best,
                    additional_info={
                        "num_timesteps": model.scheduler.num_timesteps,
                        "schedule": model.scheduler.schedule,
                    },
                )
                logger.info(f"Saved {path}")
        if (epoch + 1) % args.save_freq == 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(
                path,
                model,
                opt,
                sch,
                epoch,
                best_val_loss=best,
                additional_info={
                    "num_timesteps": model.scheduler.num_timesteps,
                    "schedule": model.scheduler.schedule,
                },
            )
        sch.step()

    writer.close()


if __name__ == "__main__":
    main()
