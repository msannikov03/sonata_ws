#!/usr/bin/env python3
"""
Training script for seq 08 split experiment.

Loads pre-voxelized .npz frames, splits into train (first 3000) / val (last 1071).
Supports --input_type lidar or --input_type da2.

Each .npz contains:
  lidar_coords, lidar_center, gt_coords_lidar   (LiDAR-centered)
  da2_coords, da2_center, gt_coords_da2         (DA2-centered)
"""

import os, sys, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Seq08NpzDataset(Dataset):
    """Loads pre-voxelized .npz frames for training/validation."""

    def __init__(self, npz_dir, indices, input_type="lidar",
                 max_input_pts=20000, max_gt_pts=20000):
        """
        Args:
            npz_dir:  directory containing 000000.npz ... 004070.npz
            indices:  list of integer frame indices to use
            input_type: "lidar" or "da2"
            max_input_pts: cap for input point cloud
            max_gt_pts: cap for GT target point cloud
        """
        self.npz_dir = npz_dir
        self.indices = sorted(indices)
        self.input_type = input_type
        self.max_input_pts = max_input_pts
        self.max_gt_pts = max_gt_pts

        # Build file list and verify existence
        self.files = []
        for idx in self.indices:
            path = os.path.join(npz_dir, f"{idx:06d}.npz")
            if os.path.exists(path):
                self.files.append(path)
        print(f"  Seq08NpzDataset: {len(self.files)} frames ({input_type})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data = np.load(self.files[i])

        if self.input_type == "lidar":
            input_coords = data["lidar_coords"].astype(np.float32)
            gt_coords = data["gt_coords_lidar"].astype(np.float32)
        else:
            input_coords = data["da2_coords"].astype(np.float32)
            gt_coords = data["gt_coords_da2"].astype(np.float32)

        # Subsample if needed
        if input_coords.shape[0] > self.max_input_pts:
            sel = np.random.choice(input_coords.shape[0], self.max_input_pts, replace=False)
            input_coords = input_coords[sel]
        if gt_coords.shape[0] > self.max_gt_pts:
            sel = np.random.choice(gt_coords.shape[0], self.max_gt_pts, replace=False)
            gt_coords = gt_coords[sel]

        # Color from z-height (same as prepare_scan in evaluate.py)
        z = input_coords[:, 2]
        zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
        colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1)

        return {
            "partial_coord": torch.from_numpy(input_coords),
            "partial_color": torch.from_numpy(colors.astype(np.float32)),
            "partial_normal": torch.zeros(input_coords.shape[0], 3),
            "complete_coord": torch.from_numpy(gt_coords),
        }


def collate_fn(batch):
    """Sparse collation: concatenate variable-length point clouds with batch indices."""
    partial_coords, partial_colors, partial_normals = [], [], []
    complete_coords = []
    batch_partial, batch_complete = [], []

    for i, sample in enumerate(batch):
        pc = sample["partial_coord"]
        partial_coords.append(pc)
        partial_colors.append(sample["partial_color"])
        partial_normals.append(sample["partial_normal"])
        batch_partial.append(torch.full((pc.shape[0],), i, dtype=torch.long))

        cc = sample["complete_coord"]
        complete_coords.append(cc)
        batch_complete.append(torch.full((cc.shape[0],), i, dtype=torch.long))

    return {
        "partial_coord": torch.cat(partial_coords, 0),
        "partial_color": torch.cat(partial_colors, 0),
        "partial_normal": torch.cat(partial_normals, 0),
        "partial_batch": torch.cat(batch_partial, 0),
        "complete_coord": torch.cat(complete_coords, 0),
        "complete_batch": torch.cat(batch_complete, 0),
    }


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model():
    """Build the full Sonata + diffusion model (same arch as train_diffusion.py)."""
    encoder = SonataEncoder(
        pretrained="facebook/sonata",
        freeze=True,
        enable_flash=False,
        feature_levels=[0],
    )
    cond = ConditionalFeatureExtractor(
        encoder,
        feature_levels=[0],
        fusion_type="concat",
    )
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=cond,
        num_timesteps=1000,
        schedule="cosine",
        denoising_steps=50,
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total/1e6:.1f}M params, {trainable/1e6:.1f}M trainable")
    return model


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, epoch, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Train epoch {epoch}")

    for batch in pbar:
        # Move to GPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()

        partial_scan = {
            "coord": batch["partial_coord"],
            "color": batch["partial_color"],
            "normal": batch["partial_normal"],
            "batch": batch["partial_batch"],
        }

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(
                partial_scan,
                batch["complete_coord"],
                batch["complete_batch"],
                return_loss=True,
            )
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Val epoch {epoch}")

    for batch in pbar:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()

        partial_scan = {
            "coord": batch["partial_coord"],
            "color": batch["partial_color"],
            "normal": batch["partial_normal"],
            "batch": batch["partial_batch"],
        }

        with torch.cuda.amp.autocast(enabled=True):
            out = model(
                partial_scan,
                batch["complete_coord"],
                batch["complete_batch"],
                return_loss=True,
            )
            loss = out["loss"]

        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Seq 08 split training")
    p.add_argument("--npz_dir", type=str, required=True,
                   help="Directory with pre-voxelized .npz frames")
    p.add_argument("--input_type", type=str, default="lidar",
                   choices=["lidar", "da2"],
                   help="Which input modality to train on")
    p.add_argument("--output_dir", type=str, default="checkpoints/seq08_split",
                   help="Directory for checkpoints and logs")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_input_pts", type=int, default=20000)
    p.add_argument("--max_gt_pts", type=int, default=20000)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.join(args.output_dir, args.input_type)
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Frame split: first 3000 train, last 1071 val
    # Seq 08 has 4071 frames (000000 - 004070)
    train_indices = list(range(0, 3000))
    val_indices = list(range(3000, 4071))

    print(f"\n=== Seq 08 Split Training ({args.input_type}) ===")
    print(f"Train: {len(train_indices)} frames, Val: {len(val_indices)} frames")
    print(f"Output: {out_dir}\n")

    # Datasets
    train_ds = Seq08NpzDataset(
        args.npz_dir, train_indices, args.input_type,
        args.max_input_pts, args.max_gt_pts,
    )
    val_ds = Seq08NpzDataset(
        args.npz_dir, val_indices, args.input_type,
        args.max_input_pts, args.max_gt_pts,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = build_model().cuda()

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Resume
    start_epoch = 0
    best_val = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    # Training loop
    history = []
    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, args.grad_clip,
        )
        val_loss = validate(model, val_loader, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={lr_now:.2e}  time={elapsed:.0f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": lr_now, "time_s": elapsed,
        })

        # Save checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "input_type": args.input_type,
        }
        torch.save(ckpt, os.path.join(out_dir, f"epoch_{epoch}.pth"))

        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val_loss"] = best_val
            torch.save(ckpt, os.path.join(out_dir, "best.pth"))
            print(f"  -> new best val loss: {best_val:.5f}")

    # Save history
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.5f}")
    print(f"Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()
