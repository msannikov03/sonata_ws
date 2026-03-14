#!/usr/bin/env python3
"""
Cross-Modal Distillation: LiDAR Teacher → RGB Student
for Diffusion-Based 3D Scene Completion

Implements three distillation strategies:
  1. Output-level: Match teacher's noise predictions (MSE)
  2. Feature-level: Match intermediate denoiser features (cosine similarity)
  3. Structural: Scene-wise geometric + point-wise landmark loss (ScoreLiDAR-inspired)

Teacher: Trained on LiDAR/DepthPro point clouds (frozen)
Student: Trained on DA2 pseudo point clouds (learning)

Usage:
    python train_distillation.py \
        --teacher_ckpt /path/to/best_model.pth \
        --data_path_student /path/to/depth_anything_v2_dataset \
        --data_path_teacher /path/to/lidar_dataset \
        --strategy all \
        --output_dir checkpoints/distillation
"""

import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add sonata workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Distillation Losses
# ============================================================================

class OutputDistillationLoss(nn.Module):
    """
    Strategy 1: Match teacher's noise predictions.
    L = MSE(student_noise_pred, teacher_noise_pred)
    Simplest form — student learns to denoise the same way as teacher.
    """
    def forward(self, student_pred, teacher_pred):
        return F.mse_loss(student_pred, teacher_pred)


class FeatureDistillationLoss(nn.Module):
    """
    Strategy 2: Match intermediate features using cosine similarity.
    Cosine > MSE for cross-modal because it's scale-invariant.
    Operates on bottleneck + skip connection features.
    """
    def __init__(self, student_dims, teacher_dims):
        super().__init__()
        # Projection layers to align dimensions if they differ
        self.projections = nn.ModuleDict()
        for name in student_dims:
            if student_dims[name] != teacher_dims[name]:
                self.projections[name] = nn.Linear(
                    student_dims[name], teacher_dims[name]
                )

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: dict of {name: (N, D)} tensors
            teacher_features: dict of {name: (N, D)} tensors
        Returns:
            loss: scalar
        """
        total_loss = 0.0
        n_terms = 0
        for name in student_features:
            if name not in teacher_features:
                continue
            s_feat = student_features[name]
            t_feat = teacher_features[name]

            # Project if dimensions differ
            if name in self.projections:
                s_feat = self.projections[name](s_feat)

            # Handle different point counts (from downsampling)
            if s_feat.shape[0] != t_feat.shape[0]:
                min_n = min(s_feat.shape[0], t_feat.shape[0])
                s_feat = s_feat[:min_n]
                t_feat = t_feat[:min_n]

            # Cosine similarity loss (1 - cosine_sim)
            cos_sim = F.cosine_similarity(s_feat, t_feat, dim=-1)
            loss = (1 - cos_sim).mean()
            total_loss += loss
            n_terms += 1

        return total_loss / max(n_terms, 1)


class StructuralDistillationLoss(nn.Module):
    """
    Strategy 3: ScoreLiDAR-inspired structural loss.
    Combines:
      (a) Scene-wise: Chamfer distance between student and teacher
          completed scenes (geometric consistency)
      (b) Point-wise: Landmark points matching — forces key structural
          points to be preserved
    """
    def __init__(self, n_landmarks=256):
        super().__init__()
        self.n_landmarks = n_landmarks

    def chamfer_distance(self, x, y, subsample=2048):
        """Approximate Chamfer distance with subsampling for efficiency."""
        if x.shape[0] > subsample:
            idx = torch.randperm(x.shape[0], device=x.device)[:subsample]
            x = x[idx]
        if y.shape[0] > subsample:
            idx = torch.randperm(y.shape[0], device=y.device)[:subsample]
            y = y[idx]

        # x->y
        dist_xy = torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
        min_xy = dist_xy.min(dim=1)[0].mean()
        # y->x
        min_yx = dist_xy.min(dim=0)[0].mean()
        return (min_xy + min_yx) / 2

    def landmark_loss(self, student_pred, teacher_pred, coords):
        """Select landmark points (farthest point sampling) and match."""
        n = coords.shape[0]
        k = min(self.n_landmarks, n)

        # Simple uniform landmark selection
        indices = torch.linspace(0, n - 1, k, dtype=torch.long, device=coords.device)
        s_landmarks = student_pred[indices]
        t_landmarks = teacher_pred[indices]

        return F.mse_loss(s_landmarks, t_landmarks)

    def forward(self, student_pred, teacher_pred, coords):
        """
        Args:
            student_pred: (N, 3) student's denoised output
            teacher_pred: (N, 3) teacher's denoised output
            coords: (N, 3) point coordinates
        """
        scene_loss = self.chamfer_distance(student_pred, teacher_pred)
        point_loss = self.landmark_loss(student_pred, teacher_pred, coords)
        return scene_loss + point_loss


# ============================================================================
# Modified DenoisingNetwork that returns intermediate features
# ============================================================================

def forward_with_intermediates(denoiser, features, coords, timestep, condition):
    """
    Run denoiser forward pass and capture intermediate features
    for feature-level distillation.

    Returns:
        noise_pred: (N, 3)
        intermediates: dict of feature tensors at various levels
    """
    intermediates = {}

    # Time embedding
    t_embed = denoiser.time_embedding(timestep)

    # Input projection
    x = denoiser.input_proj(features)
    x_coords = coords
    x_cond = condition['features']

    # Encoder path
    skip_features = []
    skip_coords = []
    skip_conds = []

    for i, enc_block in enumerate(denoiser.encoder_blocks):
        x = enc_block(x, x_coords, t_embed, x_cond)
        skip_features.append(x)
        skip_coords.append(x_coords)
        skip_conds.append(x_cond)

        intermediates[f'enc_{i}'] = x.detach() if not x.requires_grad else x

        target_num = x.shape[0] // 2
        N = x.shape[0]
        if N > target_num:
            indices = torch.linspace(0, N - 1, target_num, dtype=torch.long, device=x.device)
            x = x[indices]
            x_coords = x_coords[indices]
            x_cond = x_cond[indices]
        x = denoiser.level_up[i](x)

    # Bottleneck
    x = denoiser.bottleneck(x, x_coords, t_embed, x_cond)
    intermediates['bottleneck'] = x.detach() if not x.requires_grad else x

    # Decoder path
    for i, dec_block in enumerate(denoiser.decoder_blocks):
        skip_feat = skip_features[-(i+1)]
        skip_coord = skip_coords[-(i+1)]
        skip_cond = skip_conds[-(i+1)]

        x = denoiser._upsample_points(x, x_coords, skip_coord)
        x_coords = skip_coord
        x_cond = skip_cond

        x = torch.cat([x, skip_feat], dim=-1)
        x = dec_block(x, x_coords, t_embed, x_cond)
        intermediates[f'dec_{i}'] = x.detach() if not x.requires_grad else x

    noise_pred = denoiser.output_proj(x)
    return noise_pred, intermediates


# ============================================================================
# Distillation Trainer
# ============================================================================

class DistillationTrainer:
    """
    Manages the teacher-student distillation training loop.
    """
    def __init__(self, teacher_model, student_model, strategy='all',
                 alpha_task=1.0, alpha_output=1.0, alpha_feature=0.5,
                 alpha_structural=0.1, device='cuda'):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.strategy = strategy
        self.device = device

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Loss functions
        self.task_loss_fn = nn.MSELoss()  # Standard diffusion loss
        self.output_loss_fn = OutputDistillationLoss()
        self.feature_loss_fn = FeatureDistillationLoss(
            student_dims={'bottleneck': 512, 'enc_0': 64, 'enc_1': 128, 'enc_2': 256},
            teacher_dims={'bottleneck': 512, 'enc_0': 64, 'enc_1': 128, 'enc_2': 256},
        ).to(device)
        self.structural_loss_fn = StructuralDistillationLoss().to(device)

        # Loss weights
        self.alpha_task = alpha_task
        self.alpha_output = alpha_output
        self.alpha_feature = alpha_feature
        self.alpha_structural = alpha_structural

    def compute_losses(self, student_output, teacher_output,
                       student_intermediates, teacher_intermediates,
                       noise, coords):
        """Compute all distillation losses based on strategy."""
        losses = {}

        # Task loss (always present): MSE on noise prediction vs ground truth
        losses['task'] = self.alpha_task * self.task_loss_fn(
            student_output, noise
        )

        if self.strategy in ('output', 'all'):
            losses['output_distill'] = self.alpha_output * self.output_loss_fn(
                student_output, teacher_output
            )

        if self.strategy in ('feature', 'all'):
            losses['feature_distill'] = self.alpha_feature * self.feature_loss_fn(
                student_intermediates, teacher_intermediates
            )

        if self.strategy in ('structural', 'all'):
            # For structural loss, compute denoised coordinates
            # x_0 ≈ (noisy - sqrt(1-alpha)*pred_noise) / sqrt(alpha)
            # We use the noise predictions as proxy for structural comparison
            losses['structural_distill'] = self.alpha_structural * self.structural_loss_fn(
                student_output, teacher_output, coords
            )

        return losses


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(trainer, student_model, teacher_model,
                student_loader, teacher_loader,
                optimizer, scaler, epoch, args, writer):
    """Train for one epoch with distillation."""
    student_model.train()
    total_losses = {}

    student_iter = iter(student_loader)
    teacher_iter = iter(teacher_loader)

    n_steps = min(len(student_loader), len(teacher_loader))
    pbar = tqdm(range(n_steps), desc=f"Epoch {epoch}")

    for step in pbar:
        try:
            student_batch = next(student_iter)
            teacher_batch = next(teacher_iter)
        except StopIteration:
            break

        # Move to GPU
        for key in student_batch:
            if isinstance(student_batch[key], torch.Tensor):
                student_batch[key] = student_batch[key].cuda()
        for key in teacher_batch:
            if isinstance(teacher_batch[key], torch.Tensor):
                teacher_batch[key] = teacher_batch[key].cuda()

        # Prepare student input (DA2 pseudo point clouds)
        student_scan = {
            'coord': student_batch['partial_coord'],
            'color': student_batch['partial_color'],
            'normal': student_batch['partial_normal'],
            'batch': student_batch['partial_batch'],
        }

        # Prepare teacher input (LiDAR point clouds)
        teacher_scan = {
            'coord': teacher_batch['partial_coord'],
            'color': teacher_batch['partial_color'],
            'normal': teacher_batch['partial_normal'],
            'batch': teacher_batch['partial_batch'],
        }

        complete_coord = student_batch['complete_coord']
        complete_batch = student_batch.get('complete_batch')

        with torch.cuda.amp.autocast(enabled=args.fp16):
            # ---- Teacher forward (frozen, no grad) ----
            with torch.no_grad():
                # Get teacher condition features
                teacher_cond, _ = teacher_model.condition_extractor(teacher_scan)
                from models.diffusion_module import knn_interpolate
                teacher_cond_mapped = knn_interpolate(
                    teacher_cond, teacher_scan['coord'], complete_coord
                )

                # Sample timestep and noise (shared between teacher and student)
                if complete_batch is not None:
                    batch_size = complete_batch.max().item() + 1
                else:
                    batch_size = 1
                t = torch.randint(0, teacher_model.scheduler.num_timesteps,
                                  (batch_size,), device=complete_coord.device)
                noise = torch.randn_like(complete_coord)

                if complete_batch is not None:
                    t_per_point = t[complete_batch]
                else:
                    t_per_point = t.expand(complete_coord.shape[0])

                dev = complete_coord.device
                sa = teacher_model.scheduler.sqrt_alphas_cumprod.to(dev)[t_per_point].unsqueeze(-1)
                som = teacher_model.scheduler.sqrt_one_minus_alphas_cumprod.to(dev)[t_per_point].unsqueeze(-1)
                noisy_scan = sa * complete_coord + som * noise

                # Teacher noise prediction with intermediates
                teacher_pred, teacher_intermediates = forward_with_intermediates(
                    teacher_model.denoiser, noisy_scan, complete_coord, t,
                    {'features': teacher_cond_mapped}
                )

            # ---- Student forward ----
            # Get student condition features
            student_cond, _ = student_model.condition_extractor(student_scan)
            student_cond_mapped = knn_interpolate(
                student_cond, student_scan['coord'], complete_coord
            )

            # Student noise prediction with intermediates
            student_pred, student_intermediates = forward_with_intermediates(
                student_model.denoiser, noisy_scan, complete_coord, t,
                {'features': student_cond_mapped}
            )

            # ---- Compute losses ----
            losses = trainer.compute_losses(
                student_pred, teacher_pred,
                student_intermediates, teacher_intermediates,
                noise, complete_coord
            )
            total_loss = sum(losses.values())

        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.gradient_clip)
            optimizer.step()

        # Logging
        for k, v in losses.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            total_losses[k] = total_losses.get(k, 0) + val

        pbar.set_postfix({k: f"{v.item():.4f}" if isinstance(v, torch.Tensor) else f"{v:.4f}"
                          for k, v in losses.items()})

        global_step = epoch * n_steps + step
        for k, v in losses.items():
            writer.add_scalar(f'train/{k}', v.item() if isinstance(v, torch.Tensor) else v, global_step)

    avg_losses = {k: v / n_steps for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(student_model, val_loader, epoch, args, writer):
    """Validate student model on task loss only."""
    student_model.eval()
    total_loss = 0.0

    for batch in tqdm(val_loader, desc="Validation"):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        partial_scan = {
            'coord': batch['partial_coord'],
            'color': batch['partial_color'],
            'normal': batch['partial_normal'],
            'batch': batch['partial_batch'],
        }
        complete_coord = batch['complete_coord']
        complete_batch = batch.get('complete_batch')

        with torch.cuda.amp.autocast(enabled=args.fp16):
            output = student_model(partial_scan, complete_coord, complete_batch,
                                   return_loss=True)
            loss = output['loss']

        total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

    avg_loss = total_loss / len(val_loader)
    writer.add_scalar('val/loss', avg_loss, epoch)
    return avg_loss


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Modal Distillation Training')

    # Paths
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                        help='Path to trained teacher model checkpoint')
    parser.add_argument('--data_path_student', type=str, required=True,
                        help='Path to DA2 pseudo point cloud dataset')
    parser.add_argument('--data_path_teacher', type=str, required=True,
                        help='Path to LiDAR dataset (for teacher forward pass)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/distillation')
    parser.add_argument('--log_dir', type=str, default='logs/distillation')

    # Model
    parser.add_argument('--encoder_ckpt', type=str, default='facebook/sonata')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--fp16', action='store_true')

    # Distillation
    parser.add_argument('--strategy', type=str, default='all',
                        choices=['output', 'feature', 'structural', 'all'],
                        help='Distillation strategy')
    parser.add_argument('--alpha_task', type=float, default=1.0,
                        help='Weight for task loss (noise prediction MSE)')
    parser.add_argument('--alpha_output', type=float, default=1.0,
                        help='Weight for output distillation loss')
    parser.add_argument('--alpha_feature', type=float, default=0.5,
                        help='Weight for feature distillation loss')
    parser.add_argument('--alpha_structural', type=float, default=0.1,
                        help='Weight for structural distillation loss')

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--voxel_size', type=float, default=0.05)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    writer = SummaryWriter(args.log_dir)

    print(f"\n{'='*60}")
    print(f"Cross-Modal Distillation Training")
    print(f"Strategy: {args.strategy}")
    print(f"Teacher checkpoint: {args.teacher_ckpt}")
    print(f"Student data: {args.data_path_student}")
    print(f"Teacher data: {args.data_path_teacher}")
    print(f"{'='*60}\n")

    # ---- Build models ----
    from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
    from models.diffusion_module import SceneCompletionDiffusion
    from data.semantickitti import SemanticKITTI, collate_fn
    from utils.checkpoint import save_checkpoint, load_checkpoint

    # Build teacher model and load checkpoint
    print("Building teacher model...")
    teacher_encoder = SonataEncoder(
        pretrained=args.encoder_ckpt,
        freeze=True,  # Always freeze teacher encoder
        feature_levels=[0]
    )
    teacher_condition = ConditionalFeatureExtractor(
        teacher_encoder, feature_levels=[0], fusion_type="concat"
    )
    teacher_model = SceneCompletionDiffusion(
        encoder=teacher_encoder,
        condition_extractor=teacher_condition,
        num_timesteps=1000,
        schedule="cosine",
    )

    print(f"Loading teacher checkpoint: {args.teacher_ckpt}")
    ckpt = load_checkpoint(args.teacher_ckpt)
    teacher_model.load_state_dict(ckpt['model_state_dict'])
    teacher_model.cuda().eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print("Teacher loaded and frozen.")

    # Build student model (fresh weights for denoiser, same encoder)
    print("\nBuilding student model...")
    student_encoder = SonataEncoder(
        pretrained=args.encoder_ckpt,
        freeze=args.freeze_encoder,
        feature_levels=[0]
    )
    student_condition = ConditionalFeatureExtractor(
        student_encoder, feature_levels=[0], fusion_type="concat"
    )
    student_model = SceneCompletionDiffusion(
        encoder=student_encoder,
        condition_extractor=student_condition,
        num_timesteps=1000,
        schedule="cosine",
    )
    student_model.cuda()

    trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student_model.parameters())
    print(f"Student: {total:,} total params, {trainable:,} trainable")

    # ---- Datasets ----
    print("\nLoading datasets...")
    student_train = SemanticKITTI(
        root=args.data_path_student, split='train',
        voxel_size=args.voxel_size, use_ground_truth_maps=True, augmentation=True,
    )
    teacher_train = SemanticKITTI(
        root=args.data_path_teacher, split='train',
        voxel_size=args.voxel_size, use_ground_truth_maps=True, augmentation=True,
    )
    val_dataset = SemanticKITTI(
        root=args.data_path_student, split='val',
        voxel_size=args.voxel_size, use_ground_truth_maps=True, augmentation=False,
    )

    student_loader = DataLoader(
        student_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    teacher_loader = DataLoader(
        teacher_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"Student train: {len(student_train)} samples")
    print(f"Teacher train: {len(teacher_train)} samples")
    print(f"Validation: {len(val_dataset)} samples")

    # ---- Trainer ----
    trainer = DistillationTrainer(
        teacher_model, student_model,
        strategy=args.strategy,
        alpha_task=args.alpha_task,
        alpha_output=args.alpha_output,
        alpha_feature=args.alpha_feature,
        alpha_structural=args.alpha_structural,
    )

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # ---- Training loop ----
    best_val_loss = float('inf')
    print(f"\nStarting distillation training ({args.strategy})...\n")

    for epoch in range(args.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*50}")

        avg_losses = train_epoch(
            trainer, student_model, teacher_model,
            student_loader, teacher_loader,
            optimizer, scaler, epoch, args, writer,
        )

        for k, v in avg_losses.items():
            print(f"  {k}: {v:.6f}")

        # Validate
        if (epoch + 1) % args.eval_freq == 0:
            val_loss = validate(student_model, val_loader, epoch, args, writer)
            print(f"  val_loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(args.output_dir, 'best_model.pth'),
                    student_model, optimizer, scheduler, epoch, best_val_loss,
                )
                print(f"  Saved best model (val_loss={best_val_loss:.6f})")

        # Periodic save
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'),
                student_model, optimizer, scheduler, epoch, best_val_loss,
            )

        scheduler.step()
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

    # Final save
    save_checkpoint(
        os.path.join(args.output_dir, 'final_model.pth'),
        student_model, optimizer, scheduler, args.num_epochs - 1, best_val_loss,
    )
    print(f"\nDistillation complete! Best val loss: {best_val_loss:.6f}")
    writer.close()


if __name__ == "__main__":
    main()
