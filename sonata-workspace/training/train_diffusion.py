"""
Training Script for Sonata-LiDiff Semantic Scene Completion

Main training loop for the diffusion model with Sonata encoder.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from tqdm import tqdm
from typing import Dict
import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.semantickitti import SemanticKITTI, collate_fn
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Sonata-LiDiff for semantic scene completion'
    )
    
    # Data
    parser.add_argument(
        '--data_path', type=str, 
        default='Datasets/SemanticKITTI/dataset',
        help='Path to SemanticKITTI dataset'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--voxel_size', type=float, default=0.05,
        help='Voxel size for scene representation'
    )
    
    # Model
    parser.add_argument(
        '--encoder_ckpt', type=str, default='facebook/sonata',
        help='Sonata encoder checkpoint'
    )
    parser.add_argument(
        '--freeze_encoder', action='store_true',
        help='Freeze Sonata encoder weights'
    )
    parser.add_argument(
        '--enable_flash', action='store_true',
        help='Enable flash attention in encoder'
    )
    parser.add_argument(
        '--num_timesteps', type=int, default=1000,
        help='Number of diffusion timesteps'
    )
    parser.add_argument(
        '--schedule', type=str, default='cosine',
        choices=['linear', 'cosine', 'sigmoid'],
        help='Noise schedule type'
    )
    
    # Training
    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=10,
        help='Number of warmup epochs'
    )
    parser.add_argument(
        '--gradient_clip', type=float, default=1.0,
        help='Gradient clipping threshold'
    )
    parser.add_argument(
        '--accumulation_steps', type=int, default=1,
        help='Gradient accumulation steps'
    )
    
    # Output
    parser.add_argument(
        '--output_dir', type=str, default='checkpoints/diffusion',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs/diffusion',
        help='TensorBoard log directory'
    )
    parser.add_argument(
        '--save_freq', type=int, default=5,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--eval_freq', type=int, default=1,
        help='Evaluate every N epochs'
    )
    
    # Resume
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from checkpoint'
    )
    
    # Config file
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config
        for key, value in config.items():
            setattr(args, key, value)
    
    return args


def build_model(args) -> SceneCompletionDiffusion:
    """Build the complete scene completion model."""
    
    # Sonata encoder
    print("Loading Sonata encoder...")
    encoder = SonataEncoder(
        pretrained=args.encoder_ckpt,
        freeze=args.freeze_encoder,
        enable_flash=args.enable_flash,
        feature_levels=[2, 3, 4]
    )
    
    # Conditional feature extractor
    print("Building conditional feature extractor...")
    condition_extractor = ConditionalFeatureExtractor(
        encoder,
        feature_levels=[2, 3, 4],
        fusion_type="attention"
    )
    
    # Complete diffusion model
    print("Building diffusion model...")
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=condition_extractor,
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        denoising_steps=50
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def train_epoch(
    model: SceneCompletionDiffusion,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    args,
    writer: SummaryWriter
) -> float:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for i, batch in enumerate(pbar):
        # Move to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        
        # Prepare input
        partial_scan = {
            'coord': batch['partial_coord'],
            'color': batch['partial_color'],
            'normal': batch['partial_normal'],
            'batch': batch['partial_batch'],
        }
        
        complete_coord = batch['complete_coord']
        
        # Forward pass
        output = model(partial_scan, complete_coord, return_loss=True)
        loss = output['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / args.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (i + 1) % args.accumulation_steps == 0:
            # Gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item() * args.accumulation_steps
        pbar.set_postfix({'loss': loss.item() * args.accumulation_steps})
        
        # TensorBoard logging
        step = epoch * len(dataloader) + i
        writer.add_scalar('train/loss', loss.item(), step)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(
    model: SceneCompletionDiffusion,
    dataloader: DataLoader,
    epoch: int,
    args,
    writer: SummaryWriter
) -> float:
    """Validate the model."""
    
    model.eval()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Validation")
    
    for i, batch in enumerate(pbar):
        # Move to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        
        # Prepare input
        partial_scan = {
            'coord': batch['partial_coord'],
            'color': batch['partial_color'],
            'normal': batch['partial_normal'],
            'batch': batch['partial_batch'],
        }
        
        complete_coord = batch['complete_coord']
        
        # Forward pass
        output = model(partial_scan, complete_coord, return_loss=True)
        loss = output['loss']
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    
    # TensorBoard logging
    writer.add_scalar('val/loss', avg_loss, epoch)
    
    return avg_loss


def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Arguments: {args}")
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Build datasets
    print("\nLoading datasets...")
    train_dataset = SemanticKITTI(
        root=args.data_path,
        split='train',
        voxel_size=args.voxel_size,
        use_ground_truth_maps=True,
        augmentation=True
    )
    
    val_dataset = SemanticKITTI(
        root=args.data_path,
        split='val',
        voxel_size=args.voxel_size,
        use_ground_truth_maps=True,
        augmentation=False
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Build model
    model = build_model(args).cuda()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume is not None:
        print(f"\nResuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch, args, writer
        )
        logger.info(f"Train loss: {train_loss:.6f}")
        
        # Validate
        if (epoch + 1) % args.eval_freq == 0:
            val_loss = validate(
                model, val_loader, epoch, args, writer
            )
            logger.info(f"Val loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(
                    args.output_dir, 'best_model.pth'
                )
                save_checkpoint(
                    save_path, model, optimizer, scheduler, 
                    epoch, best_val_loss
                )
                logger.info(f"Saved best model to {save_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(
                save_path, model, optimizer, scheduler, 
                epoch, best_val_loss
            )
            logger.info(f"Saved checkpoint to {save_path}")
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
    
    # Save final model
    save_path = os.path.join(args.output_dir, 'final_model.pth')
    save_checkpoint(
        save_path, model, optimizer, scheduler, 
        args.num_epochs - 1, best_val_loss
    )
    logger.info(f"\nTraining completed! Final model saved to {save_path}")
    
    writer.close()


if __name__ == "__main__":
    main()
