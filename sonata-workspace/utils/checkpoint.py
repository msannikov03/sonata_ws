"""
Utility functions for checkpoint management
"""

import torch
import os
from typing import Dict, Optional


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
    additional_info: Optional[Dict] = None
):
    """
    Save training checkpoint.
    
    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        additional_info: Any additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return checkpoint
