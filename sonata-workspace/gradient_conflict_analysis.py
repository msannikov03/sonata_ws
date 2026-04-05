#!/usr/bin/env python3
"""
Gradient Conflict Analysis for Cross-Modal Distillation

Mechanistic proof that alignment losses (output-matching, feature-alignment,
structural similarity) create gradient conflicts with the task loss (Chamfer
Distance / MSE noise prediction), explaining why task-loss-only distillation
outperforms multi-loss strategies.

For each batch, computes per-loss gradients on the student model's shared
parameters and measures pairwise cosine similarity, magnitude ratios, and
direction variance across batches.

Usage:
    python gradient_conflict_analysis.py \
      --data_path /home/anywherevla/sonata_ws/dataset/sonata_depth_pro \
      --teacher_ckpt checkpoints/diffusion_depthpro/best_model.pth \
      --student_ckpt checkpoints/distill_task_only/best_model.pth \
      --output_dir gradient_analysis_results \
      --num_batches 200
"""

import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# Add workspace to path
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE)


# ============================================================================
# Loss Functions (matching train_distillation.py exactly)
# ============================================================================

class TaskLoss(nn.Module):
    """MSE between student noise prediction and ground truth noise (diffusion task loss)."""
    def forward(self, student_pred, noise):
        return F.mse_loss(student_pred, noise)


class OutputMatchingLoss(nn.Module):
    """L2 between student and teacher noise predictions."""
    def forward(self, student_pred, teacher_pred):
        return F.mse_loss(student_pred, teacher_pred)


class FeatureAlignmentLoss(nn.Module):
    """Cosine similarity loss between student and teacher intermediate features."""
    def forward(self, student_intermediates, teacher_intermediates):
        total_loss = 0.0
        n_terms = 0
        for name in student_intermediates:
            if name not in teacher_intermediates:
                continue
            s_feat = student_intermediates[name]
            t_feat = teacher_intermediates[name].detach()
            # Handle different point counts
            if s_feat.shape[0] != t_feat.shape[0]:
                min_n = min(s_feat.shape[0], t_feat.shape[0])
                s_feat = s_feat[:min_n]
                t_feat = t_feat[:min_n]
            cos_sim = F.cosine_similarity(s_feat, t_feat, dim=-1)
            loss = (1 - cos_sim).mean()
            total_loss += loss
            n_terms += 1
        return total_loss / max(n_terms, 1)


class StructuralLoss(nn.Module):
    """
    Chamfer distance between student and teacher denoised outputs (x_0 predictions)
    plus landmark point matching.
    """
    def __init__(self, n_landmarks=256, subsample=2048):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.subsample = subsample

    def forward(self, student_x0, teacher_x0, coords):
        # Scene-wise CD
        x = student_x0
        y = teacher_x0.detach()
        if x.shape[0] > self.subsample:
            idx = torch.randperm(x.shape[0], device=x.device)[:self.subsample]
            x_sub = x[idx]
        else:
            x_sub = x
        if y.shape[0] > self.subsample:
            idx = torch.randperm(y.shape[0], device=y.device)[:self.subsample]
            y_sub = y[idx]
        else:
            y_sub = y

        dist_xy = torch.cdist(x_sub.unsqueeze(0), y_sub.unsqueeze(0)).squeeze(0)
        cd = (dist_xy.min(dim=1)[0].mean() + dist_xy.min(dim=0)[0].mean()) / 2

        # Landmark loss
        n = coords.shape[0]
        k = min(self.n_landmarks, n)
        indices = torch.linspace(0, n - 1, k, dtype=torch.long, device=coords.device)
        landmark_loss = F.mse_loss(student_x0[indices], teacher_x0[indices].detach())

        return cd + landmark_loss


# ============================================================================
# Forward with intermediates (from train_distillation.py, gradient-enabled)
# ============================================================================

def forward_with_intermediates(denoiser, features, coords, timestep, condition):
    """
    Run denoiser forward and capture intermediate features WITH gradients
    on the student side.
    """
    intermediates = {}

    t_embed = denoiser.time_embedding(timestep)
    x = denoiser.input_proj(features)
    x_coords = coords
    x_cond = condition['features']

    skip_features = []
    skip_coords = []
    skip_conds = []

    for i, enc_block in enumerate(denoiser.encoder_blocks):
        x = enc_block(x, x_coords, t_embed, x_cond)
        skip_features.append(x)
        skip_coords.append(x_coords)
        skip_conds.append(x_cond)
        intermediates[f'enc_{i}'] = x

        target_num = x.shape[0] // 2
        N = x.shape[0]
        if N > target_num:
            indices = torch.linspace(0, N - 1, target_num, dtype=torch.long, device=x.device)
            x = x[indices]
            x_coords = x_coords[indices]
            x_cond = x_cond[indices]
        x = denoiser.level_up[i](x)

    x = denoiser.bottleneck(x, x_coords, t_embed, x_cond)
    intermediates['bottleneck'] = x

    for i, dec_block in enumerate(denoiser.decoder_blocks):
        skip_feat = skip_features[-(i+1)]
        skip_coord = skip_coords[-(i+1)]
        skip_cond = skip_conds[-(i+1)]

        x = denoiser._upsample_points(x, x_coords, skip_coord)
        x_coords = skip_coord
        x_cond = skip_cond

        x = torch.cat([x, skip_feat], dim=-1)
        x = dec_block(x, x_coords, t_embed, x_cond)
        intermediates[f'dec_{i}'] = x

    noise_pred = denoiser.output_proj(x)
    return noise_pred, intermediates


def gpu_knn_interpolate(features, source_coords, target_coords, chunk_size=4096):
    """Map features from source to target via nearest neighbor on GPU."""
    all_indices = []
    for start in range(0, target_coords.shape[0], chunk_size):
        end = min(start + chunk_size, target_coords.shape[0])
        dists = torch.cdist(
            target_coords[start:end].unsqueeze(0).float(),
            source_coords.unsqueeze(0).float()
        ).squeeze(0)
        all_indices.append(dists.argmin(dim=-1))
    indices = torch.cat(all_indices, dim=0)
    return features[indices]


# ============================================================================
# Gradient extraction utilities
# ============================================================================

def get_shared_parameters(model):
    """
    Get the student's denoiser parameters (the shared backbone that all losses
    backprop through). We exclude the frozen Sonata encoder.
    """
    params = []
    param_names = []
    for name, p in model.denoiser.named_parameters():
        if p.requires_grad:
            params.append(p)
            param_names.append(f"denoiser.{name}")
    return params, param_names


def extract_gradient_vector(params):
    """Flatten all parameter gradients into a single vector."""
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def cosine_similarity_flat(g1, g2):
    """Cosine similarity between two flat gradient vectors."""
    dot = torch.dot(g1, g2)
    norm1 = g1.norm()
    norm2 = g2.norm()
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return (dot / (norm1 * norm2)).item()


def gradient_magnitude(g):
    """L2 norm of gradient vector."""
    return g.norm().item()


# ============================================================================
# Main analysis
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Gradient Conflict Analysis')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to SemanticKITTI dataset (e.g., sonata_depth_pro)')
    parser.add_argument('--teacher_ckpt', type=str, required=True)
    parser.add_argument('--student_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='gradient_analysis_results')
    parser.add_argument('--num_batches', type=int, default=200,
                        help='Number of batches to analyze')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (1 recommended for clean gradient signals)')
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--max_points', type=int, default=15000,
                        help='Max points per scan (controls memory)')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_model(device="cuda"):
    """Build SceneCompletionDiffusion model (same as evaluate.py)."""
    from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
    from models.diffusion_module import SceneCompletionDiffusion

    encoder = SonataEncoder(
        pretrained="facebook/sonata",
        freeze=True,
        enable_flash=False,
        feature_levels=[0],
    )
    condition_extractor = ConditionalFeatureExtractor(
        encoder, feature_levels=[0], fusion_type="concat"
    )
    model = SceneCompletionDiffusion(
        encoder=encoder,
        condition_extractor=condition_extractor,
        num_timesteps=1000,
        schedule="cosine",
    )
    return model.to(device)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Imports from the project ----
    from data.semantickitti import SemanticKITTI, collate_fn
    from utils.checkpoint import load_checkpoint

    # ---- Build models ----
    print("Building teacher model...")
    teacher = build_model(device)
    ckpt = load_checkpoint(args.teacher_ckpt)
    teacher.load_state_dict(ckpt['model_state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Building student model...")
    student = build_model(device)
    ckpt = load_checkpoint(args.student_ckpt)
    student.load_state_dict(ckpt['model_state_dict'])
    student.train()  # train mode for gradient computation
    # Freeze encoder, only denoiser gets gradients
    for p in student.encoder.parameters():
        p.requires_grad = False
    for p in student.condition_extractor.encoder.parameters():
        p.requires_grad = False

    shared_params, param_names = get_shared_parameters(student)
    total_params = sum(p.numel() for p in shared_params)
    print(f"Shared parameters (denoiser): {total_params:,} ({len(shared_params)} tensors)")

    # ---- Loss functions ----
    task_loss_fn = TaskLoss()
    output_loss_fn = OutputMatchingLoss()
    feature_loss_fn = FeatureAlignmentLoss()
    structural_loss_fn = StructuralLoss()

    # ---- Dataset (validation split, no augmentation) ----
    print("Loading validation dataset...")
    val_dataset = SemanticKITTI(
        root=args.data_path,
        split='val',
        voxel_size=args.voxel_size,
        max_points=args.max_points,
        use_ground_truth_maps=True,
        augmentation=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Validation set: {len(val_dataset)} samples, analyzing {args.num_batches} batches")

    # ---- Storage for results ----
    loss_names = ['task', 'output_matching', 'feature_alignment', 'structural']
    pair_names = []
    for i in range(len(loss_names)):
        for j in range(i + 1, len(loss_names)):
            pair_names.append(f"{loss_names[i]}_vs_{loss_names[j]}")

    results = {
        'cosine_similarities': {pair: [] for pair in pair_names},
        'gradient_magnitudes': {name: [] for name in loss_names},
        'loss_values': {name: [] for name in loss_names},
        'magnitude_ratios': {pair: [] for pair in pair_names},
    }

    # ---- Main analysis loop ----
    print(f"\nRunning gradient conflict analysis over {args.num_batches} batches...\n")
    t_start = time.time()

    batch_iter = iter(val_loader)
    for batch_idx in tqdm(range(args.num_batches), desc="Analyzing gradients"):
        # Get next batch, cycling if needed
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(val_loader)
            batch = next(batch_iter)

        # Move to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        partial_scan = {
            'coord': batch['partial_coord'],
            'color': batch['partial_color'],
            'normal': batch['partial_normal'],
            'batch': batch['partial_batch'],
        }
        complete_coord = batch['complete_coord']
        complete_batch = batch.get('complete_batch')

        # ---- Shared forward pass setup ----
        with torch.no_grad():
            # Teacher: extract condition features + forward
            teacher_cond, _ = teacher.condition_extractor(partial_scan)
            teacher_cond_mapped = gpu_knn_interpolate(
                teacher_cond, partial_scan['coord'], complete_coord
            )

            # Sample timestep and noise (shared)
            if complete_batch is not None:
                bs = complete_batch.max().item() + 1
            else:
                bs = 1
            t = torch.randint(0, teacher.scheduler.num_timesteps, (bs,), device=device)
            noise = torch.randn_like(complete_coord)

            if complete_batch is not None:
                t_per_point = t[complete_batch]
            else:
                t_per_point = t.expand(complete_coord.shape[0])

            sa = teacher.scheduler.sqrt_alphas_cumprod.to(device)[t_per_point].unsqueeze(-1)
            som = teacher.scheduler.sqrt_one_minus_alphas_cumprod.to(device)[t_per_point].unsqueeze(-1)
            noisy_scan = sa * complete_coord + som * noise

            # Teacher prediction
            teacher_pred, teacher_intermediates = forward_with_intermediates(
                teacher.denoiser, noisy_scan, complete_coord, t,
                {'features': teacher_cond_mapped}
            )
            # Detach teacher intermediates
            teacher_intermediates = {k: v.detach() for k, v in teacher_intermediates.items()}
            teacher_pred = teacher_pred.detach()

            # Teacher x_0 prediction (for structural loss)
            teacher_x0 = (noisy_scan - som * teacher_pred) / (sa + 1e-8)
            teacher_x0 = teacher_x0.detach()

        # Student: extract condition features (through frozen encoder)
        with torch.no_grad():
            student_cond, _ = student.condition_extractor(partial_scan)
            student_cond_mapped = gpu_knn_interpolate(
                student_cond, partial_scan['coord'], complete_coord
            )

        # Student forward (with gradients on denoiser)
        student_pred, student_intermediates = forward_with_intermediates(
            student.denoiser, noisy_scan.detach(), complete_coord.detach(), t.detach(),
            {'features': student_cond_mapped.detach()}
        )

        # Student x_0 prediction
        student_x0 = (noisy_scan.detach() - som.detach() * student_pred) / (sa.detach() + 1e-8)

        # ---- Compute per-loss gradients ----
        gradient_vectors = {}

        # 1. Task loss: student_pred vs noise (ground truth)
        loss_task = task_loss_fn(student_pred, noise.detach())
        student.denoiser.zero_grad()
        loss_task.backward(retain_graph=True)
        gradient_vectors['task'] = extract_gradient_vector(shared_params)
        results['loss_values']['task'].append(loss_task.item())

        # 2. Output matching loss: student_pred vs teacher_pred
        loss_output = output_loss_fn(student_pred, teacher_pred)
        student.denoiser.zero_grad()
        loss_output.backward(retain_graph=True)
        gradient_vectors['output_matching'] = extract_gradient_vector(shared_params)
        results['loss_values']['output_matching'].append(loss_output.item())

        # 3. Feature alignment loss
        loss_feature = feature_loss_fn(student_intermediates, teacher_intermediates)
        student.denoiser.zero_grad()
        loss_feature.backward(retain_graph=True)
        gradient_vectors['feature_alignment'] = extract_gradient_vector(shared_params)
        results['loss_values']['feature_alignment'].append(loss_feature.item())

        # 4. Structural loss
        loss_structural = structural_loss_fn(student_x0, teacher_x0, complete_coord.detach())
        student.denoiser.zero_grad()
        loss_structural.backward()
        gradient_vectors['structural'] = extract_gradient_vector(shared_params)
        results['loss_values']['structural'].append(loss_structural.item())

        # ---- Compute pairwise metrics ----
        for name in loss_names:
            mag = gradient_magnitude(gradient_vectors[name])
            results['gradient_magnitudes'][name].append(mag)

        for i in range(len(loss_names)):
            for j in range(i + 1, len(loss_names)):
                pair = f"{loss_names[i]}_vs_{loss_names[j]}"
                g_i = gradient_vectors[loss_names[i]]
                g_j = gradient_vectors[loss_names[j]]
                cos_sim = cosine_similarity_flat(g_i, g_j)
                results['cosine_similarities'][pair].append(cos_sim)

                mag_i = gradient_magnitude(g_i)
                mag_j = gradient_magnitude(g_j)
                ratio = mag_i / (mag_j + 1e-12)
                results['magnitude_ratios'][pair].append(ratio)

        # Clear GPU cache periodically
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nAnalysis complete in {elapsed:.1f}s ({elapsed/args.num_batches:.2f}s/batch)")

    # ---- Compute summary statistics ----
    summary = {}

    print("\n" + "=" * 70)
    print("GRADIENT CONFLICT ANALYSIS RESULTS")
    print("=" * 70)

    print("\n--- Pairwise Cosine Similarities ---")
    print(f"{'Pair':<45s} {'Mean':>8s} {'Std':>8s} {'%Neg':>8s}")
    print("-" * 70)
    for pair in pair_names:
        vals = np.array(results['cosine_similarities'][pair])
        mean_val = vals.mean()
        std_val = vals.std()
        pct_neg = (vals < 0).mean() * 100
        print(f"{pair:<45s} {mean_val:>8.4f} {std_val:>8.4f} {pct_neg:>7.1f}%")
        summary[f"cosine_{pair}_mean"] = float(mean_val)
        summary[f"cosine_{pair}_std"] = float(std_val)
        summary[f"cosine_{pair}_pct_negative"] = float(pct_neg)

    print("\n--- Gradient Magnitudes ---")
    print(f"{'Loss':<30s} {'Mean':>12s} {'Std':>12s}")
    print("-" * 55)
    for name in loss_names:
        vals = np.array(results['gradient_magnitudes'][name])
        mean_val = vals.mean()
        std_val = vals.std()
        print(f"{name:<30s} {mean_val:>12.6f} {std_val:>12.6f}")
        summary[f"grad_mag_{name}_mean"] = float(mean_val)
        summary[f"grad_mag_{name}_std"] = float(std_val)

    print("\n--- Magnitude Ratios ---")
    print(f"{'Pair':<45s} {'Mean':>10s} {'Std':>10s}")
    print("-" * 66)
    for pair in pair_names:
        vals = np.array(results['magnitude_ratios'][pair])
        mean_val = vals.mean()
        std_val = vals.std()
        print(f"{pair:<45s} {mean_val:>10.4f} {std_val:>10.4f}")
        summary[f"mag_ratio_{pair}_mean"] = float(mean_val)
        summary[f"mag_ratio_{pair}_std"] = float(std_val)

    print("\n--- Loss Values ---")
    print(f"{'Loss':<30s} {'Mean':>12s} {'Std':>12s}")
    print("-" * 55)
    for name in loss_names:
        vals = np.array(results['loss_values'][name])
        mean_val = vals.mean()
        std_val = vals.std()
        print(f"{name:<30s} {mean_val:>12.6f} {std_val:>12.6f}")
        summary[f"loss_{name}_mean"] = float(mean_val)
        summary[f"loss_{name}_std"] = float(std_val)

    # Key finding
    task_vs_output = np.array(results['cosine_similarities']['task_vs_output_matching'])
    task_vs_feature = np.array(results['cosine_similarities']['task_vs_feature_alignment'])
    task_vs_structural = np.array(results['cosine_similarities']['task_vs_structural'])

    print("\n" + "=" * 70)
    print("KEY FINDING: Gradient Conflicts with Task Loss")
    print("=" * 70)
    for label, vals in [("Output matching", task_vs_output),
                        ("Feature alignment", task_vs_feature),
                        ("Structural", task_vs_structural)]:
        mean_cos = vals.mean()
        direction = "CONFLICTING" if mean_cos < 0 else ("ALIGNED" if mean_cos > 0.3 else "NEAR-ORTHOGONAL")
        pct_neg = (vals < 0).mean() * 100
        print(f"  {label:>25s} vs Task: cos={mean_cos:+.4f} ({direction}, {pct_neg:.0f}% negative)")

    summary['num_batches'] = args.num_batches
    summary['total_denoiser_params'] = total_params

    # ---- Save results ----
    # Convert numpy arrays for JSON
    json_results = {
        'summary': summary,
        'config': {
            'data_path': args.data_path,
            'teacher_ckpt': args.teacher_ckpt,
            'student_ckpt': args.student_ckpt,
            'num_batches': args.num_batches,
            'batch_size': args.batch_size,
            'voxel_size': args.voxel_size,
            'max_points': args.max_points,
            'seed': args.seed,
        },
        'per_batch': {
            'cosine_similarities': {k: [float(x) for x in v]
                                     for k, v in results['cosine_similarities'].items()},
            'gradient_magnitudes': {k: [float(x) for x in v]
                                    for k, v in results['gradient_magnitudes'].items()},
            'magnitude_ratios': {k: [float(x) for x in v]
                                 for k, v in results['magnitude_ratios'].items()},
            'loss_values': {k: [float(x) for x in v]
                           for k, v in results['loss_values'].items()},
        }
    }

    json_path = os.path.join(args.output_dir, 'gradient_conflict_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ---- Generate plots ----
    generate_plots(results, pair_names, loss_names, args.output_dir)
    print(f"Plots saved to {args.output_dir}/")


# ============================================================================
# Plotting
# ============================================================================

def generate_plots(results, pair_names, loss_names, output_dir):
    """Generate all analysis plots."""

    # Pairs involving task loss (the key comparisons)
    task_pairs = [p for p in pair_names if p.startswith('task_vs_')]
    task_pair_labels = {
        'task_vs_output_matching': 'Task vs Output Matching',
        'task_vs_feature_alignment': 'Task vs Feature Alignment',
        'task_vs_structural': 'Task vs Structural',
    }
    # All other pairs
    other_pairs = [p for p in pair_names if not p.startswith('task_vs_')]

    # ---- Plot 1: Histogram of cosine similarities (task vs each alignment loss) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for ax, pair, color in zip(axes, task_pairs, colors):
        vals = np.array(results['cosine_similarities'][pair])
        label = task_pair_labels.get(pair, pair)
        ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Zero (orthogonal)')
        ax.axvline(x=vals.mean(), color='darkred', linestyle='-', linewidth=2,
                   label=f'Mean: {vals.mean():.4f}')
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(-1, 1)
        pct_neg = (vals < 0).mean() * 100
        ax.text(0.02, 0.95, f'{pct_neg:.0f}% negative',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Gradient Cosine Similarity: Task Loss vs Alignment Losses',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_histograms.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Plot 2: Gradient magnitude ratios over batches ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, pair, color in zip(axes, task_pairs, colors):
        vals = np.array(results['magnitude_ratios'][pair])
        batches = np.arange(len(vals))
        ax.plot(batches, vals, color=color, alpha=0.4, linewidth=0.5)
        # Smoothed trend
        window = min(20, len(vals) // 5)
        if window > 1:
            smoothed = np.convolve(vals, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                    color='black', linewidth=2, label=f'Moving avg (w={window})')
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Equal magnitude')
        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel('Magnitude Ratio (task / alignment)', fontsize=10)
        label = task_pair_labels.get(pair, pair)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_yscale('log')

    fig.suptitle('Gradient Magnitude Ratio Over Batches',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_ratios_over_batches.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Plot 3: Box plot comparing all loss pairs ----
    fig, ax = plt.subplots(figsize=(12, 6))

    box_data = []
    box_labels = []
    box_colors = []
    color_map = {
        'task_vs_output_matching': '#e74c3c',
        'task_vs_feature_alignment': '#3498db',
        'task_vs_structural': '#2ecc71',
        'output_matching_vs_feature_alignment': '#9b59b6',
        'output_matching_vs_structural': '#f39c12',
        'feature_alignment_vs_structural': '#1abc9c',
    }

    for pair in pair_names:
        vals = np.array(results['cosine_similarities'][pair])
        box_data.append(vals)
        # Shorter labels
        parts = pair.split('_vs_')
        short = f"{parts[0].replace('_', ' ').title()}\nvs\n{parts[1].replace('_', ' ').title()}"
        box_labels.append(short)
        box_colors.append(color_map.get(pair, '#95a5a6'))

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    notch=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Zero (orthogonal)')
    ax.set_ylabel('Cosine Similarity', fontsize=13)
    ax.set_title('Pairwise Gradient Cosine Similarity (All Loss Pairs)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_boxplot.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Plot 4: Gradient magnitude comparison ----
    fig, ax = plt.subplots(figsize=(10, 6))

    mag_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for name, color in zip(loss_names, mag_colors):
        vals = np.array(results['gradient_magnitudes'][name])
        ax.hist(vals, bins=40, color=color, alpha=0.5, label=name.replace('_', ' ').title(),
                edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Gradient L2 Norm', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Gradient Magnitude Distribution by Loss Type',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_magnitude_distributions.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Plot 5: Cosine similarity over batches (trend) ----
    fig, ax = plt.subplots(figsize=(14, 5))

    for pair, color in zip(task_pairs, colors[:3]):
        vals = np.array(results['cosine_similarities'][pair])
        batches = np.arange(len(vals))
        ax.scatter(batches, vals, color=color, alpha=0.2, s=8)
        window = min(20, len(vals) // 5)
        if window > 1:
            smoothed = np.convolve(vals, np.ones(window) / window, mode='valid')
            label = task_pair_labels.get(pair, pair)
            ax.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                    color=color, linewidth=2.5, label=f'{label} (smoothed)')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Gradient Cosine Similarity: Task Loss vs Alignment Losses (per batch)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_trend.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # ---- Plot 6: Summary figure (paper-ready) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean cosine similarities as bar chart
    task_pair_means = []
    task_pair_stds = []
    task_pair_short_labels = []
    for pair in task_pairs:
        vals = np.array(results['cosine_similarities'][pair])
        task_pair_means.append(vals.mean())
        task_pair_stds.append(vals.std())
        parts = pair.split('_vs_')
        task_pair_short_labels.append(parts[1].replace('_', '\n'))

    x_pos = np.arange(len(task_pairs))
    bars = ax1.bar(x_pos, task_pair_means, yerr=task_pair_stds,
                   color=colors[:3], alpha=0.8, edgecolor='black', linewidth=0.5,
                   capsize=5, error_kw={'linewidth': 1.5})
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(task_pair_short_labels, fontsize=11)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title('Mean Gradient Cosine Similarity\n(Task Loss vs Alignment Losses)',
                  fontsize=12, fontweight='bold')
    # Annotate bars with values
    for bar, mean_val in zip(bars, task_pair_means):
        y_offset = -0.05 if mean_val > 0 else 0.02
        ax1.text(bar.get_x() + bar.get_width() / 2, mean_val + y_offset,
                f'{mean_val:.3f}', ha='center', va='bottom' if mean_val < 0 else 'top',
                fontsize=11, fontweight='bold')

    # Right: magnitude ratios as bar chart
    task_mag_means = []
    task_mag_stds = []
    for name in loss_names:
        vals = np.array(results['gradient_magnitudes'][name])
        task_mag_means.append(vals.mean())
        task_mag_stds.append(vals.std())

    x_pos = np.arange(len(loss_names))
    mag_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    bars2 = ax2.bar(x_pos, task_mag_means, yerr=task_mag_stds,
                    color=mag_colors, alpha=0.8, edgecolor='black', linewidth=0.5,
                    capsize=5, error_kw={'linewidth': 1.5})
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([n.replace('_', '\n') for n in loss_names], fontsize=10)
    ax2.set_ylabel('Gradient L2 Norm', fontsize=12)
    ax2.set_title('Mean Gradient Magnitude by Loss Type',
                  fontsize=12, fontweight='bold')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_figure.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nGenerated 6 plots in {output_dir}/:")
    print("  1. cosine_similarity_histograms.png  — per-pair histograms")
    print("  2. magnitude_ratios_over_batches.png  — magnitude ratio trends")
    print("  3. cosine_similarity_boxplot.png      — all pairs box plot")
    print("  4. gradient_magnitude_distributions.png — magnitude histograms")
    print("  5. cosine_similarity_trend.png        — cosine sim over batches")
    print("  6. summary_figure.png                 — paper-ready summary")


if __name__ == "__main__":
    main()
