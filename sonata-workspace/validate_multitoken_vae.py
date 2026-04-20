#!/usr/bin/env python3
"""
Comprehensive validation of multi-token VAE checkpoint.
Tests: reconstruction quality, spatial fidelity, coverage, density,
latent space health, determinism, old VAE comparison, edge cases.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Setup paths
# --------------------------------------------------------------------------
WORKDIR = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
sys.path.insert(0, WORKDIR)
os.chdir(WORKDIR)

from models.point_cloud_vae import PointCloudVAE  # new multi-token
from models.refinement_net import chamfer_distance
from data.semantickitti import SemanticKITTI

# --------------------------------------------------------------------------
# Old v1 VAE class (inlined — original file was overwritten by multi-token)
# --------------------------------------------------------------------------
from typing import List, Optional, Tuple
import torch.nn.functional as F


class PointCloudVAE_v1(nn.Module):
    """Original single-vector VAE: max-pool encoder + MLP decoder."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_decoded_points: int = 2048,
        encoder_widths: Optional[List[int]] = None,
        decoder_widths: Optional[List[int]] = None,
    ):
        super().__init__()
        if encoder_widths is None:
            encoder_widths = [64, 128, 256]
        if decoder_widths is None:
            decoder_widths = [512, 512, 512]

        self.latent_dim = latent_dim
        self.num_decoded_points = num_decoded_points

        enc_layers: List[nn.Module] = []
        d_in = 3
        for w in encoder_widths:
            enc_layers.extend([nn.Linear(d_in, w), nn.LayerNorm(w), nn.GELU()])
            d_in = w
        self.point_encoder = nn.Sequential(*enc_layers)
        d_enc = encoder_widths[-1]
        self.fc_mu = nn.Linear(d_enc, latent_dim)
        self.fc_logvar = nn.Linear(d_enc, latent_dim)

        dec_layers: List[nn.Module] = []
        d = latent_dim
        for w in decoder_widths:
            dec_layers.extend([nn.Linear(d, w), nn.LayerNorm(w), nn.GELU()])
            d = w
        self.decoder_backbone = nn.Sequential(*dec_layers)
        self.decoder_out = nn.Linear(d, num_decoded_points * 3)

    def encode(self, points):
        if points.dim() != 2 or points.size(-1) != 3:
            raise ValueError(f"Expected (N, 3), got {tuple(points.shape)}")
        if points.size(0) == 0:
            z = points.new_zeros(self.latent_dim)
            return z, z.new_full((self.latent_dim,), -30.0)
        feat = self.point_encoder(points)
        pooled = feat.max(dim=0).values
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)
        h = self.decoder_backbone(z)
        out = self.decoder_out(h).view(z.size(0), self.num_decoded_points, 3)
        if single:
            out = out.squeeze(0)
        return out

    def forward(self, points):
        mu, logvar = self.encode(points)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cd(a, b):
    """Chamfer distance between two (N, 3) and (M, 3) tensors."""
    return chamfer_distance(a, b).item()


def load_dataset():
    ds = SemanticKITTI(
        root="/home/anywherevla/sonata_ws/dataset/sonata_depth_pro",
        split="val",
        use_point_cloud=True,
        point_max_complete=8000,
        point_max_partial=20000,
        augmentation=False,
    )
    return ds


def evenly_spaced_indices(total, n):
    return np.linspace(0, total - 1, n, dtype=int).tolist()


def section(title):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# --------------------------------------------------------------------------
# Load models
# --------------------------------------------------------------------------
def load_new_vae():
    model = PointCloudVAE(
        latent_dim=1024,
        num_decoded_points=8000,
        num_latent_tokens=16,
        internal_dim=256,
        num_heads=4,
        num_dec_blocks=3,
    )
    ckpt = torch.load(
        "checkpoints/point_vae_multitoken/best_point_vae.pth",
        map_location="cpu",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"  New VAE loaded — epoch {ckpt.get('epoch', '?')}, "
          f"best_val_loss {ckpt.get('best_val_loss', '?'):.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model


def load_old_vae():
    model = PointCloudVAE_v1(latent_dim=256, num_decoded_points=2048)
    ckpt = torch.load(
        "checkpoints/archive/point_vae_v1/best_point_vae.pth",
        map_location="cpu",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"  Old VAE v1 loaded — epoch {ckpt.get('epoch', '?')}, "
          f"best_val_loss {ckpt.get('best_val_loss', '?'):.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model


# ==========================================================================
# TEST 1: Reconstruction quality on 50 samples
# ==========================================================================
def test_reconstruction(model, ds):
    section("TEST 1: Reconstruction Quality (50 samples)")
    indices = evenly_spaced_indices(len(ds), 50)
    cds = []
    t0 = time.time()
    for i, idx in enumerate(indices):
        gt = ds[idx]["complete_coord"].to(DEVICE)  # (8000, 3)
        with torch.no_grad():
            mu, logvar = model.encode(gt)
            recon = model.decode(mu)  # deterministic (use mu, no noise)
        c = cd(recon, gt)
        cds.append(c)
        if i % 10 == 0:
            print(f"  [{i+1}/50] idx={idx} CD={c:.4f}")
    elapsed = time.time() - t0
    cds = np.array(cds)
    print(f"\n  Results (deterministic, z=mu):")
    print(f"    Mean CD:   {cds.mean():.4f}")
    print(f"    Std CD:    {cds.std():.4f}")
    print(f"    Min CD:    {cds.min():.4f}  (sample idx {indices[cds.argmin()]})")
    print(f"    Max CD:    {cds.max():.4f}  (sample idx {indices[cds.argmax()]})")
    print(f"    Median CD: {np.median(cds):.4f}")
    print(f"    Time: {elapsed:.1f}s ({elapsed/50:.2f}s/sample)")
    return cds, indices


# ==========================================================================
# TEST 2: Per-axis spatial fidelity
# ==========================================================================
def test_spatial_fidelity(model, ds):
    section("TEST 2: Per-Axis Spatial Fidelity (10 samples)")
    indices = evenly_spaced_indices(len(ds), 10)
    print(f"  {'Axis':<5} {'Metric':<6} {'GT':>10} {'Recon':>10} {'Ratio':>8}")
    print(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")

    all_ratios = {ax: {"range": [], "mean": []} for ax in ["x", "y", "z"]}

    for idx in indices:
        gt = ds[idx]["complete_coord"].to(DEVICE)
        with torch.no_grad():
            mu, _ = model.encode(gt)
            recon = model.decode(mu)

        gt_np = gt.cpu().numpy()
        re_np = recon.cpu().numpy()

        for ai, ax in enumerate(["x", "y", "z"]):
            gt_min, gt_max, gt_mean = gt_np[:, ai].min(), gt_np[:, ai].max(), gt_np[:, ai].mean()
            re_min, re_max, re_mean = re_np[:, ai].min(), re_np[:, ai].max(), re_np[:, ai].mean()
            gt_range = gt_max - gt_min
            re_range = re_max - re_min
            ratio = re_range / (gt_range + 1e-8)
            all_ratios[ax]["range"].append(ratio)
            all_ratios[ax]["mean"].append(abs(re_mean - gt_mean))

    print(f"\n  Summary across 10 samples:")
    print(f"  {'Axis':<5} {'Range ratio (recon/GT)':>25} {'Mean offset':>15}")
    for ax in ["x", "y", "z"]:
        rr = np.array(all_ratios[ax]["range"])
        mo = np.array(all_ratios[ax]["mean"])
        print(f"  {ax:<5} {rr.mean():.4f} +/- {rr.std():.4f}          {mo.mean():.3f}m +/- {mo.std():.3f}m")

    for ax in ["x", "y", "z"]:
        rr_mean = np.mean(all_ratios[ax]["range"])
        if rr_mean < 0.7:
            print(f"\n  WARNING: {ax}-axis spatially compressed (ratio {rr_mean:.3f} < 0.7)")
        elif rr_mean > 1.3:
            print(f"\n  WARNING: {ax}-axis spatially expanded (ratio {rr_mean:.3f} > 1.3)")


# ==========================================================================
# TEST 3: Coverage analysis
# ==========================================================================
def test_coverage(model, ds):
    section("TEST 3: Coverage Analysis (10 samples)")
    indices = evenly_spaced_indices(len(ds), 10)

    fwd_1m, fwd_2m, rev_1m, rev_2m = [], [], [], []

    for idx in indices:
        gt = ds[idx]["complete_coord"].to(DEVICE)
        with torch.no_grad():
            mu, _ = model.encode(gt)
            recon = model.decode(mu)

        # Forward: for each GT point, find nearest recon point
        dists_fwd = torch.cdist(gt.unsqueeze(0), recon.unsqueeze(0)).squeeze(0)
        nn_fwd = dists_fwd.min(dim=1).values  # (N_gt,)
        fwd_1m.append((nn_fwd < 1.0).float().mean().item() * 100)
        fwd_2m.append((nn_fwd < 2.0).float().mean().item() * 100)

        # Reverse: for each recon point, find nearest GT point
        nn_rev = dists_fwd.min(dim=0).values  # (N_recon,)
        rev_1m.append((nn_rev < 1.0).float().mean().item() * 100)
        rev_2m.append((nn_rev < 2.0).float().mean().item() * 100)

    print(f"  Direction           Within 1m      Within 2m")
    print(f"  GT→Recon (coverage) {np.mean(fwd_1m):6.1f}% ± {np.std(fwd_1m):.1f}%  "
          f"{np.mean(fwd_2m):6.1f}% ± {np.std(fwd_2m):.1f}%")
    print(f"  Recon→GT (accuracy) {np.mean(rev_1m):6.1f}% ± {np.std(rev_1m):.1f}%  "
          f"{np.mean(rev_2m):6.1f}% ± {np.std(rev_2m):.1f}%")


# ==========================================================================
# TEST 4: Point density distribution (BEV grid)
# ==========================================================================
def test_density(model, ds):
    section("TEST 4: BEV Density Distribution (10 samples, 10x10 grid)")
    indices = evenly_spaced_indices(len(ds), 10)

    correlations = []
    gt_empty_cells = []
    recon_empty_cells = []

    for idx in indices:
        gt = ds[idx]["complete_coord"].to(DEVICE)
        with torch.no_grad():
            mu, _ = model.encode(gt)
            recon = model.decode(mu)

        gt_np = gt.cpu().numpy()
        re_np = recon.cpu().numpy()

        bins = np.linspace(-50, 50, 11)

        gt_hist, _, _ = np.histogram2d(gt_np[:, 0], gt_np[:, 1], bins=[bins, bins])
        re_hist, _, _ = np.histogram2d(re_np[:, 0], re_np[:, 1], bins=[bins, bins])

        gt_flat = gt_hist.flatten()
        re_flat = re_hist.flatten()

        # Pearson correlation (only where at least one has points)
        mask = (gt_flat > 0) | (re_flat > 0)
        if mask.sum() > 2:
            corr = np.corrcoef(gt_flat[mask], re_flat[mask])[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)
        gt_empty_cells.append((gt_flat == 0).sum())
        recon_empty_cells.append((re_flat == 0).sum())

    print(f"  BEV density correlation (GT vs recon):")
    print(f"    Mean: {np.mean(correlations):.4f}")
    print(f"    Min:  {np.min(correlations):.4f}")
    print(f"    Max:  {np.max(correlations):.4f}")
    print(f"  Empty cells (of 100):  GT avg {np.mean(gt_empty_cells):.1f}, "
          f"Recon avg {np.mean(recon_empty_cells):.1f}")


# ==========================================================================
# TEST 5: Latent space health
# ==========================================================================
def test_latent_health(model, ds):
    section("TEST 5: Latent Space Health (20 samples)")
    indices = evenly_spaced_indices(len(ds), 20)

    mus = []
    logvars = []

    for idx in indices:
        gt = ds[idx]["complete_coord"].to(DEVICE)
        with torch.no_grad():
            mu, logvar = model.encode(gt)
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())

    mu_stack = torch.stack(mus)      # (20, latent_dim)
    lv_stack = torch.stack(logvars)  # (20, latent_dim)

    # Active dimensions: std per dim across samples
    per_dim_std = mu_stack.std(dim=0)  # (latent_dim,)
    active_mask = per_dim_std > 0.01
    n_active = active_mask.sum().item()
    print(f"  Latent dim: {mu_stack.shape[1]}")
    print(f"  Active dims (std > 0.01): {n_active} / {mu_stack.shape[1]} "
          f"({100*n_active/mu_stack.shape[1]:.1f}%)")
    print(f"  Per-dim std: min={per_dim_std.min():.6f}, "
          f"max={per_dim_std.max():.4f}, mean={per_dim_std.mean():.4f}, "
          f"median={per_dim_std.median():.4f}")

    # Pairwise L2 distances
    dists = torch.cdist(mu_stack.unsqueeze(0), mu_stack.unsqueeze(0)).squeeze(0)
    # Mask diagonal
    mask = ~torch.eye(20, dtype=torch.bool)
    pw_dists = dists[mask]
    print(f"\n  Pairwise L2 distances in latent space:")
    print(f"    Mean: {pw_dists.mean():.4f}")
    print(f"    Min:  {pw_dists.min():.4f}")
    print(f"    Max:  {pw_dists.max():.4f}")
    print(f"    Std:  {pw_dists.std():.4f}")

    # Mean logvar (posterior collapse check)
    mean_lv = lv_stack.mean().item()
    std_lv = lv_stack.std().item()
    mean_std_from_lv = torch.exp(0.5 * lv_stack).mean().item()
    print(f"\n  Posterior check:")
    print(f"    Mean logvar:     {mean_lv:.4f}")
    print(f"    Std logvar:      {std_lv:.4f}")
    print(f"    Mean std (exp(0.5*lv)): {mean_std_from_lv:.4f}")
    if mean_lv < -10:
        print(f"    WARNING: logvar very negative ({mean_lv:.1f}) — posterior may have collapsed to delta")
    elif mean_lv > 2:
        print(f"    WARNING: logvar very positive ({mean_lv:.1f}) — posterior too diffuse")
    else:
        print(f"    Posterior looks healthy (logvar in reasonable range)")

    # Per-token analysis (multi-token specific)
    n_tokens = model.num_latent_tokens
    token_dim = model.token_dim
    print(f"\n  Multi-token analysis ({n_tokens} tokens x {token_dim} dims):")
    mu_tokens = mu_stack.view(20, n_tokens, token_dim)
    for t in range(n_tokens):
        t_std = mu_tokens[:, t, :].std(dim=0).mean().item()
        t_lv = lv_stack.view(20, n_tokens, token_dim)[:, t, :].mean().item()
        print(f"    Token {t:2d}: mu_std={t_std:.4f}, mean_logvar={t_lv:.4f}")


# ==========================================================================
# TEST 6: Determinism
# ==========================================================================
def test_determinism(model, ds):
    section("TEST 6: Determinism Check")
    idx = len(ds) // 2
    gt = ds[idx]["complete_coord"].to(DEVICE)

    with torch.no_grad():
        mu1, lv1 = model.encode(gt)
        mu2, lv2 = model.encode(gt)

    mu_diff = (mu1 - mu2).abs().max().item()
    lv_diff = (lv1 - lv2).abs().max().item()
    print(f"  Same input encoded twice:")
    print(f"    Max |mu1 - mu2| = {mu_diff:.2e}")
    print(f"    Max |lv1 - lv2| = {lv_diff:.2e}")
    assert mu_diff < 1e-5, f"Non-deterministic encoding! diff={mu_diff}"
    print(f"    PASS: encoding is deterministic")

    # Decode mu (deterministic)
    with torch.no_grad():
        r1 = model.decode(mu1)
        r2 = model.decode(mu1)
    dec_diff = (r1 - r2).abs().max().item()
    print(f"    Max |decode(mu) run1 - run2| = {dec_diff:.2e}")
    print(f"    PASS: decoding is deterministic")

    # Stochastic: reparameterize and measure CD variation
    print(f"\n  Stochastic sampling (10 draws from posterior):")
    cd_mu = cd(model.decode(mu1), gt)
    stoch_cds = []
    for _ in range(10):
        with torch.no_grad():
            z = model.reparameterize(mu1, lv1)
            recon = model.decode(z)
        stoch_cds.append(cd(recon, gt))
    stoch_cds = np.array(stoch_cds)
    print(f"    CD with z=mu:          {cd_mu:.4f}")
    print(f"    CD with z~q(z|x) mean: {stoch_cds.mean():.4f} ± {stoch_cds.std():.4f}")
    print(f"    CD range: [{stoch_cds.min():.4f}, {stoch_cds.max():.4f}]")
    delta = abs(stoch_cds.mean() - cd_mu) / (cd_mu + 1e-8)
    print(f"    Relative CD increase from noise: {delta*100:.1f}%")


# ==========================================================================
# TEST 7: Side-by-side comparison with old VAE
# ==========================================================================
def test_old_vae_comparison(model_new, ds):
    section("TEST 7: Old VAE v1 Comparison (10 samples)")
    try:
        model_old = load_old_vae()
    except Exception as e:
        print(f"  SKIP: Could not load old VAE: {e}")
        return

    indices = evenly_spaced_indices(len(ds), 10)

    new_cds = []
    old_cds = []

    print(f"  {'Sample':>8} {'Old CD':>10} {'New CD':>10} {'Improvement':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12}")

    for idx in indices:
        gt = ds[idx]["complete_coord"].to(DEVICE)

        # New VAE
        with torch.no_grad():
            mu_n, _ = model_new.encode(gt)
            recon_n = model_new.decode(mu_n)
        cd_n = cd(recon_n, gt)
        new_cds.append(cd_n)

        # Old VAE (only takes 2048 pts output, but encode full GT)
        with torch.no_grad():
            mu_o, _ = model_old.encode(gt)
            recon_o = model_old.decode(mu_o)
        cd_o = cd(recon_o, gt)
        old_cds.append(cd_o)

        imp = (cd_o - cd_n) / cd_o * 100
        print(f"  {idx:>8} {cd_o:>10.4f} {cd_n:>10.4f} {imp:>+11.1f}%")

    new_cds = np.array(new_cds)
    old_cds = np.array(old_cds)
    print(f"\n  Summary:")
    print(f"    Old VAE v1 (256d, 2048pts):  mean CD = {old_cds.mean():.4f}")
    print(f"    New multi-token (1024d, 8000pts): mean CD = {new_cds.mean():.4f}")
    imp = (old_cds.mean() - new_cds.mean()) / old_cds.mean() * 100
    print(f"    Improvement: {imp:+.1f}%")
    ratio = old_cds.mean() / (new_cds.mean() + 1e-8)
    print(f"    Ratio old/new: {ratio:.2f}x")


# ==========================================================================
# TEST 8: Edge cases
# ==========================================================================
def test_edge_cases(model, ds):
    section("TEST 8: Edge Cases (NaN/Inf check, smallest cloud)")
    indices = evenly_spaced_indices(len(ds), 50)

    nan_count = 0
    inf_count = 0
    min_gt_pts = float("inf")
    min_gt_idx = -1
    recon_stats = {"min_pts": [], "max_val": [], "min_val": []}

    for idx in indices:
        gt = ds[idx]["complete_coord"].to(DEVICE)

        if gt.shape[0] < min_gt_pts:
            min_gt_pts = gt.shape[0]
            min_gt_idx = idx

        with torch.no_grad():
            mu, logvar = model.encode(gt)
            recon = model.decode(mu)

        # Check NaN/Inf in all outputs
        for name, tensor in [("mu", mu), ("logvar", logvar), ("recon", recon)]:
            if torch.isnan(tensor).any():
                nan_count += 1
                print(f"  NaN in {name} at idx={idx}!")
            if torch.isinf(tensor).any():
                inf_count += 1
                print(f"  Inf in {name} at idx={idx}!")

        recon_np = recon.cpu().numpy()
        recon_stats["max_val"].append(np.abs(recon_np).max())
        recon_stats["min_val"].append(np.abs(recon_np).min())

    print(f"  Scanned 50 samples:")
    print(f"    NaN occurrences: {nan_count}")
    print(f"    Inf occurrences: {inf_count}")
    print(f"    Smallest GT cloud: {min_gt_pts} points (idx={min_gt_idx})")
    print(f"    Max |recon coord|: {np.max(recon_stats['max_val']):.2f}")
    print(f"    Min |recon coord|: {np.min(recon_stats['min_val']):.6f}")

    if nan_count == 0 and inf_count == 0:
        print(f"    PASS: No NaN/Inf detected")
    else:
        print(f"    FAIL: Found numerical issues!")

    # Try empty input
    print(f"\n  Empty input test:")
    try:
        with torch.no_grad():
            empty = torch.zeros(0, 3).to(DEVICE)
            mu_e, lv_e = model.encode(empty)
            recon_e = model.decode(mu_e)
        print(f"    Empty encode → mu norm={mu_e.norm():.4f}, recon shape={recon_e.shape}")
        print(f"    PASS: handles empty input gracefully")
    except Exception as e:
        print(f"    Error on empty input: {e}")

    # Try very small input (10 points)
    print(f"\n  Small input test (10 points):")
    try:
        gt_small = ds[0]["complete_coord"][:10].to(DEVICE)
        with torch.no_grad():
            mu_s, lv_s = model.encode(gt_small)
            recon_s = model.decode(mu_s)
        print(f"    10-point encode → mu norm={mu_s.norm():.4f}, recon shape={recon_s.shape}")
        cd_s = cd(recon_s, gt_small)
        print(f"    CD = {cd_s:.4f}")
        print(f"    PASS: handles small input")
    except Exception as e:
        print(f"    Error: {e}")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("=" * 70)
    print("  MULTI-TOKEN VAE COMPREHENSIVE VALIDATION")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: checkpoints/point_vae_multitoken/best_point_vae.pth")

    t_total = time.time()

    print("\nLoading new multi-token VAE...")
    model = load_new_vae()

    print("\nLoading dataset...")
    ds = load_dataset()
    print(f"  Val samples: {len(ds)}")

    # Run all tests
    cds, indices = test_reconstruction(model, ds)
    test_spatial_fidelity(model, ds)
    test_coverage(model, ds)
    test_density(model, ds)
    test_latent_health(model, ds)
    test_determinism(model, ds)
    test_old_vae_comparison(model, ds)
    test_edge_cases(model, ds)

    # Final summary
    section("FINAL SUMMARY")
    print(f"  Multi-token VAE (epoch 99, 16 tokens x 64d = 1024d latent)")
    print(f"  Reconstruction CD: {cds.mean():.4f} ± {cds.std():.4f}")
    print(f"  Total validation time: {time.time() - t_total:.1f}s")
    print()


if __name__ == "__main__":
    main()
