"""
VAE-GAN training: PointCloudVAE V3 + WGAN-GP discriminator.

The VAE V3 (multi-token, 32 latent tokens) is kept intact and acts as the
generator.  A multi-scale PointNet critic is trained adversarially with
Wasserstein loss + gradient penalty (WGAN-GP) + feature-matching loss.

Generator (VAE) loss per step:
    L_G = L_chamfer + beta_kl * L_KL
        + lambda_adv * L_wasserstein_gen
        + lambda_fm  * L_feature_match
    L_wasserstein_gen = -E[D(fake)]   (fool the critic)
    L_feature_match   = sum_l L1(D_l(fake), D_l(real).detach())

Critic (discriminator) loss, n_critic steps per generator step:
    L_D = E[D(fake)] - E[D(real)] + lambda_gp * GP
    GP  = E[(||∇D(x̂)||₂ - 1)²]  on interpolated samples

Publication-quality additions (defaults ON):
  • Feature-matching loss (Salimans 2016, Larsen 2016): stabilises training
    and improves sample quality beyond pure adversarial loss.
  • EMA of VAE weights (StyleGAN/BigGAN convention): a running average of
    the generator is maintained throughout training and saved alongside
    the live weights.  Use EMA weights at inference time for ~0.5–2 dB
    better reconstruction quality.

Training schedule:
    Epochs 0 … disc_warmup_epochs-1  : VAE-only (no adversarial loss)
    Epochs disc_warmup_epochs … end  : full VAE-GAN

This warm-up ensures the VAE has learned a reasonable reconstruction before
the discriminator starts pushing it, preventing early adversarial collapse.

Recommended usage:
    # Fine-tune from a pre-trained VAE V3 checkpoint (strongly recommended):
    python training/train_vae_gan.py --resume_vae checkpoints/point_vae_v3/best_point_vae.pth

    # Train from scratch (longer convergence):
    python training/train_vae_gan.py
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

import copy

from data.semantickitti import SemanticKITTI, collate_fn
from models.discriminator import (
    MultiScalePointDiscriminator,
    feature_matching_loss,
    gradient_penalty,
)
from models.point_cloud_vae import PointCloudVAE, kl_divergence
from models.refinement_net import chamfer_distance
from utils.logger import setup_logger


# ---------------------------------------------------------------------------
# EMA of generator weights (StyleGAN-style)
# ---------------------------------------------------------------------------

class WeightEMA:
    """
    Exponential moving average of a model's parameters.

    Maintains a separate "shadow" copy of the generator that decays toward the
    live weights at each step:
        ema_w  ←  decay * ema_w  +  (1 - decay) * live_w

    Inference / evaluation should use these EMA weights — they consistently
    yield lower reconstruction error than the live weights in modern GANs
    (Karras et al. StyleGAN2, Brock et al. BigGAN).

    The shadow model is in `eval()` mode and never receives gradients.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for p_ema, p in zip(self.shadow.parameters(), model.parameters()):
            p_ema.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers (e.g. None here, but safe-guard) — copy from live.
        for b_ema, b in zip(self.shadow.buffers(), model.buffers()):
            b_ema.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train VAE-GAN: PointCloudVAE V3 + WGAN-GP discriminator"
    )
    p.add_argument("--data_path", type=str,
                   default=os.path.expanduser("~/Simon_ws/dataset/SemanticKITTI/dataset"))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=100)

    # Generator (VAE) optimiser — matches V3 baseline
    p.add_argument("--lr_gen", type=float, default=3e-4)
    p.add_argument("--weight_decay_gen", type=float, default=1e-4)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=10,
                   help="LR warmup for the generator scheduler")

    # Critic (discriminator) optimiser — Adam(beta1=0) as per WGAN-GP paper
    p.add_argument("--lr_disc", type=float, default=1e-4)
    p.add_argument("--n_critic", type=int, default=5,
                   help="Discriminator update steps per generator step")
    p.add_argument("--lambda_gp", type=float, default=10.0,
                   help="Gradient penalty coefficient (WGAN-GP, default 10)")
    p.add_argument("--lambda_adv", type=float, default=0.1,
                   help="Adversarial loss weight in generator loss")
    p.add_argument("--lambda_fm", type=float, default=10.0,
                   help="Feature-matching loss weight (Salimans 2016). Set to 0 to disable.")
    p.add_argument("--disc_warmup_epochs", type=int, default=10,
                   help="Epochs of VAE-only training before enabling the discriminator")

    # EMA of generator weights (publication-grade GAN practice)
    p.add_argument("--use_ema", action="store_true", default=True,
                   help="Maintain EMA of VAE weights (recommended for publication)")
    p.add_argument("--no_ema", dest="use_ema", action="store_false",
                   help="Disable EMA tracking of VAE weights")
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="EMA decay (closer to 1.0 = slower-moving average)")
    p.add_argument("--ema_start_epoch", type=int, default=0,
                   help="Begin EMA tracking only after this epoch (warmup-friendly)")

    # VAE architecture — keep identical to V3
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--num_decoded_points", type=int, default=8000)
    p.add_argument("--num_latent_tokens", type=int, default=32)
    p.add_argument("--internal_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_dec_blocks", type=int, default=5)

    # VAE loss
    p.add_argument("--beta_kl", type=float, default=1e-3)

    # Data
    p.add_argument("--point_max_complete", type=int, default=8000)
    p.add_argument("--point_max_partial", type=int, default=20000)

    # Per-sample normalisation (must match V3 training)
    p.add_argument("--center_gt", action="store_true", default=True)
    p.add_argument("--scale_gt", action="store_true", default=True)

    # IO
    p.add_argument("--output_dir", type=str, default="checkpoints/vae_gan")
    p.add_argument("--log_dir", type=str, default="logs/vae_gan")
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--resume_vae", type=str, default=None,
                   help="Pre-trained VAE V3 checkpoint (warm-start, strongly recommended)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume full VAE-GAN from a previous VAE-GAN checkpoint")

    # Device
    p.add_argument("--device", type=str, default=None,
                   help="Training device: 'cuda', 'cuda:1', 'cpu'. "
                        "Defaults to cuda if available, otherwise cpu.")

    # GT data version
    p.add_argument("--gt_subdir", type=str, default="ground_truth")
    p.add_argument("--gt_name_suffix", type=str, default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_points(
    pts: torch.Tensor, center: bool = True, scale: bool = True
):
    """Center and optionally scale a point cloud to [-1, 1]. Returns (pts_norm, centroid, scale)."""
    centroid = pts.mean(dim=0) if center else torch.zeros(3, device=pts.device)
    pts_c = pts - centroid
    s = pts_c.abs().max().clamp(min=1e-6) if scale else torch.ones(1, device=pts.device)
    return pts_c / s, centroid, s


def sample_to_n(pts: torch.Tensor, n: int) -> torch.Tensor:
    """Random subsample or tile to exactly n points."""
    N = pts.size(0)
    if N == n:
        return pts
    if N > n:
        return pts[torch.randperm(N, device=pts.device)[:n]]
    # tile if too few points
    repeats = (n + N - 1) // N
    return pts.repeat(repeats, 1)[:n]


def build_norm_batch(cc, cb, bsz, center, scale, n_pts, device):
    """
    Builds per-sample normalised point cloud list and a stacked (B, n_pts, 3)
    real batch subsampled to n_pts for the discriminator.
    """
    pts_list = []
    for b in range(bsz):
        pts = cc[cb == b]
        pts_n, _, _ = normalize_points(pts, center, scale)
        pts_list.append(pts_n)
    real_batch = torch.stack([sample_to_n(p, n_pts) for p in pts_list])
    return pts_list, real_batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    vae, disc, opt_gen, opt_disc, loader, epoch, args, writer, device,
    ema: "WeightEMA | None" = None,
):
    vae.train()
    disc.train()

    use_adv = epoch >= args.disc_warmup_epochs
    use_fm = use_adv and args.lambda_fm > 0
    K = args.num_decoded_points

    stats = dict(gen=0.0, disc=0.0, chamfer=0.0, kl=0.0, adv=0.0, fm=0.0)
    n_batches = 0

    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        cc = batch["complete_coord"]
        cb = batch["complete_batch"]
        bsz = int(cb.max().item()) + 1

        pts_list, real_batch = build_norm_batch(
            cc, cb, bsz, args.center_gt, args.scale_gt, K, device
        )

        # ----------------------------------------------------------------
        # Discriminator steps (n_critic per generator step)
        # ----------------------------------------------------------------
        if use_adv:
            disc.requires_grad_(True)
            vae.requires_grad_(False)

            disc_loss_accum = 0.0
            for _ in range(args.n_critic):
                with torch.no_grad():
                    fake_list = []
                    for pts_n in pts_list:
                        mu, logvar = vae.encode(pts_n)
                        z = vae.reparameterize(mu, logvar)
                        fake_list.append(vae.decode(z))
                    fake_batch = torch.stack(fake_list)   # (B, K, 3)

                opt_disc.zero_grad()
                real_score = disc(real_batch)             # (B, 1)
                fake_score = disc(fake_batch.detach())    # (B, 1)
                gp = gradient_penalty(disc, real_batch, fake_batch.detach(), device)
                d_loss = (
                    fake_score.mean() - real_score.mean()
                    + args.lambda_gp * gp
                )
                d_loss.backward()
                opt_disc.step()
                disc_loss_accum += d_loss.item()

            stats["disc"] += disc_loss_accum / args.n_critic
            disc.requires_grad_(False)
            vae.requires_grad_(True)

        # ----------------------------------------------------------------
        # Generator (VAE) step
        # ----------------------------------------------------------------
        opt_gen.zero_grad()

        recon_list, mu_list, logvar_list = [], [], []
        for pts_n in pts_list:
            mu, logvar = vae.encode(pts_n)
            z = vae.reparameterize(mu, logvar)
            recon_list.append(vae.decode(z))
            mu_list.append(mu)
            logvar_list.append(logvar)

        # Chamfer: per-sample mean
        chamfer_losses = [
            chamfer_distance(recon_list[b], pts_list[b]) for b in range(bsz)
        ]
        chamfer_term = torch.stack(chamfer_losses).mean()

        # KL divergence across batch
        mu_all = torch.stack(mu_list)           # (B, latent_dim)
        logvar_all = torch.stack(logvar_list)   # (B, latent_dim)
        kl_term = kl_divergence(mu_all, logvar_all)

        g_loss = chamfer_term + args.beta_kl * kl_term

        fm_value = 0.0
        if use_adv:
            fake_batch_g = torch.stack(recon_list)          # (B, K, 3) — with grads
            if use_fm:
                # Single D forward pass on each side, taking score + features
                fake_score_g, fake_feats = disc(fake_batch_g, return_features=True)
                with torch.no_grad():
                    _, real_feats = disc(real_batch, return_features=True)
                adv_loss = -fake_score_g.mean()
                fm_loss = feature_matching_loss(real_feats, fake_feats)
                g_loss = (
                    g_loss
                    + args.lambda_adv * adv_loss
                    + args.lambda_fm * fm_loss
                )
                fm_value = fm_loss.item()
                stats["fm"] += fm_value
            else:
                adv_loss = -disc(fake_batch_g).mean()       # Wasserstein gen loss
                g_loss = g_loss + args.lambda_adv * adv_loss
            stats["adv"] += adv_loss.item()

        g_loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.gradient_clip)
        opt_gen.step()

        # EMA update — track every generator step once enabled
        if ema is not None and epoch >= args.ema_start_epoch:
            ema.update(vae)

        # Ensure disc params are trainable again for next batch
        disc.requires_grad_(True)

        stats["gen"] += g_loss.item()
        stats["chamfer"] += chamfer_term.item()
        stats["kl"] += kl_term.item()
        n_batches += 1

        step = epoch * len(loader) + i
        writer.add_scalar("train/gen_loss", g_loss.item(), step)
        writer.add_scalar("train/chamfer", chamfer_term.item(), step)
        writer.add_scalar("train/kl", kl_term.item(), step)
        if use_adv:
            writer.add_scalar("train/disc_loss", disc_loss_accum / args.n_critic, step)
            writer.add_scalar("train/adv_loss", adv_loss.item(), step)
            if use_fm:
                writer.add_scalar("train/fm_loss", fm_value, step)

    nb = max(n_batches, 1)
    return {k: v / nb for k, v in stats.items()}


@torch.no_grad()
def validate(vae, loader, args, device):
    vae.eval()
    tot = 0.0
    n = 0
    for batch in tqdm(loader, desc="Val"):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        cc = batch["complete_coord"]
        cb = batch["complete_batch"]
        bsz = int(cb.max().item()) + 1
        for b in range(bsz):
            pts = cc[cb == b]
            pts_n, _, _ = normalize_points(pts, args.center_gt, args.scale_gt)
            mu, logvar = vae.encode(pts_n)
            z = vae.reparameterize(mu, logvar)
            recon = vae.decode(z)
            cd = chamfer_distance(recon, pts_n)
            kl = kl_divergence(mu.unsqueeze(0), logvar.unsqueeze(0))
            tot += (cd + args.beta_kl * kl).item() / bsz
        n += 1
    return tot / max(n, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_vaegan_checkpoint(
    path, vae, disc, opt_gen, opt_disc, epoch, best_val,
    ema: "WeightEMA | None" = None,
):
    """Save full VAE-GAN state, including EMA shadow weights when enabled."""
    payload = {
        "epoch": epoch,
        "vae_state_dict": vae.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "best_val_loss": best_val,
    }
    if ema is not None:
        # EMA weights — preferred for downstream inference / evaluation
        payload["vae_ema_state_dict"] = ema.state_dict()
    torch.save(payload, path)


def load_vaegan_checkpoint(
    path, vae, disc, opt_gen, opt_disc, device,
    ema: "WeightEMA | None" = None,
):
    ck = torch.load(path, map_location=device)
    vae.load_state_dict(ck["vae_state_dict"])
    disc.load_state_dict(ck["disc_state_dict"])
    opt_gen.load_state_dict(ck["opt_gen_state_dict"])
    opt_disc.load_state_dict(ck["opt_disc_state_dict"])
    if ema is not None and "vae_ema_state_dict" in ck:
        ema.load_state_dict(ck["vae_ema_state_dict"])
    return ck.get("epoch", 0), ck.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(args)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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

    # --- Models ---
    vae = PointCloudVAE(
        latent_dim=args.latent_dim,
        num_decoded_points=args.num_decoded_points,
        num_latent_tokens=args.num_latent_tokens,
        internal_dim=args.internal_dim,
        num_heads=args.num_heads,
        num_dec_blocks=args.num_dec_blocks,
    ).to(device)

    disc = MultiScalePointDiscriminator(in_dim=3).to(device)

    logger.info(f"VAE parameters:           {sum(p.numel() for p in vae.parameters() if p.requires_grad):,}")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters() if p.requires_grad):,}")

    # --- Optimisers ---
    # Generator: AdamW matches V3 baseline
    opt_gen = optim.AdamW(vae.parameters(), lr=args.lr_gen, weight_decay=args.weight_decay_gen)
    # Discriminator: Adam(beta1=0, beta2=0.9) as per WGAN-GP paper
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(0.0, 0.9))

    # Generator LR schedule: linear warmup → cosine anneal
    warmup_sch = optim.lr_scheduler.LinearLR(
        opt_gen, start_factor=0.01, total_iters=args.warmup_epochs,
    )
    cosine_sch = optim.lr_scheduler.CosineAnnealingLR(
        opt_gen, T_max=max(args.num_epochs - args.warmup_epochs, 1), eta_min=1e-6,
    )
    sch_gen = optim.lr_scheduler.SequentialLR(
        opt_gen, [warmup_sch, cosine_sch], milestones=[args.warmup_epochs],
    )

    start = 0
    best = float("inf")

    # Warm-start generator from pre-trained VAE V3 (recommended)
    if args.resume_vae:
        ck = torch.load(args.resume_vae, map_location=device)
        state = ck.get("model_state_dict", ck)
        vae.load_state_dict(state, strict=True)
        logger.info(f"Loaded pre-trained VAE V3 from {args.resume_vae}")

    # Build EMA shadow AFTER any warm-start so it begins from the right state.
    ema = WeightEMA(vae, decay=args.ema_decay) if args.use_ema else None
    if ema is not None:
        logger.info(f"EMA tracking enabled (decay={args.ema_decay}, start_epoch={args.ema_start_epoch})")

    # Or resume a previous VAE-GAN run
    if args.resume:
        start, best = load_vaegan_checkpoint(
            args.resume, vae, disc, opt_gen, opt_disc, device, ema=ema,
        )
        start += 1
        logger.info(f"Resumed VAE-GAN from epoch {start - 1}, best_val={best:.6f}")

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start, args.num_epochs):
        s = train_epoch(
            vae, disc, opt_gen, opt_disc, train_loader, epoch, args, writer, device,
            ema=ema,
        )

        # Validate live and (if available) EMA weights — log both.
        va_live = validate(vae, val_loader, args, device)
        va_ema = None
        if ema is not None and epoch >= args.ema_start_epoch:
            va_ema = validate(ema.shadow, val_loader, args, device)
            writer.add_scalar("val/loss_ema", va_ema, epoch)
        writer.add_scalar("val/loss", va_live, epoch)
        writer.add_scalar("lr/gen", opt_gen.param_groups[0]["lr"], epoch)
        sch_gen.step()

        # Use EMA val loss for best-checkpoint selection when available
        # (EMA generally outperforms live weights once warmup is over).
        va = va_ema if va_ema is not None else va_live

        use_adv = epoch >= args.disc_warmup_epochs
        ema_str = f"  val_ema={va_ema:.6f}" if va_ema is not None else ""
        fm_str = f"  fm={s['fm']:.6f}" if s["fm"] > 0 else ""
        logger.info(
            f"Epoch {epoch:3d} | "
            f"gen={s['gen']:.6f}  disc={s['disc']:.6f}  "
            f"cd={s['chamfer']:.6f}  kl={s['kl']:.4f}  "
            f"adv={s['adv']:.6f}{fm_str}  "
            f"val={va_live:.6f}{ema_str}  "
            f"adv_mode={'ON' if use_adv else 'OFF'}"
        )

        if va < best:
            best = va
            path = os.path.join(args.output_dir, "best_vae_gan.pth")
            save_vaegan_checkpoint(
                path, vae, disc, opt_gen, opt_disc, epoch, best, ema=ema,
            )
            logger.info(f"  → saved best checkpoint (val={best:.6f})")

        if (epoch + 1) % args.save_freq == 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            save_vaegan_checkpoint(
                path, vae, disc, opt_gen, opt_disc, epoch, best, ema=ema,
            )

    writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
