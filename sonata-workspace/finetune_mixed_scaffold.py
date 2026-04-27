#!/usr/bin/env python3
"""
Deployment-scaffold fine-tune.

Loads existing diffusion_v2gt teacher checkpoint and continues training with
mixed scaffolds (GT + LIDAR-cropped-to-GT-bbox) using x0-Chamfer loss against GT.
Saves to a NEW checkpoint path; original is untouched.

Goal: teach the denoiser to map LIDAR-style scaffolds back to GT geometry,
restoring competitive scaffold-free CD on stride-80 paired evaluation.
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

WORK = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
sys.path.insert(0, WORK)
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate

PREVOX = Path("/home/anywherevla/sonata_ws/prevoxelized_seq08")
CKPT_IN = Path(f"{WORK}/checkpoints/diffusion_v2gt/best_model.pth")
CKPT_DIR = Path(f"{WORK}/checkpoints/diffusion_v2gt_finetune_mixed_scaffold")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = Path(f"{WORK}/finetune_mixed_scaffold.log")

SEED = 42
DEVICE = "cuda"
EPOCHS = 3
LR = 5e-5
P_LIDAR = 0.30           # 30% LIDAR scaffold, 70% GT (keep most GT performance)
T_RANGE = (50, 400)      # focus on inference timesteps
GRAD_CLIP = 1.0
LOG_EVERY = 50
SAMPLE_PTS = 5000        # Chamfer sub-sample (M x K cdist bound)

torch.manual_seed(SEED); np.random.seed(SEED)

# ---- model ----
encoder = SonataEncoder(pretrained="facebook/sonata", freeze=True,
                        enable_flash=False, feature_levels=[0])
cond_extractor = ConditionalFeatureExtractor(encoder, feature_levels=[0],
                                             fusion_type="concat")
model = SceneCompletionDiffusion(encoder=encoder,
                                 condition_extractor=cond_extractor,
                                 num_timesteps=1000, schedule="cosine",
                                 denoising_steps=50)
model = model.to(DEVICE)

ckpt = torch.load(CKPT_IN, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
start_epoch = ckpt.get("epoch", 0)
print(f"[init] loaded {CKPT_IN} (epoch {start_epoch})")

# Freeze encoder explicitly
for p in model.encoder.parameters():
    p.requires_grad = False

trainable = [p for p in model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable)
print(f"[init] trainable params: {n_trainable:,}")

optimizer = torch.optim.AdamW(trainable, lr=LR)

# ---- data ----
all_files = sorted(PREVOX.glob("*.npz"))
train_files = all_files[:3000]
val_files = all_files[3000:4070]
print(f"[data] train={len(train_files)} val={len(val_files)}")

def load_frame(p):
    d = np.load(p)
    return (d["lidar_coords"].astype(np.float32),
            d["gt_coords_lidar"].astype(np.float32),
            d["lidar_center"].astype(np.float32),
            d["gt_raw"].astype(np.float32))

def make_pd(pts, device=DEVICE):
    z = pts[:, 2]
    zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
    cols = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1).astype(np.float32)
    return {
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(cols).to(device),
        "normal": torch.zeros(pts.shape[0], 3, dtype=torch.float32, device=device),
        "batch": torch.zeros(pts.shape[0], dtype=torch.long, device=device),
    }

def chamfer(a, b, max_pts=SAMPLE_PTS):
    """Bidirectional mean L2 chamfer on subsampled points (differentiable)."""
    if a.shape[0] > max_pts:
        a = a[torch.randperm(a.shape[0], device=a.device)[:max_pts]]
    if b.shape[0] > max_pts:
        b = b[torch.randperm(b.shape[0], device=b.device)[:max_pts]]
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)  # M x K
    a_to_b = d.min(dim=1)[0].mean()
    b_to_a = d.min(dim=0)[0].mean()
    return a_to_b + b_to_a, a_to_b.item(), b_to_a.item()

def build_scaffold(lidar, gt, mode):
    """Returns Nx3 numpy scaffold."""
    if mode == "gt":
        s = gt.copy()
    elif mode == "lidar_gt_bbox":
        mn, mx = gt.min(0), gt.max(0)
        m = ((lidar >= mn) & (lidar <= mx)).all(1)
        s = lidar[m].copy()
    else:
        raise ValueError(mode)
    if len(s) < 64:
        return None
    if len(s) > 20000:
        s = s[np.random.choice(len(s), 20000, replace=False)]
    return s.astype(np.float32)

def step(lidar, gt, t_val, scaffold_mode, train=True):
    """One forward step. Returns loss (or float)."""
    scaffold = build_scaffold(lidar, gt, scaffold_mode)
    if scaffold is None:
        return None

    pd = make_pd(lidar)
    scaffold_t = torch.from_numpy(scaffold).to(DEVICE)
    gt_t = torch.from_numpy(gt).to(DEVICE)

    # Encoder features (no grad)
    with torch.no_grad():
        feats, _ = model.condition_extractor(pd)
    cond_feat_s = knn_interpolate(feats, pd["coord"], scaffold_t)

    t_tensor = torch.full((1,), t_val, device=DEVICE)
    noise = torch.randn_like(scaffold_t)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val].to(DEVICE)
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val].to(DEVICE)
    noisy = sa * scaffold_t + som * noise

    pred_noise = model.denoiser(noisy, scaffold_t, t_tensor,
                                {"features": cond_feat_s})
    pred_x0 = (noisy - som * pred_noise) / sa

    loss, p2g, g2p = chamfer(pred_x0, gt_t)
    return loss, p2g, g2p, pred_x0.detach()

# ---- training loop ----
def train_epoch(ep):
    model.train()
    model.encoder.eval()
    np.random.shuffle(train_files)
    losses = []; p2gs = []; g2ps = []; mode_counts = {"gt":0, "lidar_gt_bbox":0}
    t0 = time.time()
    for i, f in enumerate(train_files):
        try:
            lidar, gt, _, _ = load_frame(f)
        except Exception as e:
            print(f"  skip {f.stem}: {e}"); continue

        mode = "lidar_gt_bbox" if np.random.random() < P_LIDAR else "gt"
        t_val = int(np.random.randint(*T_RANGE))

        out = step(lidar, gt, t_val, mode, train=True)
        if out is None:
            continue
        loss, p2g, g2p, _ = out
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()

        losses.append(loss.item()); p2gs.append(p2g); g2ps.append(g2p)
        mode_counts[mode] += 1

        if (i + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i+1)/elapsed
            eta = (len(train_files)-i-1)/rate
            msg = (f"  [E{ep} {i+1}/{len(train_files)}] "
                   f"loss={np.mean(losses[-LOG_EVERY:]):.4f} "
                   f"p2g={np.mean(p2gs[-LOG_EVERY:]):.3f} "
                   f"g2p={np.mean(g2ps[-LOG_EVERY:]):.3f} "
                   f"rate={rate:.1f}/s eta={eta/60:.1f}min "
                   f"gt:{mode_counts['gt']} lidar:{mode_counts['lidar_gt_bbox']}")
            print(msg, flush=True)
            with open(LOG_PATH, "a") as lf: lf.write(msg+"\n")
    return float(np.mean(losses)) if losses else float("nan")

def validate(ep):
    model.eval()
    losses_gt = []; losses_lidar = []
    with torch.no_grad():
        for i, f in enumerate(val_files[::5]):  # subsample val for speed (~214 frames)
            try:
                lidar, gt, _, _ = load_frame(f)
            except Exception:
                continue
            for mode in ("gt", "lidar_gt_bbox"):
                out = step(lidar, gt, 200, mode, train=False)
                if out is None: continue
                loss, p2g, g2p, _ = out
                if mode == "gt":
                    losses_gt.append(loss.item())
                else:
                    losses_lidar.append(loss.item())
    msg = (f"  [val E{ep}] gt-scaffold loss={np.mean(losses_gt):.4f} "
           f"lidar-scaffold loss={np.mean(losses_lidar):.4f}")
    print(msg, flush=True)
    with open(LOG_PATH, "a") as lf: lf.write(msg+"\n")
    return np.mean(losses_lidar)

with open(LOG_PATH, "w") as lf:
    lf.write(f"[init] {CKPT_IN} epoch {start_epoch}\n")

print(f"[init] starting fine-tune for {EPOCHS} epochs, P_LIDAR={P_LIDAR}, lr={LR}")
best_lidar_loss = float("inf")
for ep in range(EPOCHS):
    t_ep = time.time()
    train_loss = train_epoch(ep)
    val_lidar = validate(ep)
    elapsed = time.time() - t_ep
    print(f"\n[epoch {ep} done in {elapsed/60:.1f}min] train_loss={train_loss:.4f} "
          f"val_lidar={val_lidar:.4f}", flush=True)

    # Save every epoch
    state = {"epoch": start_epoch + ep + 1, "model_state_dict": model.state_dict(),
             "train_loss": train_loss, "val_lidar_loss": val_lidar}
    torch.save(state, CKPT_DIR / f"epoch_{ep}.pth")
    if val_lidar < best_lidar_loss:
        best_lidar_loss = val_lidar
        torch.save(state, CKPT_DIR / "best.pth")
        print(f"  [save] new best val_lidar={val_lidar:.4f} -> best.pth", flush=True)

print(f"\n[done] best val_lidar={best_lidar_loss:.4f}")
