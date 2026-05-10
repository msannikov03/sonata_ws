#!/usr/bin/env python3
"""Re-eval epoch_2.pth for all 6 seeds using the proper C2 protocol
(matching run_scaffoldfree_fair_finetuned*.py random-state setup).
Only the kdtree variant since that's the headline."""
import os, sys, json, time
from pathlib import Path
import torch
import numpy as np

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance

PREVOX_DIR = Path('/home/anywherevla/sonata_ws/prevoxelized_seq08')
DEVICE = 'cuda'
MAX_PTS = 20000
SEED = 42
EGO_BBOX_MIN = np.array([-40.0, -40.0, -5.0], dtype=np.float32)
EGO_BBOX_MAX = np.array([ 40.0,  40.0,  5.0], dtype=np.float32)
LIDIFF_MARGIN = 1.0

def build_model():
    encoder = SonataEncoder(pretrained='facebook/sonata', freeze=True, enable_flash=False, feature_levels=[0])
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type='concat')
    model = SceneCompletionDiffusion(encoder=encoder, condition_extractor=cond,
                                      num_timesteps=1000, schedule='cosine', denoising_steps=50)
    return model.to(DEVICE)

def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

def make_point_dict(coords):
    pts = coords if isinstance(coords, np.ndarray) else coords.cpu().numpy()
    z = pts[:, 2]
    zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1)
    return {'coord': torch.from_numpy(pts).float().to(DEVICE),
            'color': torch.from_numpy(colors).float().to(DEVICE),
            'normal': torch.zeros(pts.shape[0], 3, dtype=torch.float32, device=DEVICE),
            'batch': torch.zeros(pts.shape[0], dtype=torch.long, device=DEVICE)}

def compute_cd_lidiff_kdtree(pred, gt, max_pts=60000):
    from scipy.spatial import cKDTree
    a = np.asarray(pred, dtype=np.float32)
    b = np.asarray(gt, dtype=np.float32)
    if a.shape[0] == 0 or b.shape[0] == 0: return float('nan')
    if a.shape[0] > max_pts:
        a = a[np.random.default_rng(0).choice(a.shape[0], max_pts, replace=False)]
    if b.shape[0] > max_pts:
        b = b[np.random.default_rng(0).choice(b.shape[0], max_pts, replace=False)]
    tree_b = cKDTree(b); d_ab,_ = tree_b.query(a, k=1)
    tree_a = cKDTree(a); d_ba,_ = tree_a.query(b, k=1)
    return 0.5 * (float(np.mean(d_ab**2)) + float(np.mean(d_ba**2)))

def load_frames(n=50):
    all_files = sorted(PREVOX_DIR.glob('*.npz')); files = all_files[::80][:n]
    frames = []
    for f in files:
        d = np.load(f)
        frames.append({'name': f.stem, 'lidar_coords': d['lidar_coords'],
                       'lidar_center': d['lidar_center'], 'gt_raw': d['gt_raw']})
    return frames

def scaffold_ego_bbox(fr, n_dup=10, jitter=0.05):
    lidar = fr['lidar_coords']
    dups = [lidar]
    for i in range(1, n_dup):
        dups.append(lidar + np.random.normal(0.0, jitter, size=lidar.shape).astype(np.float32))
    cloud = np.concatenate(dups, axis=0)
    mask = np.all((cloud >= EGO_BBOX_MIN) & (cloud <= EGO_BBOX_MAX), axis=1)
    cloud = cloud[mask]
    if cloud.shape[0] > MAX_PTS:
        cloud = cloud[np.random.choice(cloud.shape[0], MAX_PTS, replace=False)]
    return cloud.astype(np.float32)

@torch.no_grad()
def run_x0_single_step(model, point_dict, target_coords, t_val=200):
    model.eval()
    model.scheduler._to_device(target_coords.device)
    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict['coord'], target_coords)
    t_tensor = torch.full((1,), t_val, device=target_coords.device)
    noise = torch.randn_like(target_coords)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * target_coords + som * noise
    pred_noise = model.denoiser(noisy, target_coords, t_tensor, {'features': cond_features})
    return (noisy - som * pred_noise) / sa

def eval_kdtree_variant(model, frames):
    """Replicates A_fair_lidiff_match_kdtree from run_scaffoldfree_fair_finetuned.py.
    Per-frame seed handled by the variant-level np.random.seed(SEED) at start."""
    cds = []
    for fr in frames:
        scaffold = scaffold_ego_bbox(fr)
        if scaffold.shape[0] < 64: continue
        point_dict = make_point_dict(fr['lidar_coords'])
        target = torch.from_numpy(scaffold).float().to(DEVICE)
        try:
            pred = run_x0_single_step(model, point_dict, target).cpu().numpy()
        except Exception as e:
            continue
        pred_world = pred + fr['lidar_center']
        gt_raw = fr['gt_raw']
        bbox_min = gt_raw.min(axis=0) - LIDIFF_MARGIN
        bbox_max = gt_raw.max(axis=0) + LIDIFF_MARGIN
        mask = np.all((pred_world >= bbox_min) & (pred_world <= bbox_max), axis=1)
        cd = compute_cd_lidiff_kdtree(pred_world[mask], gt_raw)
        if np.isfinite(cd): cds.append(cd)
    return float(np.mean(cds)), float(np.std(cds)), len(cds)

def main():
    print('Loading frames...')
    frames = load_frames()
    print(f'{len(frames)} frames')
    model = build_model()

    seeds = [42, 43, 44, 45, 46, 47]
    ep2_results = {}
    for s in seeds:
        d = 'diffusion_v2gt_finetune_mixed_scaffold' if s == 42 else f'diffusion_v2gt_finetune_mixed_scaffold_seed{s}'
        ckpt_path = WORK / 'checkpoints' / d / 'epoch_2.pth'
        print(f'\n[seed {s}] loading {ckpt_path.name} from {d}...')
        load_ckpt(model, str(ckpt_path))

        # Match C2 protocol: seed BEFORE variant runs
        np.random.seed(SEED); torch.manual_seed(SEED)
        t0 = time.time()
        mean, std, n = eval_kdtree_variant(model, frames)
        ep2_results[s] = mean
        print(f'  seed {s} epoch_2 kdtree: mean={mean:.4f}, std={std:.4f}, n={n}  [{time.time()-t0:.1f}s]')

    # Save
    with open(WORK / 'eval_epoch2_recovery.json', 'w') as f:
        json.dump({str(k): v for k, v in ep2_results.items()}, f, indent=2)

    # Summary with the C2 best.pth numbers for direct comparison
    c2_best = {42: 0.7271, 43: 1.1445, 44: 0.7547, 45: 1.2338, 46: 1.1050, 47: 1.2081}
    print('\n' + '='*70)
    print('RECOVERY ATTEMPT: epoch_2 vs best.pth (C2 protocol)')
    print('='*70)
    print(f'{"seed":>5} {"best.pth (C2)":>15} {"epoch_2 (NEW)":>15} {"delta":>10}')
    for s in seeds:
        delta = ep2_results[s] - c2_best[s]
        print(f'{s:>5} {c2_best[s]:>15.4f} {ep2_results[s]:>15.4f} {delta:>+10.4f}')
    import statistics as stats
    bp = list(c2_best.values())
    e2 = [ep2_results[s] for s in seeds]
    print()
    print(f'best.pth 6-seed: mean={stats.mean(bp):.4f}, std={stats.stdev(bp):.4f}')
    print(f'epoch_2 6-seed:  mean={stats.mean(e2):.4f}, std={stats.stdev(e2):.4f}')
    print(f'\nPaper 3-seed (best.pth, 42/43/44): 0.8754 +/- 0.2334')
    print(f'NEW 3-seed (epoch_2, 42/43/44):    {stats.mean(e2[:3]):.4f} +/- {stats.stdev(e2[:3]):.4f}')

if __name__ == '__main__':
    main()
