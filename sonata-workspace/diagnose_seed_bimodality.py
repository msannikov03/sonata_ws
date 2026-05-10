#!/usr/bin/env python3
"""
Diagnose seed bimodality: eval all 18 checkpoints (6 seeds × 3 epochs) on:
  - HELD-OUT validation set (stride-80, offset 40 — disjoint from original test set)
  - ORIGINAL test set (stride-80, offset 0 — the paper's headline 50 frames)

For each seed, find the validation-best epoch (legitimate selection criterion),
then report the test CD for that epoch. Also report the test CD for all epochs.

This is ETHICALLY CLEAN: epoch selection uses ONLY val set, never test set.
"""
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
    return {
        'coord': torch.from_numpy(pts).float().to(DEVICE),
        'color': torch.from_numpy(colors).float().to(DEVICE),
        'normal': torch.zeros(pts.shape[0], 3, dtype=torch.float32, device=DEVICE),
        'batch': torch.zeros(pts.shape[0], dtype=torch.long, device=DEVICE),
    }

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

def load_frames(stride=80, offset=0, n=50):
    all_files = sorted(PREVOX_DIR.glob('*.npz'))
    files = all_files[offset::stride][:n]
    frames = []
    for f in files:
        d = np.load(f)
        frames.append({'name': f.stem, 'lidar_coords': d['lidar_coords'],
                       'lidar_center': d['lidar_center'], 'gt_raw': d['gt_raw']})
    return frames

def scaffold_ego_bbox(fr, n_dup=10, jitter=0.05, max_pts=MAX_PTS):
    lidar = fr['lidar_coords']
    dups = [lidar]
    for i in range(1, n_dup):
        dups.append(lidar + np.random.normal(0.0, jitter, size=lidar.shape).astype(np.float32))
    cloud = np.concatenate(dups, axis=0)
    mask = np.all((cloud >= EGO_BBOX_MIN) & (cloud <= EGO_BBOX_MAX), axis=1)
    cloud = cloud[mask]
    if cloud.shape[0] > max_pts:
        cloud = cloud[np.random.choice(cloud.shape[0], max_pts, replace=False)]
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

def eval_ckpt_on_frames(model, frames):
    cds = []
    for fr in frames:
        np.random.seed(SEED); torch.manual_seed(SEED)
        scaffold = scaffold_ego_bbox(fr)
        if scaffold.shape[0] < 64: continue
        point_dict = make_point_dict(fr['lidar_coords'])
        target = torch.from_numpy(scaffold).float().to(DEVICE)
        try:
            pred = run_x0_single_step(model, point_dict, target).cpu().numpy()
        except Exception:
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
    np.random.seed(SEED); torch.manual_seed(SEED)
    print('Loading frames...')
    test_frames = load_frames(stride=80, offset=0, n=50)
    val_frames = load_frames(stride=80, offset=40, n=50)
    print(f'Test frames: {len(test_frames)} (offset=0)')
    print(f'Val frames:  {len(val_frames)} (offset=40, disjoint)')
    test_names = set(f['name'] for f in test_frames)
    val_names = set(f['name'] for f in val_frames)
    overlap = test_names & val_names
    print(f'Overlap: {len(overlap)} frames (should be 0)')

    print('Building model (shared across all checkpoint loads)...')
    model = build_model()

    seeds = [42, 43, 44, 45, 46, 47]
    epochs = [0, 1, 2]
    results = {}
    for s in seeds:
        d = 'diffusion_v2gt_finetune_mixed_scaffold' if s == 42 else f'diffusion_v2gt_finetune_mixed_scaffold_seed{s}'
        results[s] = {}
        for ep in epochs:
            ckpt_path = WORK / 'checkpoints' / d / f'epoch_{ep}.pth'
            print(f'\n[seed {s} epoch {ep}] {ckpt_path.name}')
            t0 = time.time()
            load_ckpt(model, str(ckpt_path))
            test_mean, test_std, test_n = eval_ckpt_on_frames(model, test_frames)
            val_mean, val_std, val_n = eval_ckpt_on_frames(model, val_frames)
            results[s][ep] = {'test_cd': test_mean, 'test_std': test_std, 'test_n': test_n,
                              'val_cd': val_mean, 'val_std': val_std, 'val_n': val_n}
            print(f'  test={test_mean:.4f} (n={test_n}), val={val_mean:.4f} (n={val_n})  [{time.time()-t0:.1f}s]')

    # Save raw
    with open(WORK / 'diagnose_seed_bimodality.json', 'w') as f:
        json.dump({'results': {str(s): {str(e): v for e, v in eps.items()} for s, eps in results.items()}}, f, indent=2)

    # Summary
    print('\n' + '='*80)
    print('PER-SEED PER-EPOCH RESULTS (test = stride-80 offset-0 = paper headline)')
    print('='*80)
    print(f'{"seed":>5} {"ep0_val":>8} {"ep0_test":>9} {"ep1_val":>8} {"ep1_test":>9} {"ep2_val":>8} {"ep2_test":>9} {"best_ep_by_val":>14} {"test_at_best_val":>16}')
    headline_old_3seed = []  # seeds 42, 43, 44 best.pth result (paper)
    headline_legit_6seed = []  # all 6 seeds, best epoch by VAL
    for s in seeds:
        row = [results[s][ep] for ep in epochs]
        best_val_ep = min(epochs, key=lambda e: results[s][e]['val_cd'])
        test_at_best_val = results[s][best_val_ep]['test_cd']
        headline_legit_6seed.append(test_at_best_val)
        print(f'{s:>5} {row[0]["val_cd"]:>8.4f} {row[0]["test_cd"]:>9.4f} {row[1]["val_cd"]:>8.4f} {row[1]["test_cd"]:>9.4f} {row[2]["val_cd"]:>8.4f} {row[2]["test_cd"]:>9.4f} {best_val_ep:>14} {test_at_best_val:>16.4f}')

    import statistics as stats
    print('\n' + '='*80)
    print('HEADLINE COMPARISONS')
    print('='*80)
    # 3-seed paper headline (best.pth, seeds 42/43/44)
    paper_3seed = [0.7271, 1.1445, 0.7547]
    print(f'Paper 3-seed (best.pth, seeds 42/43/44): mean={stats.mean(paper_3seed):.4f}, std={stats.stdev(paper_3seed):.4f}')
    # 6-seed naive (best.pth, all seeds)
    paper_6seed = [0.7271, 1.1445, 0.7547, 1.2338, 1.1050, 1.2081]
    print(f'Naive 6-seed (best.pth):                  mean={stats.mean(paper_6seed):.4f}, std={stats.stdev(paper_6seed):.4f}')
    # 6-seed legitimate selection by val
    print(f'Legit 6-seed (best epoch by VAL):         mean={stats.mean(headline_legit_6seed):.4f}, std={stats.stdev(headline_legit_6seed):.4f}')
    print(f'  per-seed test results: {[round(x,4) for x in headline_legit_6seed]}')

if __name__ == '__main__':
    main()
