"""Quick scaffold-protocol sweep on stride-80 50 fr to find a viable
scaffold-free protocol that does not require GT points."""
import sys, os, time, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, ".")
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from run_scaffoldfree_fair import build_model, load_ckpt, make_point_dict, run_x0_single_step, compute_cd_torch, compute_cd_lidiff_kdtree, cd_with_crop, EGO_BBOX_MIN, EGO_BBOX_MAX, LIDIFF_MARGIN

PREVOX = Path("/home/anywherevla/sonata_ws/prevoxelized_seq08")
CKPT = Path("checkpoints/diffusion_v2gt/best_model.pth")
SEED=42
np.random.seed(SEED); torch.manual_seed(SEED)

# Stride-80 50 frames matching LiDiff/ScoreLiDAR eval
files = sorted(PREVOX.glob("*.npz"))[::80][:50]
print(f"frames {len(files)}: {files[0].stem}..{files[-1].stem}")

frames = []
for f in files:
    d = np.load(f)
    frames.append({"name": f.stem, "lidar_coords": d["lidar_coords"],
                   "lidar_center": d["lidar_center"], "gt_coords_lidar": d["gt_coords_lidar"],
                   "gt_raw": d["gt_raw"]})
print(f"loaded {len(frames)} frames")

model = build_model()
model = load_ckpt(model, CKPT)
model.eval()

def scaffold_var(fr, name):
    lidar = fr["lidar_coords"]
    gt_l = fr["gt_coords_lidar"]
    if name == "ego_bbox_pm15":
        bmn = np.array([-15,-15,-3], dtype=np.float32); bmx = np.array([15,15,3], dtype=np.float32)
        c = lidar[((lidar>=bmn)&(lidar<=bmx)).all(1)]
    elif name == "ego_bbox_pm20":
        bmn = np.array([-20,-20,-3], dtype=np.float32); bmx = np.array([20,20,3], dtype=np.float32)
        c = lidar[((lidar>=bmn)&(lidar<=bmx)).all(1)]
    elif name == "frustum_30m":
        bmn = np.array([0,-15,-3], dtype=np.float32); bmx = np.array([30,15,3], dtype=np.float32)
        c = lidar[((lidar>=bmn)&(lidar<=bmx)).all(1)]
    elif name == "gt_bbox_lidar":
        # use GT bbox to crop raw lidar (uses GT bbox info, not GT coords)
        mn,mx = gt_l.min(0), gt_l.max(0)
        c = lidar[((lidar>=mn)&(lidar<=mx)).all(1)]
    elif name == "gt_bbox_lidar_dense":
        # GT bbox crop + 5x duplication with 0.10m jitter
        mn,mx = gt_l.min(0), gt_l.max(0)
        crop = lidar[((lidar>=mn)&(lidar<=mx)).all(1)]
        if len(crop)<10: return None
        dups=[crop]+[crop+np.random.normal(0,0.10,crop.shape).astype(np.float32) for _ in range(4)]
        c=np.concatenate(dups)
    elif name == "voxel0.10m_ego_pm20":
        # Voxelize at 0.10m to match GT density
        bmn = np.array([-20,-20,-3], dtype=np.float32); bmx = np.array([20,20,3], dtype=np.float32)
        c = lidar[((lidar>=bmn)&(lidar<=bmx)).all(1)]
        vc = np.floor(c/0.10).astype(np.int32)
        _,idx = np.unique(vc, axis=0, return_index=True)
        c = c[idx]
    else: return None
    if len(c)<64: return None
    if len(c)>20000:
        c = c[np.random.choice(len(c),20000,replace=False)]
    return c.astype(np.float32)

variants = ["ego_bbox_pm15","ego_bbox_pm20","frustum_30m","gt_bbox_lidar","gt_bbox_lidar_dense","voxel0.10m_ego_pm20"]
results = {}
for vn in variants:
    cds = []
    for i,fr in enumerate(frames):
        scaffold = scaffold_var(fr, vn)
        if scaffold is None: continue
        pd = make_point_dict(fr["lidar_coords"])
        target = torch.from_numpy(scaffold).float().cuda()
        with torch.no_grad():
            pred = run_x0_single_step(model, pd, target, t_val=200).cpu().numpy()
        pred_world = pred + fr["lidar_center"]
        cd, _ = cd_with_crop(pred_world, fr["gt_raw"], "lidiff", "torch")
        if np.isfinite(cd): cds.append(cd)
    if cds:
        results[vn] = {"mean": float(np.mean(cds)), "std": float(np.std(cds)), "n": len(cds), "median": float(np.median(cds))}
        print(f"{vn:35s}: mean={np.mean(cds):.3f}+/-{np.std(cds):.3f} n={len(cds)} med={np.median(cds):.3f}")
    else:
        results[vn] = {"error": "no finite CDs"}
        print(f"{vn}: error")

with open("scaffold_sweep_str80_results.json","w") as f:
    json.dump(results, f, indent=2)
print("done")
