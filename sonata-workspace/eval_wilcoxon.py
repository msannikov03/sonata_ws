"""Per-frame paired CD-squared from fine-tuned scaffold-free model."""
import sys, os, json
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from run_scaffoldfree_fair import (
    build_model, load_ckpt, make_point_dict, run_x0_single_step,
    cd_with_crop, EGO_BBOX_MIN, EGO_BBOX_MAX,
)
import torch

PREVOX = Path("/home/anywherevla/sonata_ws/prevoxelized_seq08")
CKPT = Path("checkpoints/diffusion_v2gt_finetune_mixed_scaffold/best.pth")
np.random.seed(42); torch.manual_seed(42)

model = build_model()
model = load_ckpt(model, CKPT)
model.eval()

files = sorted(PREVOX.glob("*.npz"))[::80][:50]
out = {"frame_id": [], "ours_ft_cd_sq": []}

for f in files:
    d = np.load(f)
    lidar = d["lidar_coords"].astype(np.float32)
    gt_raw = d["gt_raw"].astype(np.float32)
    center = d["lidar_center"].astype(np.float32)

    dups = [lidar] + [lidar + np.random.normal(0, 0.05, lidar.shape).astype(np.float32) for _ in range(9)]
    cloud = np.concatenate(dups)
    mask = ((cloud >= EGO_BBOX_MIN) & (cloud <= EGO_BBOX_MAX)).all(axis=1)
    scaffold = cloud[mask]
    if len(scaffold) > 20000:
        scaffold = scaffold[np.random.choice(len(scaffold), 20000, replace=False)]

    pd = make_point_dict(lidar)
    target = torch.from_numpy(scaffold).float().cuda()
    pred = run_x0_single_step(model, pd, target, t_val=200).cpu().numpy()
    pred_world = pred + center
    cd, _ = cd_with_crop(pred_world, gt_raw, "lidiff", "kdtree")
    out["frame_id"].append(f.stem)
    out["ours_ft_cd_sq"].append(float(cd))
    print(f.stem, "cd", round(cd, 4))

with open("eval_wilcoxon_finetuned.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("saved n =", len(out["ours_ft_cd_sq"]))
