#!/usr/bin/env python3
"""
N-Seed Ensemble — Step 3 of apr17_morning experiment queue.

For each frame, run N completions with different torch seeds, average the
predictions coordinate-wise, and evaluate CD of the average.

Compare N = 1, 2, 4, 8 for both LiDAR and DA2 inputs on 100 frames of seq 08.

Since all completions share the same GT target coords (scaffold), point ordering
is aligned — we can just average coord-wise.

Results -> results/apr17_morning/n_seed_ensemble.json
"""

import os, sys, time, json
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np

WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR))

from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance

# ---- config ----
PREVOX_DIR = Path("/home/anywherevla/sonata_ws/prevoxelized_seq08")
CKPT_PATH = WORK_DIR / "checkpoints" / "diffusion_v2gt" / "best_model.pth"
RESULTS_DIR = WORK_DIR / "results" / "apr17_morning"
RESULTS_PATH = RESULTS_DIR / "n_seed_ensemble.json"
NUM_FRAMES = 100
DEVICE = "cuda"
N_VALUES = [1, 2, 4, 8]
MAX_N = max(N_VALUES)
BASE_SEED = 42


def build_model(device=DEVICE):
    encoder = SonataEncoder(
        pretrained="facebook/sonata", freeze=True,
        enable_flash=False, feature_levels=[0],
    )
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    model = SceneCompletionDiffusion(
        encoder=encoder, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50,
    )
    return model.to(device)


def load_ckpt(model, path, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[ckpt] loaded {path}  (epoch {ckpt.get('epoch', '?')})")
    return model


def make_point_dict(coords, device=DEVICE):
    pts = coords if isinstance(coords, np.ndarray) else coords.cpu().numpy()
    z = pts[:, 2]
    zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1)
    return {
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(pts.shape[0], 3, dtype=torch.float32, device=device),
        "batch": torch.zeros(pts.shape[0], dtype=torch.long, device=device),
    }


def compute_cd(pred, gt, max_pts=10000):
    if pred.shape[0] > max_pts:
        pred = pred[np.random.choice(pred.shape[0], max_pts, replace=False)]
    if gt.shape[0] > max_pts:
        gt = gt[np.random.choice(gt.shape[0], max_pts, replace=False)]
    cd = chamfer_distance(
        torch.from_numpy(pred).float().cuda(),
        torch.from_numpy(gt).float().cuda(),
        chunk_size=512,
    )
    return cd.item()


def load_frames(n=NUM_FRAMES):
    files = sorted(PREVOX_DIR.glob("*.npz"))[:n]
    frames = []
    for f in files:
        d = np.load(f)
        frames.append({
            "name": f.stem,
            "lidar_coords": d["lidar_coords"],
            "lidar_center": d["lidar_center"],
            "da2_coords": d["da2_coords"],
            "da2_center": d["da2_center"],
            "gt_coords_lidar": d["gt_coords_lidar"],
            "gt_coords_da2": d["gt_coords_da2"],
            "gt_raw": d["gt_raw"],
        })
    print(f"[data] loaded {len(frames)} frames from {PREVOX_DIR}")
    return frames


@torch.no_grad()
def get_cond_features(model, point_dict, target_coords):
    model.eval()
    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], target_coords)
    return cond_features


@torch.no_grad()
def run_x0_with_noise(model, target_coords, cond_features, noise, t_val=200):
    device = target_coords.device
    model.scheduler._to_device(device)
    t_tensor = torch.full((1,), t_val, device=device)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]
    noisy = sa * target_coords + som * noise
    pred_noise = model.denoiser(noisy, target_coords, t_tensor, {"features": cond_features})
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0


def process_modality(model, frames, modality):
    """Run N-seed ensemble for either 'lidar' or 'da2' input."""
    print(f"\n{'='*70}\n  Modality: {modality}\n{'='*70}")
    t0 = time.time()

    per_frame_results = []
    # accumulate CDs for each N (ensemble size)
    cds_by_n = {n: [] for n in N_VALUES}
    # per-run CDs for variance estimates
    cds_single_runs = []  # list of per-frame mean CD across MAX_N runs
    cds_single_stds = []

    for fi, fr in enumerate(frames):
        if modality == "lidar":
            input_coords = fr["lidar_coords"]
            input_center = fr["lidar_center"]
            gt_target = torch.from_numpy(fr["gt_coords_lidar"]).float().to(DEVICE)
        else:  # da2
            input_coords = fr["da2_coords"]
            input_center = fr["da2_center"]
            gt_target = torch.from_numpy(fr["gt_coords_da2"]).float().to(DEVICE)

        point_dict = make_point_dict(input_coords)

        # cache conditioning once per frame
        cond = get_cond_features(model, point_dict, gt_target)

        # collect MAX_N completions
        preds = []
        for k in range(MAX_N):
            seed = BASE_SEED + 1000 * fi + k
            torch.manual_seed(seed)
            noise = torch.randn_like(gt_target)
            pred = run_x0_with_noise(model, gt_target, cond, noise)
            preds.append(pred.cpu().numpy())

        preds_stack = np.stack(preds, axis=0)  # (MAX_N, N_pts, 3) centered
        # per-point std across the MAX_N runs, norm = uncertainty
        per_point_std = np.linalg.norm(preds_stack.std(axis=0), axis=1).mean()

        # per-run CDs (no ensembling)
        per_run_cds = []
        for k in range(MAX_N):
            pw = preds_stack[k] + input_center
            per_run_cds.append(compute_cd(pw, fr["gt_raw"]))
        cds_single_runs.append(float(np.mean(per_run_cds)))
        cds_single_stds.append(float(np.std(per_run_cds)))

        # CD for each ensemble size: average the first n completions
        row = {"frame": fr["name"], "per_run_cd_mean": float(np.mean(per_run_cds)),
               "per_run_cd_std": float(np.std(per_run_cds)),
               "uncertainty_mean_std": float(per_point_std)}
        for n in N_VALUES:
            avg_pred = preds_stack[:n].mean(axis=0) + input_center
            cd = compute_cd(avg_pred, fr["gt_raw"])
            cds_by_n[n].append(cd)
            row[f"cd_n{n}"] = float(cd)

        per_frame_results.append(row)

        if (fi + 1) % 10 == 0:
            line = f"  [{fi+1}/{len(frames)}]"
            for n in N_VALUES:
                line += f"  n={n}: {np.mean(cds_by_n[n]):.4f}"
            print(line)

    elapsed = time.time() - t0
    result = {
        "modality": modality,
        "n_frames": len(frames),
        "time_s": round(elapsed, 1),
        "per_run_cd_mean": float(np.mean(cds_single_runs)),
        "per_run_cd_std_across_frames": float(np.std(cds_single_runs)),
        "per_run_cd_std_within_frame_mean": float(np.mean(cds_single_stds)),
    }
    for n in N_VALUES:
        result[f"n{n}_cd_mean"] = float(np.mean(cds_by_n[n]))
        result[f"n{n}_cd_std"] = float(np.std(cds_by_n[n]))

    # summary
    print(f"\n  === {modality} summary ===")
    for n in N_VALUES:
        print(f"   N={n}  CD = {result[f'n{n}_cd_mean']:.4f} +/- {result[f'n{n}_cd_std']:.4f}")
    print(f"   Per-run CD (no ensemble): {result['per_run_cd_mean']:.4f}")
    print(f"   Time: {elapsed:.1f}s")

    result["per_frame"] = per_frame_results
    return result


def main():
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  N-SEED ENSEMBLE")
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Frames: {NUM_FRAMES} from seq 08 (prevoxelized)")
    print(f"  N values: {N_VALUES}")
    print(f"  Results: {RESULTS_PATH}")
    print("=" * 70)

    model = load_ckpt(build_model(DEVICE), str(CKPT_PATH), DEVICE)
    model.eval()
    frames = load_frames(NUM_FRAMES)

    all_results = OrderedDict()
    total_t0 = time.time()

    for modality in ["lidar", "da2"]:
        try:
            all_results[modality] = process_modality(model, frames, modality)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[modality] = {"error": str(e)}
        # save incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "results": all_results,
                "num_frames": len(frames),
                "checkpoint": str(CKPT_PATH),
                "n_values": N_VALUES,
                "base_seed": BASE_SEED,
                "partial": True,
            }, f, indent=2)

    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Modality':<10} " + " ".join(f"N={n:<3}" for n in N_VALUES))
    for modality, r in all_results.items():
        if "error" in r:
            print(f"{modality:<10} ERROR")
            continue
        line = f"{modality:<10} "
        for n in N_VALUES:
            line += f"{r[f'n{n}_cd_mean']:.4f} "
        print(line)

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "results": all_results,
            "num_frames": len(frames),
            "checkpoint": str(CKPT_PATH),
            "n_values": N_VALUES,
            "base_seed": BASE_SEED,
            "total_time_s": round(total_elapsed, 1),
            "partial": False,
        }, f, indent=2)
    print(f"\n[saved] {RESULTS_PATH}")


if __name__ == "__main__":
    main()
