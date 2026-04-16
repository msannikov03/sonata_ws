#!/usr/bin/env python3
"""
Uncertainty estimation via stochastic diffusion completions.

For each frame, runs N completions with different noise seeds at t=200,
then computes per-point spatial variance as an uncertainty measure.
Outputs metrics JSON and BEV uncertainty visualizations.
"""
import os, sys, torch, numpy as np, argparse, time, json
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion, knn_interpolate
from models.refinement_net import chamfer_distance


# -- Model setup (identical to evaluate_teacher_da2.py) --------------------

def build_model(device="cuda"):
    encoder = SonataEncoder(
        pretrained="facebook/sonata", freeze=True,
        enable_flash=False, feature_levels=[0]
    )
    cond = ConditionalFeatureExtractor(encoder, feature_levels=[0], fusion_type="concat")
    model = SceneCompletionDiffusion(
        encoder=encoder, condition_extractor=cond,
        num_timesteps=1000, schedule="cosine", denoising_steps=50
    )
    return model.to(device)


def load_ckpt(model, path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {path} (epoch {ckpt.get('epoch', '?')})")
    return model


# -- Data helpers ----------------------------------------------------------

def prepare_scan(pts_raw, device="cuda", max_points=20000, voxel_size=0.05):
    center = pts_raw.mean(axis=0)
    pts = pts_raw - center
    vc = np.floor(pts / voxel_size).astype(np.int32)
    _, idx = np.unique(vc, axis=0, return_index=True)
    pts = pts[idx]
    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]
    z = pts[:, 2]
    zn = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([zn, 1 - np.abs(zn - 0.5) * 2, 1 - zn], axis=1)
    return {
        "coord": torch.from_numpy(pts).float().to(device),
        "color": torch.from_numpy(colors).float().to(device),
        "normal": torch.zeros(pts.shape[0], 3).float().to(device),
        "batch": torch.zeros(pts.shape[0], dtype=torch.long).to(device),
    }, center


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_gt(path):
    return np.load(path)["points"]


# -- Core: single-step x0 with controllable noise -------------------------

@torch.no_grad()
def get_conditioning(model, point_dict, target_coords):
    """
    Pre-compute conditioning features once per frame so we only run the
    encoder a single time instead of N times.
    """
    model.eval()
    cond_features, _ = model.condition_extractor(point_dict)
    cond_features = knn_interpolate(cond_features, point_dict["coord"], target_coords)
    model.scheduler._to_device(target_coords.device)
    return cond_features


@torch.no_grad()
def run_completion_x0_fast(model, target_coords, cond_features, noise):
    """
    Single-step x0 prediction at t=200, using pre-computed conditioning
    and caller-provided noise tensor. This avoids re-running the encoder
    for each stochastic sample and gives full control over the noise seed.
    """
    device = target_coords.device
    t_val = 200
    t_tensor = torch.full((1,), t_val, device=device)
    sa = model.scheduler.sqrt_alphas_cumprod[t_val]
    som = model.scheduler.sqrt_one_minus_alphas_cumprod[t_val]

    noisy = sa * target_coords + som * noise
    pred_noise = model.denoiser(noisy, target_coords, t_tensor, {"features": cond_features})
    pred_x0 = (noisy - som * pred_noise) / sa
    return pred_x0.cpu().numpy()


# -- Uncertainty computation -----------------------------------------------

def compute_per_point_uncertainty(completions):
    """
    Given a list of N completions (each N_pts x 3 numpy arrays, all sharing
    the same point ordering from GT target coords), compute per-point std.

    Returns:
        mean_completion: (N_pts, 3) mean across all runs
        per_point_std: (N_pts,) L2 std per point
        per_point_std_xyz: (N_pts, 3) std per coordinate axis
    """
    stacked = np.stack(completions, axis=0)  # (N_runs, N_pts, 3)
    mean_completion = stacked.mean(axis=0)   # (N_pts, 3)
    per_point_std_xyz = stacked.std(axis=0)  # (N_pts, 3)
    per_point_std = np.linalg.norm(per_point_std_xyz, axis=1)  # (N_pts,)
    return mean_completion, per_point_std, per_point_std_xyz


def compute_per_point_error(mean_completion, gt_centered):
    """
    Per-point L2 error between mean completion and GT (nearest neighbor).
    Both arrays should be in the same centered coordinate frame.
    """
    tree = cKDTree(gt_centered)
    dists, _ = tree.query(mean_completion, k=1)
    return dists


def compute_cd(pred, gt, max_pts=10000):
    if pred.shape[0] > max_pts:
        pred = pred[np.random.choice(pred.shape[0], max_pts, replace=False)]
    if gt.shape[0] > max_pts:
        gt = gt[np.random.choice(gt.shape[0], max_pts, replace=False)]
    cd = chamfer_distance(
        torch.from_numpy(pred).float().cuda(),
        torch.from_numpy(gt).float().cuda(),
        chunk_size=512
    )
    return cd.item()


# -- Visualization ---------------------------------------------------------

def bev_uncertainty_plot(pts, std_vals, title, save_path, xlim=(-40, 40), ylim=(-40, 40)):
    """BEV scatter colored by per-point uncertainty (std)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: completion colored by height
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3, cmap="viridis", vmin=-2, vmax=4)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_title("Mean completion (height)")

    # Right: completion colored by uncertainty
    ax = axes[1]
    vmax = np.percentile(std_vals, 97)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=std_vals, s=0.3, cmap="hot", vmin=0, vmax=vmax)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_title("Per-point uncertainty (std)")
    plt.colorbar(sc, ax=ax, shrink=0.7, label="std (m)")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def correlation_plot(errors, stds, title, save_path):
    """Scatter of per-point error vs uncertainty with Spearman rho."""
    rho, pval = spearmanr(stds, errors)

    fig, ax = plt.subplots(figsize=(7, 6))
    # subsample for plotting clarity
    n = len(errors)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        errors_s, stds_s = errors[idx], stds[idx]
    else:
        errors_s, stds_s = errors, stds

    ax.scatter(stds_s, errors_s, s=0.5, alpha=0.3)
    ax.set_xlabel("Uncertainty (std, m)")
    ax.set_ylabel("NN error to GT (m)")
    ax.set_title(f"{title}\nSpearman rho={rho:.4f}, p={pval:.2e}")

    # trend line via binning
    n_bins = 20
    bin_edges = np.linspace(stds.min(), np.percentile(stds, 99), n_bins + 1)
    bin_centers, bin_means = [], []
    for j in range(n_bins):
        mask = (stds >= bin_edges[j]) & (stds < bin_edges[j + 1])
        if mask.sum() > 10:
            bin_centers.append((bin_edges[j] + bin_edges[j + 1]) / 2)
            bin_means.append(errors[mask].mean())
    if bin_centers:
        ax.plot(bin_centers, bin_means, "r-o", markersize=4, linewidth=2, label="binned mean")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return rho, pval


# -- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Uncertainty estimation via stochastic completions")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str,
                        default="/home/anywherevla/data2/dataset/sonata_depth_pro",
                        help="Root data path with sequences/ and ground_truth/")
    parser.add_argument("--da2_cloud_dir", type=str,
                        default="/home/anywherevla/data2/dataset/sonata_depth_pro/da2_output/pointclouds/sequences/08",
                        help="DA2 point cloud directory for --input_type da2")
    parser.add_argument("--input_type", type=str, default="lidar", choices=["lidar", "da2"],
                        help="Input modality: lidar or da2")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of stochastic completions per frame")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of frames to evaluate")
    parser.add_argument("--sequence", type=str, default="08")
    parser.add_argument("--output_dir", type=str, default="uncertainty_results")
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # -- Build model --
    print(f"Building model, input_type={args.input_type}, num_runs={args.num_runs}")
    model = load_ckpt(build_model(device), args.checkpoint, device)

    # -- Frame list --
    seq_dir = os.path.join(args.data_path, "sequences", args.sequence)
    vel_dir = os.path.join(seq_dir, "velodyne")
    gt_dir = os.path.join(args.data_path, "ground_truth", args.sequence)

    frames = sorted([f.replace(".bin", "") for f in os.listdir(vel_dir) if f.endswith(".bin")])
    step = max(1, len(frames) // args.num_samples)
    sample_frames = frames[::step][:args.num_samples]
    print(f"Evaluating {len(sample_frames)} frames from seq {args.sequence}")

    all_results = []
    all_stds = []
    all_errors = []

    for i, fid in enumerate(sample_frames):
        print(f"\n{'='*60}")
        print(f"Frame {fid} ({i+1}/{len(sample_frames)})")
        print(f"{'='*60}")

        # -- Load input point cloud --
        if args.input_type == "lidar":
            input_pts = load_bin(os.path.join(vel_dir, f"{fid}.bin"))
        else:
            da2_path = os.path.join(args.da2_cloud_dir, f"{fid}.bin")
            if not os.path.exists(da2_path):
                print(f"  SKIP: no DA2 cloud at {da2_path}")
                continue
            input_pts = load_bin(da2_path)

        # -- Load GT --
        gt_path = os.path.join(gt_dir, f"{fid}.npz")
        if not os.path.exists(gt_path):
            print(f"  SKIP: no GT at {gt_path}")
            continue
        gt_raw = load_gt(gt_path)

        # -- Prepare input scan (centered on input mean) --
        input_dict, center = prepare_scan(input_pts, device)

        # -- Prepare GT target coords (centered on input mean, same frame) --
        gt_centered = gt_raw - center
        vc = np.floor(gt_centered / 0.05).astype(np.int32)
        _, idx = np.unique(vc, axis=0, return_index=True)
        gt_sub = gt_centered[idx]
        if gt_sub.shape[0] > 20000:
            np.random.seed(0)  # deterministic subsampling
            sel = np.random.choice(gt_sub.shape[0], 20000, replace=False)
            gt_sub = gt_sub[sel]
        gt_target = torch.from_numpy(gt_sub).float().to(device)
        num_pts = gt_target.shape[0]

        # -- Pre-compute conditioning (encoder runs once) --
        t_start = time.time()
        cond_features = get_conditioning(model, input_dict, gt_target)
        cond_time = time.time() - t_start

        # -- Run N completions with different noise seeds --
        completions = []
        run_times = []
        for r in range(args.num_runs):
            torch.manual_seed(args.base_seed + r)
            noise = torch.randn(num_pts, 3, device=device)

            t0 = time.time()
            comp = run_completion_x0_fast(model, gt_target, cond_features, noise)
            run_times.append(time.time() - t0)
            completions.append(comp)

        avg_run_time = np.mean(run_times)
        total_time = time.time() - t_start
        print(f"  {args.num_runs} runs done: {avg_run_time:.3f}s/run, {total_time:.1f}s total (incl. encoder)")

        # -- Compute per-point uncertainty --
        mean_comp, per_point_std, per_point_std_xyz = compute_per_point_uncertainty(completions)

        # -- Per-point error (mean completion vs GT) --
        per_point_err = compute_per_point_error(mean_comp, gt_sub)

        # -- Chamfer distance of mean completion vs GT --
        mean_comp_world = mean_comp + center
        cd_val = compute_cd(mean_comp_world, gt_raw)

        # -- Correlation --
        rho, pval = spearmanr(per_point_std, per_point_err)

        print(f"  CD (mean completion): {cd_val:.4f}")
        print(f"  Uncertainty: mean={per_point_std.mean():.4f}, "
              f"median={np.median(per_point_std):.4f}, "
              f"p95={np.percentile(per_point_std, 95):.4f}")
        print(f"  Error: mean={per_point_err.mean():.4f}, "
              f"median={np.median(per_point_err):.4f}")
        print(f"  Spearman rho(std, error): {rho:.4f} (p={pval:.2e})")

        # -- BEV uncertainty visualization --
        bev_uncertainty_plot(
            mean_comp_world, per_point_std,
            f"Frame {fid} | {args.input_type.upper()} input | N={args.num_runs} | CD={cd_val:.4f}",
            os.path.join(args.output_dir, f"bev_uncertainty_{fid}.png")
        )

        # -- Correlation plot --
        rho_plot, _ = correlation_plot(
            per_point_err, per_point_std,
            f"Frame {fid} | {args.input_type.upper()} input",
            os.path.join(args.output_dir, f"correlation_{fid}.png")
        )

        # -- Collect --
        result = {
            "frame": fid,
            "input_type": args.input_type,
            "num_runs": args.num_runs,
            "num_points": int(num_pts),
            "cd": cd_val,
            "uncertainty_mean": float(per_point_std.mean()),
            "uncertainty_median": float(np.median(per_point_std)),
            "uncertainty_p95": float(np.percentile(per_point_std, 95)),
            "uncertainty_max": float(per_point_std.max()),
            "error_mean": float(per_point_err.mean()),
            "error_median": float(np.median(per_point_err)),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "cond_time_s": cond_time,
            "avg_run_time_s": avg_run_time,
            "total_time_s": total_time,
        }
        all_results.append(result)
        all_stds.extend(per_point_std.tolist())
        all_errors.extend(per_point_err.tolist())

    # -- Aggregate summary -------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not all_results:
        print("No frames processed!")
        return

    cds = [r["cd"] for r in all_results]
    unc_means = [r["uncertainty_mean"] for r in all_results]
    rhos = [r["spearman_rho"] for r in all_results]

    print(f"Input type: {args.input_type}")
    print(f"Num runs per frame: {args.num_runs}")
    print(f"Frames evaluated: {len(all_results)}")
    print(f"CD (mean completion): {np.mean(cds):.4f} +/- {np.std(cds):.4f}")
    print(f"Uncertainty (mean std): {np.mean(unc_means):.4f} +/- {np.std(unc_means):.4f}")
    print(f"Spearman rho (mean): {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}")

    # Global correlation across all points
    all_stds_arr = np.array(all_stds)
    all_errors_arr = np.array(all_errors)
    global_rho, global_pval = spearmanr(all_stds_arr, all_errors_arr)
    print(f"Global Spearman rho (all points): {global_rho:.4f} (p={global_pval:.2e})")

    # -- Global correlation plot --
    correlation_plot(
        all_errors_arr, all_stds_arr,
        f"All frames | {args.input_type.upper()} | N={args.num_runs}",
        os.path.join(args.output_dir, "correlation_global.png")
    )

    # -- Per-frame table --
    print(f"\n{'Frame':<10} {'CD':<10} {'Unc mean':<12} {'Err mean':<12} {'Spearman':<10}")
    print("-" * 54)
    for r in all_results:
        print(f"{r['frame']:<10} {r['cd']:<10.4f} {r['uncertainty_mean']:<12.4f} "
              f"{r['error_mean']:<12.4f} {r['spearman_rho']:<10.4f}")

    # -- Save JSON --
    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "input_type": args.input_type,
            "num_runs": args.num_runs,
            "num_samples": args.num_samples,
            "sequence": args.sequence,
            "base_seed": args.base_seed,
        },
        "aggregate": {
            "cd_mean": float(np.mean(cds)),
            "cd_std": float(np.std(cds)),
            "uncertainty_mean": float(np.mean(unc_means)),
            "uncertainty_std": float(np.std(unc_means)),
            "spearman_rho_mean": float(np.mean(rhos)),
            "spearman_rho_std": float(np.std(rhos)),
            "global_spearman_rho": float(global_rho),
            "global_spearman_pval": float(global_pval),
            "num_frames": len(all_results),
        },
        "per_frame": all_results,
    }
    out_path = os.path.join(args.output_dir, "uncertainty_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/")
    print(f"  uncertainty_metrics.json")
    print(f"  bev_uncertainty_*.png")
    print(f"  correlation_*.png")
    print(f"  correlation_global.png")


if __name__ == "__main__":
    main()
