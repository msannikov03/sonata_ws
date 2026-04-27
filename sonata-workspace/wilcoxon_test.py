#!/usr/bin/env python3
"""
Reproducible Wilcoxon signed-rank test for RA-L Table III.

Backs the "p < 10^-12" claim in the Table III caption: paired
50-frame stride-80 scaffold-free comparison of the mixed-scaffold
fine-tuned teacher (Ours) against LiDiff and ScoreLiDAR
author-released checkpoints, all evaluated on the same v2 GT under
the same protocol.

Reads three committed JSONs:
  - eval_wilcoxon_finetuned.json     (ours-FT, 50 frames)
  - eval_lidiff_on_v2gt_50fr.json    (LiDiff, 48 frames)
  - eval_scorelidar_on_v2gt_50fr.json (ScoreLiDAR, 50 frames)

Outputs the four paired Wilcoxon statistics matching the Table III
caption: ours < LiDiff diff/refine and ours < ScoreLiDAR diff/refine.
"""
import json
from pathlib import Path
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent

ours = json.loads((HERE / "eval_wilcoxon_finetuned.json").read_text())
ours_map = dict(zip(ours["frame_id"], ours["ours_ft_cd_sq"]))

lidiff = json.loads((HERE / "eval_lidiff_on_v2gt_50fr.json").read_text())
sl = json.loads((HERE / "eval_scorelidar_on_v2gt_50fr.json").read_text())


def paired(per_frame_baseline, ours_map):
    """Pair ours-FT and baseline per-frame CD² by frame_id."""
    o, bd, br = [], [], []
    for r in per_frame_baseline:
        fid = r["frame_id"]
        if fid not in ours_map:
            continue
        o.append(ours_map[fid])
        bd.append(r["cd_diff"]["cd_mean"])
        br.append(r["cd_refine"]["cd_mean"])
    return o, bd, br


for name, base in [("LiDiff", lidiff), ("ScoreLiDAR", sl)]:
    o, bd, br = paired(base["per_frame"], ours_map)
    n = len(o)
    diff = wilcoxon(o, bd, alternative="less")
    refine = wilcoxon(o, br, alternative="less")
    om = sum(o) / n
    bdm = sum(bd) / n
    brm = sum(br) / n
    print(f"\n{name} (n={n} paired):")
    print(f"  ours_ft mean={om:.4f}  {name}_diff mean={bdm:.4f}  refine mean={brm:.4f}")
    print(f"  Wilcoxon ours<{name}_diff   stat={diff.statistic:.1f}  p={diff.pvalue:.2e}")
    print(f"  Wilcoxon ours<{name}_refine stat={refine.statistic:.1f}  p={refine.pvalue:.2e}")
