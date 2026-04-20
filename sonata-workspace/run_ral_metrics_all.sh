#!/bin/bash
# Autonomous runner for RA-L metric suite across all 5 configurations.
# Runs sequentially, coexists with other GPU jobs (uses <4GB VRAM, fast inference).
# Logs to /home/anywherevla/ral_metrics_run.log

set -u  # but NOT -e, so one failure doesn't kill everything
WD=/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace
LOG=/home/anywherevla/ral_metrics_run.log
OUT_DIR=results/apr17_morning
NF=200

cd "$WD"
exec >> "$LOG" 2>&1

echo ""
echo "=========================================="
echo "[$(date)] RA-L metrics run starting"
echo "=========================================="
echo "Output dir: $OUT_DIR"
echo "Frames per config: $NF"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
    teacher_v2gt_lidar_v2
    teacher_v2gt_da2_v2
    teacher_v2gt_lidar_v1
    random_ptv3_lidar_v2
    random_ptv3_da2_v2
)

for CFG in "${CONFIGS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "[$(date)] CONFIG: $CFG"
    echo "------------------------------------------"
    python3 evaluate_ral_metrics.py \
        --config "$CFG" \
        --num_frames "$NF" \
        --output_dir "$OUT_DIR"
    RC=$?
    echo "[$(date)] $CFG exit code: $RC"
done

echo ""
echo "=========================================="
echo "[$(date)] RA-L metrics run COMPLETE"
echo "=========================================="

# Consolidate summaries
python3 - <<'PY'
import os, json, glob
base = "/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace/results/apr17_morning"
out = {}
for p in sorted(glob.glob(os.path.join(base, "*", "all_metrics.json"))):
    cfg = os.path.basename(os.path.dirname(p))
    with open(p) as f:
        out[cfg] = json.load(f)["summary"]
with open(os.path.join(base, "ALL_SUMMARIES.json"), "w") as f:
    json.dump(out, f, indent=2)

print("\nFinal table:")
keys = ["cd_mean", "cd_sq_mean", "jsd_mean",
        "f_score@0.1_mean", "f_score@0.2_mean",
        "iou@0.1_mean", "iou@0.2_mean", "hausdorff_95_mean"]
header = f"{'config':<30s} " + " ".join(f"{k:<14s}" for k in keys)
print(header)
print("-" * len(header))
for cfg, s in out.items():
    row = f"{cfg:<30s} " + " ".join(f"{s.get(k, float('nan')):<14.4f}" for k in keys)
    print(row)
PY

# Telegram notification
ssh -o StrictHostKeyChecking=no business 'telegram-notify "[sonata] RA-L metrics run complete. Results in results/apr17_morning/ALL_SUMMARIES.json on compute."' || \
    echo "[$(date)] telegram notification failed"

echo "[$(date)] done"
