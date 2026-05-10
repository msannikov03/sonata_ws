#!/bin/bash
# Master orchestrator for Tier-3 (C-tier) RA-L experiments
# Total compute estimate: ~7.5 hours on RTX 4090
# Run with: bash run_C_experiments.sh 2>&1 | tee /tmp/C_experiments.log

set -e
WORK=/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace
cd "$WORK"

START=$(date +%s)
echo '======================================================================'
echo '  TIER-3 (C) EXPERIMENTS — RA-L PAPER'
echo '  C3a: LiDPM-protocol on FT model (~10min)'
echo '  C3b: LiDPM-protocol on pre-FT teacher (~10min) — clean ablation, no train/test contamination'
echo '  C1:  FT-checkpoint ablations (~3hr)'
echo '  C2:  Seeds 45/46/47 train+eval (~3hr)'
echo "  Started: $(date)"
echo '======================================================================'

# ---- C3 first (cheapest, fastest signal) ----
echo
echo '>>> C3a: LiDPM-protocol on FT model <<<'
python3 run_lidpm_protocol_baseline.py
echo "[C3a done at $(date)]"

echo
echo '>>> C3b: LiDPM-protocol on PRE-FT teacher (clean ablation) <<<'
python3 run_lidpm_protocol_baseline_preFT.py
echo "[C3b done at $(date)]"

# ---- C1: FT-checkpoint ablations ----
echo
echo '>>> C1: FT-checkpoint ablations <<<'
python3 run_ablations_FT.py
echo "[C1 done at $(date)]"

# ---- C2: Seeds 45/46/47 sequential ----
for seed in 45 46 47; do
  echo
  echo ">>> C2: Seed ${seed} training <<<"
  python3 finetune_mixed_scaffold_seed${seed}.py
  echo ">>> C2: Seed ${seed} evaluation <<<"
  python3 run_scaffoldfree_fair_finetuned_seed${seed}.py
  echo "[C2 seed ${seed} done at $(date)]"
done

END=$(date +%s)
ELAPSED=$((END - START))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo
echo '======================================================================'
echo "  ALL C-TIER EXPERIMENTS COMPLETE in ${HOURS}h ${MINS}m"
echo "  Outputs:"
echo "    C1:  $WORK/ablation_results_FT.json"
echo "    C2:  $WORK/results/apr17_morning/teacher_finetuned_str80_seed{45,46,47}.json"
echo "    C3a: $WORK/results/apr17_morning/lidpm_protocol_baseline.json (FT model)"
echo "    C3b: $WORK/results/apr17_morning/lidpm_protocol_baseline_preFT.json (pre-FT teacher)"
echo "  C3 2x2 grid: {pre-FT, post-FT} x {ego-bbox, LiDPM-protocol} cleanly isolates scaffold choice from FT recipe."
echo '======================================================================'

# Telegram notify on completion
if command -v telegram-notify >/dev/null 2>&1; then
  telegram-notify "RA-L Tier-3 C experiments complete in ${HOURS}h ${MINS}m"
fi
