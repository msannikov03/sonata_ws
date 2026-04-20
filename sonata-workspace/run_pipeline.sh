#!/bin/bash
cd /home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace

echo "[$(date)] Pipeline started. Waiting for ablations to finish..."

# Wait for ablations to complete (check if run_ablations.py is still running)
while pgrep -f "run_ablations.py" > /dev/null 2>&1; do
    sleep 30
done

echo "[$(date)] Ablations DONE."
ssh business "telegram-notify \"Ablations COMPLETE. Results in logs/ablations.log. Starting DA2 training now.\""

# Launch DA2 training (random unfrozen encoder)
echo "[$(date)] Starting DA2 training..."
python experiments/train_random_ptv3.py \
    --input_type da2 \
    --encoder_mode random_unfrozen \
    --epochs 10 \
    --batch_size 1 \
    --accum_steps 2 \
    --lr 1e-4 \
    --data_dir /home/anywherevla/sonata_ws/prevoxelized_seq08 \
    --output_dir checkpoints/random_ptv3_da2 \
    2>&1 | tee logs/random_ptv3_da2.log

echo "[$(date)] DA2 training DONE."

# Get final results
LIDAR_BEST=$(grep "^Epoch" logs/random_ptv3_lidar.log | sort -t= -k3 -n | head -1)
DA2_BEST=$(grep "^Epoch" logs/random_ptv3_da2.log | sort -t= -k3 -n | head -1)

ssh business "telegram-notify \"ALL EXPERIMENTS COMPLETE!
LiDAR best: $LIDAR_BEST
DA2 best: $DA2_BEST
Ablation results: logs/ablations.log
Ready for paper writing.\""

echo "[$(date)] Pipeline complete."
