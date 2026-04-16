#!/bin/bash
set -e

WORKSPACE="/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
DATA_PATH="/home/anywherevla/sonata_ws/dataset/sonata_depth_pro"
LOG_FILE="/home/anywherevla/sonata_ws/vae_training.log"

cd "$WORKSPACE"

notify() {
    curl -s -X POST "https://api.pushcut.io/D8JGLuy3yU6eiYPzCzhBo/notifications/Claude%20code" \
        -H "Content-Type: application/json" \
        -d "{\"text\":\"$1\"}" > /dev/null 2>&1 || true
}

echo "=== VAE Training Started: $(date) ===" | tee -a "$LOG_FILE"
notify "VAE training started on compute (4M params, 50 epochs, bs=4)"

python3 training/train_point_vae.py \
    --data_path "$DATA_PATH" \
    --output_dir checkpoints/point_vae \
    --log_dir logs/point_vae \
    --batch_size 8 \
    --num_workers 8 \
    --num_epochs 50 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --gradient_clip 1.0 \
    --latent_dim 256 \
    --num_decoded_points 2048 \
    --beta_kl 1e-3 \
    --point_max_complete 8000 \
    --save_freq 5 \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== VAE Training Completed: $(date) ===" | tee -a "$LOG_FILE"
    notify "VAE training COMPLETED on compute. Check checkpoints/point_vae/best_point_vae.pth"
else
    echo "=== VAE Training FAILED (exit $EXIT_CODE): $(date) ===" | tee -a "$LOG_FILE"
    notify "VAE training FAILED on compute (exit $EXIT_CODE)"
fi

exit $EXIT_CODE
