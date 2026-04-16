#!/bin/bash
set -euo pipefail

WORKSPACE="/home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace"
DATA_PATH="/home/anywherevla/sonata_ws/dataset/sonata_depth_pro"
VAE_CKPT="${WORKSPACE}/checkpoints/point_vae/best_point_vae.pth"
LOG_FILE="/home/anywherevla/sonata_ws/latent_diffusion.log"

cd "${WORKSPACE}"

# Notify start
curl -s -X POST "https://api.pushcut.io/D8JGLuy3yU6eiYPzCzhBo/notifications/Claude%20code" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Latent diffusion training STARTED on compute (4090). 100 epochs, batch_size=8, fp16.\"}" > /dev/null 2>&1 || true

python -m training.train_diffusion_latent \
  --data_path "${DATA_PATH}" \
  --vae_ckpt "${VAE_CKPT}" \
  --freeze_encoder \
  --fp16 \
  --batch_size 8 \
  --num_workers 8 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --num_timesteps 1000 \
  --schedule cosine \
  --denoising_steps 50 \
  --voxel_size_sonata 0.05 \
  --point_max_partial 20000 \
  --point_max_complete 8000 \
  --output_dir checkpoints/latent_diffusion \
  --log_dir logs/latent_diffusion \
  --save_freq 5 \
  --eval_freq 1 \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  MSG="Latent diffusion training FINISHED SUCCESSFULLY on compute."
else
  MSG="Latent diffusion training FAILED on compute (exit code ${EXIT_CODE}). Check ${LOG_FILE}"
fi

curl -s -X POST "https://api.pushcut.io/D8JGLuy3yU6eiYPzCzhBo/notifications/Claude%20code" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"${MSG}\"}" > /dev/null 2>&1 || true

exit $EXIT_CODE
