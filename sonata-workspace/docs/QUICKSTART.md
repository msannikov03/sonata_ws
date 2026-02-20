# Quick Start Guide

This guide will walk you through setting up and training Sonata-LiDiff from scratch.

## Prerequisites

- Ubuntu 20.04/22.04 or similar Linux distribution
- NVIDIA GPU with ≥16GB VRAM (24GB recommended)
- CUDA 11.8 or 12.x
- Python 3.8+
- 200GB free disk space for dataset

## Step 1: Environment Setup (15 minutes)

### Option A: Conda (Recommended)

```bash
# Create conda environment
conda create -n sonata_lidiff python=3.8
conda activate sonata_lidiff

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install core dependencies
pip install numpy scipy pyyaml tqdm tensorboard open3d

# Install MinkowskiEngine
sudo apt install build-essential python3-dev libopenblas-dev
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
  --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
  --install-option="--blas=openblas"

# Install Sonata dependencies
pip install spconv-cu121
pip install torch-scatter -f "https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html"
pip install git+https://github.com/Dao-AILab/flash-attention.git
pip install huggingface_hub timm

# Install project
git clone https://github.com/yourusername/sonata-lidiff.git
cd sonata-lidiff
pip install -e .
```

### Option B: Docker

```bash
# Build Docker image
docker build -t sonata-lidiff:latest .

# Run container
docker run --gpus all -it \
  -v /path/to/data:/workspace/data \
  -v /path/to/output:/workspace/output \
  sonata-lidiff:latest
```

## Step 2: Download SemanticKITTI (30 minutes)

```bash
# Create data directory
mkdir -p Datasets/SemanticKITTI
cd Datasets/SemanticKITTI

# Download dataset (80GB)
# Visit: http://www.semantic-kitti.org/dataset.html#download
# Download the following:
# - Velodyne point clouds (80GB)
# - Labels (179MB)
# - Calibration files (1KB)

# Extract files
# Final structure:
# Datasets/SemanticKITTI/dataset/sequences/
#   ├── 00/
#   │   ├── velodyne/  (*.bin files)
#   │   ├── labels/    (*.label files)
#   │   └── calib.txt
#   ├── 01/
#   └── ...
```

## Step 3: Generate Ground Truth Maps (2 hours)

```bash
cd sonata-lidiff

# Generate complete scene maps from sequential scans
python data/ground_truth.py \
  --dataset_path Datasets/SemanticKITTI/dataset/sequences/ \
  --output_path Datasets/SemanticKITTI/ground_truth/ \
  --sequences 00 01 02 03 04 05 06 07 09 10 \
  --window_size 5 \
  --voxel_size 0.05 \
  --num_workers 8

# This processes ~19k scans and creates complete maps
# Output: ~50GB of .npz files
```

**What this does:**
- For each scan, loads nearby scans (±5 frames)
- Transforms them using poses to current frame
- Aggregates all points into complete scene
- Voxelizes at 5cm resolution
- Saves as ground truth for training

## Step 4: Verify Installation (5 minutes)

```bash
# Test dataset loading
python -c "
from data.semantickitti import SemanticKITTI
dataset = SemanticKITTI(
    root='Datasets/SemanticKITTI/dataset',
    split='train',
    voxel_size=0.05
)
print(f'Dataset loaded: {len(dataset)} samples')
sample = dataset[0]
print(f'Sample keys: {sample.keys()}')
print('✓ Dataset OK')
"

# Test Sonata encoder
python -c "
from models.sonata_encoder import SonataEncoder
encoder = SonataEncoder(
    pretrained='facebook/sonata',
    freeze=True,
    enable_flash=False
).cuda()
print('✓ Sonata encoder OK')
"

# Test complete model
python -c "
from models.sonata_encoder import SonataEncoder, ConditionalFeatureExtractor
from models.diffusion_module import SceneCompletionDiffusion

encoder = SonataEncoder('facebook/sonata', freeze=True, enable_flash=False)
cond_ext = ConditionalFeatureExtractor(encoder, feature_levels=[2,3,4])
model = SceneCompletionDiffusion(encoder, cond_ext).cuda()
print('✓ Complete model OK')
"
```

## Step 5: Start Training (2-3 days on single GPU)

### Quick Training (Test Run)

```bash
# Train for 5 epochs to verify everything works
python training/train_diffusion.py \
  --config configs/diffusion_model.yaml \
  --data_path Datasets/SemanticKITTI/dataset \
  --batch_size 2 \
  --num_epochs 5 \
  --output_dir checkpoints/test \
  --log_dir logs/test
```

### Full Training

```bash
# Train complete model
python training/train_diffusion.py \
  --config configs/diffusion_model.yaml \
  --data_path Datasets/SemanticKITTI/dataset \
  --encoder_ckpt facebook/sonata \
  --freeze_encoder \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --output_dir checkpoints/diffusion \
  --log_dir logs/diffusion \
  --save_freq 5 \
  --eval_freq 1

# Monitor training
tensorboard --logdir logs/diffusion --port 6006
# Open browser: http://localhost:6006
```

**Training Tips:**

- **Single GPU (24GB):** Use batch_size=4
- **Single GPU (16GB):** Use batch_size=2, accumulation_steps=2
- **Multiple GPUs:** Use DDP (see distributed training section)
- **Speed:** ~2-3 days for 100 epochs on RTX 4090

### Resume Training

```bash
# Resume from checkpoint
python training/train_diffusion.py \
  --config configs/diffusion_model.yaml \
  --resume checkpoints/diffusion/checkpoint_epoch_50.pth \
  --num_epochs 100
```

## Step 6: Evaluation (30 minutes)

```bash
# Evaluate on validation set
python evaluation/evaluate.py \
  --checkpoint checkpoints/diffusion/best_model.pth \
  --data_path Datasets/SemanticKITTI/dataset \
  --split val \
  --output_dir results/val \
  --denoising_steps 50 \
  --visualize_samples 10

# Results will be saved to results/val/
# - metrics.json: Quantitative results
# - visualizations/: Completed scenes
```

**Expected Results (Validation Set):**

| Metric | Value |
|--------|-------|
| Completion @ 0.5cm | ~30% |
| Completion @ 0.2cm | ~15% |
| Completion @ 0.1cm | ~4% |
| Chamfer Distance | ~0.15m |
| Mean IoU | ~25% |

## Step 7: Inference on New Scans (1 minute)

```bash
# Complete a single scan
python inference.py \
  --input /path/to/scan.bin \
  --checkpoint checkpoints/diffusion/best_model.pth \
  --output output_completed.ply \
  --denoising_steps 50 \
  --visualize

# This will:
# 1. Load the scan
# 2. Complete the scene
# 3. Save result to .ply file
# 4. Show visualization (if --visualize)
```

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory

```bash
# Solution: Reduce batch size and use gradient accumulation
python training/train_diffusion.py \
  --batch_size 1 \
  --accumulation_steps 4 \
  ...
```

### Issue 2: Slow Data Loading

```bash
# Solution: Precompute GT maps and increase workers
# 1. Verify GT maps exist in ground_truth/
# 2. Increase num_workers
python training/train_diffusion.py \
  --num_workers 8 \
  ...
```

### Issue 3: Flash Attention Not Available

```bash
# Solution: Disable flash attention
python training/train_diffusion.py \
  --config configs/diffusion_model.yaml
  # Edit config: enable_flash: false
  # And reduce enc_patch_size to [512, 512, 512, 512, 512]
```

### Issue 4: Poor Completion Quality

```bash
# Solutions:
# 1. Train longer (100+ epochs)
# 2. Increase denoising steps
python inference.py \
  --denoising_steps 100 \
  ...

# 3. Fine-tune encoder
python training/train_diffusion.py \
  --resume checkpoints/diffusion/best_model.pth \
  # Set freeze_encoder: false in config
  --learning_rate 1e-5 \
  --num_epochs 120
```

## Next Steps

### 1. Fine-tune with Semantic Labels

```bash
# Add semantic segmentation head
# Edit config: semantic_weight: 0.1
python training/train_diffusion.py \
  --config configs/diffusion_semantic.yaml \
  --resume checkpoints/diffusion/best_model.pth
```

### 2. Train Refinement Network

```bash
# Stage 2: Detail enhancement
python training/train_refinement.py \
  --config configs/refinement.yaml \
  --diffusion_ckpt checkpoints/diffusion/best_model.pth \
  --num_epochs 50
```

### 3. Distributed Training (Multiple GPUs)

```bash
# Use all available GPUs
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  training/train_diffusion.py \
  --config configs/diffusion_model.yaml \
  --batch_size 4  # Per GPU
```

### 4. Export for Deployment

```bash
# Convert to ONNX
python tools/export_onnx.py \
  --checkpoint checkpoints/diffusion/best_model.pth \
  --output models/sonata_lidiff.onnx
```

## Benchmarking

```bash
# Test inference speed
python tools/benchmark.py \
  --checkpoint checkpoints/diffusion/best_model.pth \
  --num_samples 100 \
  --batch_size 1

# Typical results (RTX 4090):
# - Single scan completion: ~5 seconds (50 steps)
# - Throughput: ~12 scans/minute
```

## Visualization Tools

```bash
# Generate comparison video
python tools/visualize_sequence.py \
  --checkpoint checkpoints/diffusion/best_model.pth \
  --sequence 08 \
  --output_video completion_video.mp4

# Interactive viewer
python tools/viewer.py \
  --checkpoint checkpoints/diffusion/best_model.pth \
  --data_path Datasets/SemanticKITTI/dataset
```

## Getting Help

- **Documentation:** See `docs/TECHNICAL.md` for architecture details
- **Issues:** Open issue on GitHub
- **Discord:** Join our community server
- **Email:** your.email@example.com

## Tips for Best Results

1. **Always generate GT maps** before training (Step 3)
2. **Monitor TensorBoard** during training
3. **Start with frozen encoder** then fine-tune if needed
4. **Use validation set** to select best checkpoint
5. **Increase denoising steps** for better quality inference
6. **Experiment with conditioning** fusion types
7. **Try different noise schedules** (cosine usually best)
8. **Use mixed precision** for faster training
9. **Batch size 4-8** is optimal for most GPUs
10. **Save checkpoints regularly** in case of crashes

## Success Checklist

- [ ] Environment installed correctly
- [ ] SemanticKITTI dataset downloaded
- [ ] Ground truth maps generated
- [ ] Verification tests passed
- [ ] Training started successfully
- [ ] TensorBoard accessible
- [ ] Validation loss decreasing
- [ ] Checkpoints being saved
- [ ] Evaluation metrics computed
- [ ] Inference working on test scans

## Congratulations!

You now have a working semantic scene completion system. Happy training! 🚀
