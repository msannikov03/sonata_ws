# Semantic Scene Completion with Sonata-LiDiff

A learning pipeline for semantic scene completion that combines:
- **Sonata**: Self-supervised Point Transformer V3 encoder for robust point representations
- **LiDiff**: Diffusion-based scene completion framework
- **SemanticKITTI**: Benchmark dataset for outdoor LiDAR scene understanding

## Architecture Overview

```
Input LiDAR Scan → Sonata Encoder → Diffusion Process → Scene Completion
                        ↓
                 Rich Point Features
                        ↓
              Conditional Generation
                        ↓
              Complete 3D Scene + Semantics
```

### Key Components

1. **Sonata Encoder** (Point Transformer V3)
   - Pre-trained on large-scale multi-dataset
   - Hierarchical point cloud encoding
   - Self-attention with efficient grouped vector attention
   - Outputs rich, reliable point representations

2. **LiDiff Diffusion Framework**
   - Point-wise local diffusion modeling with Sonata-style transformer blocks
   - Uses grouped vector attention (replacing sparse convolutions)
   - Learns neighborhood distributions via transformer attention
   - Conditional generation from partial scans
   - Refinement network for detail enhancement

3. **SemanticKITTI Integration**
   - Outdoor driving scene completion
   - 19 semantic classes
   - Ground truth generation from sequential scans
   - Standard benchmarking metrics

## Project Structure

```
sonata_lidiff/
├── configs/
│   ├── sonata_encoder.yaml      # Sonata encoder configuration
│   ├── diffusion_model.yaml     # Diffusion process configuration
│   ├── refinement.yaml          # Refinement network configuration
│   └── training.yaml            # Training hyperparameters
├── data/
│   ├── semantickitti.py         # SemanticKITTI dataset handler
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── ground_truth.py          # GT generation from sequences
│   └── augmentation.py          # Data augmentation
├── models/
│   ├── sonata_encoder.py        # Sonata encoder wrapper
│   ├── diffusion_module.py      # Diffusion process implementation
│   ├── refinement_net.py        # Refinement network
│   └── complete_model.py        # Full pipeline integration
├── training/
│   ├── train.py                 # Main training script
│   ├── train_diffusion.py       # Diffusion model training
│   ├── train_refinement.py      # Refinement network training
│   └── losses.py                # Loss functions
├── evaluation/
│   ├── metrics.py               # Evaluation metrics (IoU, completion)
│   ├── evaluate.py              # Evaluation script
│   └── visualize.py             # Visualization tools
├── utils/
│   ├── transforms.py            # Point cloud transformations
│   ├── checkpoint.py            # Model checkpointing
│   └── logger.py                # Training logging
└── scripts/
    ├── setup_dataset.sh         # Dataset download and setup
    ├── generate_gt.sh           # Ground truth generation
    └── run_training.sh          # Training launcher
```

## Installation

### Prerequisites
- Python 3.9+ (spconv-cu124 requires 3.9+)
- CUDA 11.8+ / 12.x
- PyTorch 2.0+

### Environment Setup

**Recommended: use the install script** (handles numpy/scipy compatibility):

```bash
conda create -n sonata_lidiff python=3.9 -y
conda activate sonata_lidiff
bash install_dependencies.sh
```

**Manual setup:**

```bash
# Create conda environment (Python 3.9+ for spconv-cu124)
conda create -n sonata_lidiff python=3.9 -y
conda activate sonata_lidiff

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies (numpy<1.28 + scipy==1.11.4 for Open3D compatibility)
pip install -r requirements.txt

# Install Sonata encoder dependencies
# torch-scatter: use PyG wheel matching your PyTorch version (conda build can ABI-mismatch)
pip install torch-scatter -f "https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html"
pip install flash-attn --no-build-isolation  # optional
pip install huggingface_hub timm

# Install project
pip install -e .
```

## Dataset Setup

### SemanticKITTI Dataset

1. **Download Dataset**
```bash
# Download from http://www.semantic-kitti.org/dataset.html
# Organize as:
# Datasets/SemanticKITTI/dataset/sequences/
#   ├── 00/velodyne/, 00/labels/
#   ├── 01/velodyne/, 01/labels/
#   ├── ...
#   └── 21/velodyne/, 21/labels/
```

2. **Generate Ground Truth Maps**
```bash
python data/ground_truth.py \
  --dataset_path Datasets/SemanticKITTI/dataset/sequences/ \
  --output_path Datasets/SemanticKITTI/ground_truth/ \
  --sequences 00 01 02 03 04 05 06 07 09 10
```

This creates complete scene maps by aggregating sequential scans using poses.

## Training Pipeline

### Stage 1: Train Diffusion Model with Sonata Encoder

```bash
python training/train_diffusion.py \
  --config configs/diffusion_model.yaml \
  --encoder_ckpt facebook/sonata \
  --data_path Datasets/SemanticKITTI \
  --output_dir checkpoints/diffusion \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-4
```

**Key Training Details:**
- Freezes Sonata encoder initially (optional fine-tuning later)
- Learns point-wise diffusion using transformer attention blocks
- Uses Sonata-style grouped vector attention for local neighborhood processing
- Uses ground truth complete scenes as targets
- Conditional generation from partial LiDAR scans

### Stage 2: Train Refinement Network

```bash
python training/train_refinement.py \
  --config configs/refinement.yaml \
  --diffusion_ckpt checkpoints/diffusion/best_model.pth \
  --data_path Datasets/SemanticKITTI \
  --output_dir checkpoints/refinement \
  --batch_size 4 \
  --num_epochs 50
```

**Refinement Objectives:**
- Detail enhancement of diffusion outputs
- Semantic consistency enforcement
- Boundary sharpening
- Noise reduction

## Evaluation

```bash
python evaluation/evaluate.py \
  --diffusion_ckpt checkpoints/diffusion/best_model.pth \
  --refinement_ckpt checkpoints/refinement/best_model.pth \
  --data_path Datasets/SemanticKITTI \
  --split test \
  --output_dir results/
```

**Metrics:**
- **Scene Completion IoU** (Intersection over Union)
- **Semantic Scene Completion (SSC) IoU**: Per-class and mean IoU
- **Completion Accuracy** at 0.5cm, 0.2cm, 0.1cm thresholds
- **Chamfer Distance**: Geometric accuracy

## Inference

```bash
python inference.py \
  --input_scan /path/to/scan.bin \
  --diffusion_ckpt checkpoints/diffusion/best_model.pth \
  --refinement_ckpt checkpoints/refinement/best_model.pth \
  --output /path/to/output.ply \
  --visualize
```

## Configuration

### Key Hyperparameters

**Sonata Encoder:**
```yaml
encoder:
  model: "sonata"
  pretrained: "facebook/sonata"
  freeze: true  # Freeze initially, fine-tune later
  patch_size: [1024, 1024, 1024, 1024, 1024]
  enable_flash: true
  feature_dim: 384
```

**Diffusion Process:**
```yaml
diffusion:
  timesteps: 1000
  noise_schedule: "cosine"
  beta_start: 0.0001
  beta_end: 0.02
  conditioning: "encoder_features"
  denoising_steps: 50  # Inference
```

**Training:**
```yaml
training:
  batch_size: 4
  learning_rate: 1e-4
  weight_decay: 0.01
  epochs: 100
  warmup_epochs: 10
  gradient_clip: 1.0
```

## Technical Details

### Architecture Improvements

**Transformer-based Diffusion Module:**
- The diffusion denoising network now uses Sonata-style transformer blocks instead of sparse convolutions
- **Grouped Vector Attention**: Efficient attention mechanism that processes features in groups
- **Point-based Processing**: Works directly with point features and coordinates, no sparse tensor conversion needed
- **Local Attention**: Each point attends to its k-nearest neighbors for efficient local processing
- **Benefits**: More flexible than sparse convolutions, better captures long-range dependencies, easier to extend

### Sonata Encoder Integration

The Sonata encoder provides hierarchical point features:
- **Level 0** (Input): Raw scan points
- **Level 1-5**: Progressively downsampled with increased receptive field
- **Features**: 384-dimensional embeddings per point
- **Attention**: Efficient grouped vector attention

### Diffusion Framework

Following LiDiff's approach with Sonata-style transformer architecture:
- **Forward Process**: Gradually adds Gaussian noise to complete scenes
- **Reverse Process**: Learns to denoise using transformer blocks, conditioned on:
  - Partial scan geometry
  - Sonata encoder features
  - Semantic class information
- **Transformer-based Denoising**: Uses Sonata-style grouped vector attention blocks
  - Replaces sparse convolutions with transformer attention mechanisms
  - Processes points directly with local attention neighborhoods
  - More flexible and expressive than sparse convolution-based approaches
- **Point-wise Modeling**: Treats completion as local neighborhood prediction via attention

### Loss Functions

```python
# Total loss
L_total = L_diffusion + λ_sem * L_semantic + λ_geo * L_geometric

# Diffusion loss (denoising score matching)
L_diffusion = MSE(ε_θ(x_t, t, c), ε)

# Semantic segmentation loss
L_semantic = CrossEntropy(pred_classes, gt_classes)

# Geometric consistency loss
L_geometric = Chamfer(completed_points, gt_points)
```

## Results

Expected performance on SemanticKITTI validation set:

| Method | mIoU | Completion@0.2cm | Completion@0.1cm |
|--------|------|------------------|------------------|
| LiDiff (original) | - | 16.79 | 4.67 |
| LiDiff + Refined | - | 22.99 | 13.40 |
| **Sonata-LiDiff** (ours) | TBD | TBD | TBD |

## Visualization

```bash
# Visualize completion results
python evaluation/visualize.py \
  --input results/sequence_00/ \
  --output visualizations/ \
  --type comparison  # [completion, semantic, comparison]
```

## Citations

If you use this work, please cite:

```bibtex
@inproceedings{wu2025sonata,
  title={Sonata: Self-Supervised Learning of Reliable Point Representations},
  author={Wu, Xiaoyang and others},
  booktitle={CVPR},
  year={2025}
}

@inproceedings{nunes2024cvpr,
  title={Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion},
  author={Nunes, Lucas and Marcuzzi, Rodrigo and Mersch, Benedikt and Behley, Jens and Stachniss, Cyrill},
  booktitle={CVPR},
  year={2024}
}
```

## License

- Code: MIT License
- Sonata weights: CC-BY-NC 4.0 (non-commercial)
- LiDiff code: MIT License

## Acknowledgments

- Sonata team at Meta Reality Labs
- LiDiff team at University of Bonn
- SemanticKITTI dataset maintainers
