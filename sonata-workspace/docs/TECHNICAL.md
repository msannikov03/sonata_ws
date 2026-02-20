# Technical Documentation: Sonata-LiDiff

## Architecture Deep Dive

### 1. Sonata Encoder Architecture

The Sonata encoder is a Point Transformer V3 with the following key features:

#### Hierarchical Structure
```
Level 0 (Input): Raw points
    ↓ Patch Embedding
Level 1: 1024 patches → Self-Attention
    ↓ Stride Pooling (2x)
Level 2: 512 patches → Self-Attention
    ↓ Stride Pooling (2x)
Level 3: 256 patches → Self-Attention
    ↓ Stride Pooling (2x)
Level 4: 128 patches → Self-Attention
    ↓ Stride Pooling (2x)
Level 5: 64 patches → Self-Attention (bottleneck)
```

#### Key Components

**Grouped Vector Attention (GVA):**
- Divides features into groups
- Computes attention within each group
- More efficient than standard attention
- Better for large point clouds

**Patch-based Processing:**
- Groups nearby points into patches
- Processes patches in parallel
- Reduces computational complexity
- Maintains local structure

**Feature Dimensions:**
- Each level produces 384-dimensional features
- Projected to 256-dim for diffusion conditioning
- Multi-level features capture different scales

### 2. Diffusion Process

#### Forward Process (Training)

The forward process gradually adds Gaussian noise to complete scenes:

```
x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
```

where:
- `x_0`: Clean complete scene
- `x_t`: Noisy version at timestep t
- `α̅_t`: Cumulative product of alphas
- `ε`: Standard Gaussian noise

**Noise Schedule:**
- Cosine schedule (default): Smooth noise addition
- Linear schedule: Uniform noise increase
- Sigmoid schedule: Accelerated noise at start/end

#### Reverse Process (Inference)

The reverse process iteratively denoises to recover complete scene:

```
x_{t-1} = μ_θ(x_t, t, c) + σ_t * z

where μ_θ is predicted by the denoising network
```

**Denoising Steps:**
- Training: 1000 steps
- Inference: 50 steps (DDIM sampling)
- Can trade quality vs speed

### 3. Point-wise Local Modeling

Following LiDiff's approach, we model diffusion point-wise:

**Key Insight:**
Instead of modeling entire scene distribution, model local neighborhood distribution for each point.

**Advantages:**
1. Scales to large scenes
2. Learns local geometric patterns
3. Generalizes across different scene sizes
4. Efficient memory usage

**Implementation:**
```python
# For each point:
1. Extract local neighborhood (K=16 neighbors)
2. Encode with sparse convolution
3. Predict noise for that point
4. Condition on Sonata encoder features
```

### 4. Conditional Generation

#### Multi-level Conditioning

Features from 3 hierarchical levels (2, 3, 4) are fused:

**Attention Fusion:**
```
# Compute attention weights
weights = softmax(MLP([f_2, f_3, f_4]))

# Weighted combination
f_cond = w_2 * f_2 + w_3 * f_3 + w_4 * f_4
```

**Hierarchical Fusion:**
```
# Progressive refinement
f = f_2
f = MLP([f, f_3])  # Refine with level 3
f = MLP([f, f_4])  # Refine with level 4
```

#### Conditioning Mechanism

Conditional features are injected at each diffusion block:

```python
# Time embedding
t_feat = time_mlp(t)

# Condition injection
x = x + t_feat + condition_proj(cond_features)

# Process
x = conv(x)
x = mlp(x)
```

### 5. Loss Functions

#### Primary Loss: Denoising Score Matching

```
L_diffusion = E[||ε_θ(x_t, t, c) - ε||²]
```

Predicts the added noise at each timestep.

#### Optional Auxiliary Losses

**Semantic Loss:**
```
L_sem = CrossEntropy(pred_classes, gt_classes)
```

Encourages semantic consistency in completed scenes.

**Geometric Loss:**
```
L_geo = Chamfer(completed_points, gt_points)
```

Ensures geometric accuracy.

**Total Loss:**
```
L_total = L_diffusion + λ_sem * L_sem + λ_geo * L_geo
```

### 6. Training Strategy

#### Two-Stage Training

**Stage 1: Diffusion Model**
1. Freeze Sonata encoder
2. Train diffusion process
3. 100 epochs, lr=1e-4
4. Cosine annealing schedule

**Stage 2: Fine-tuning (Optional)**
1. Unfreeze encoder
2. Fine-tune end-to-end
3. 20 epochs, lr=1e-5
4. Lower learning rate to avoid catastrophic forgetting

#### Data Augmentation

Applied during training:
- Random rotation (±180° around z-axis)
- Random flip (50% probability)
- Random scaling (0.95-1.05)
- Point jitter (σ=0.01m)

#### Memory Optimization

**Gradient Accumulation:**
```python
# Effective batch size = 4 * 4 = 16
batch_size = 4
accumulation_steps = 4
```

**Mixed Precision Training:**
```python
# Enable AMP for 2x speedup
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(...)
```

**Sparse Tensors:**
- Use MinkowskiEngine for efficient sparse ops
- Only process occupied voxels
- Saves ~10x memory

### 7. Evaluation Protocol

#### Scene Completion Metrics

**Completion Ratio @ threshold:**
```
CR_τ = (# GT points with pred neighbor < τ) / (total GT points)
```

Standard thresholds: 0.5cm, 0.2cm, 0.1cm

**Chamfer Distance:**
```
CD = 1/2 * (mean d(pred→GT) + mean d(GT→pred))
```

Measures geometric accuracy.

#### Semantic Metrics

**Per-class IoU:**
```
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

**Mean IoU:**
```
mIoU = (1/C) * Σ IoU_c
```

Averaged over all classes.

### 8. SemanticKITTI Specifics

#### Dataset Structure

**Training sequences:** 00-07, 09-10 (19,130 scans)
**Validation sequence:** 08 (4,071 scans)
**Test sequences:** 11-21 (20,351 scans)

#### Class Distribution

19 semantic classes + unlabeled (20 total):
```
0:  unlabeled          10: parking
1:  car                11: sidewalk
2:  bicycle            12: other-ground
3:  motorcycle         13: building
4:  truck              14: fence
5:  other-vehicle      15: vegetation
6:  person             16: trunk
7:  bicyclist          17: terrain
8:  motorcyclist       18: pole
9:  road               19: traffic-sign
```

#### Ground Truth Generation

For each scan:
1. Load sequential scans (-5 to +5 frames)
2. Transform to current frame using poses
3. Aggregate points from all frames
4. Voxelize at 5cm resolution
5. Store as complete scene ground truth

**Command:**
```bash
python data/ground_truth.py \
  --dataset_path Datasets/SemanticKITTI/dataset/sequences/ \
  --output_path Datasets/SemanticKITTI/ground_truth/ \
  --window_size 5 \
  --voxel_size 0.05
```

### 9. Hyperparameter Tuning

#### Key Hyperparameters

**Learning Rate:**
- Start: 1e-4 (diffusion), 1e-5 (fine-tuning)
- Schedule: Cosine annealing
- Min LR: 1e-6

**Batch Size:**
- GPU memory dependent
- 4-8 typical on 24GB GPU
- Use gradient accumulation for larger effective batch

**Voxel Size:**
- 0.05m: Standard, good balance
- 0.02m: High detail, more memory
- 0.1m: Lower detail, faster training

**Denoising Steps:**
- Training: 1000 (full schedule)
- Inference: 50 (DDIM fast sampling)
- Can reduce to 25 for 2x speedup with minimal quality loss

**Feature Levels:**
- [2, 3, 4]: Multi-scale, best quality
- [3, 4]: Faster, similar performance
- [4]: Fastest, coarsest features

### 10. Common Issues & Solutions

#### Issue: CUDA OOM

**Solutions:**
1. Reduce batch size
2. Enable gradient accumulation
3. Reduce max_points per sample
4. Use gradient checkpointing
5. Reduce patch sizes in Sonata

#### Issue: Slow Training

**Solutions:**
1. Enable flash attention
2. Use mixed precision (AMP)
3. Increase num_workers for data loading
4. Precompute ground truth maps
5. Use SSD for data storage

#### Issue: Poor Completion Quality

**Solutions:**
1. Increase denoising steps
2. Try different noise schedules
3. Adjust conditioning fusion type
4. Fine-tune encoder
5. Add auxiliary losses

#### Issue: Semantic Inconsistency

**Solutions:**
1. Enable semantic loss
2. Use stronger conditioning
3. Fine-tune with semantic KITTI labels
4. Post-process with CRF

### 11. Extensions & Future Work

#### Possible Improvements

1. **Multi-frame Fusion:**
   - Use multiple past scans as input
   - Temporal consistency loss
   - Better completion in occluded areas

2. **Dynamic Objects:**
   - Separate dynamic/static completion
   - Motion prediction
   - Track-before-complete

3. **Uncertainty Estimation:**
   - Ensemble multiple samples
   - Predict completion confidence
   - Active learning for labeling

4. **Real-time Optimization:**
   - Distillation to smaller model
   - Quantization
   - Fewer denoising steps with learned scheduler

5. **Other Datasets:**
   - nuScenes (multi-camera + LiDAR)
   - Waymo Open Dataset
   - KITTI-360

### 12. Citation

When using this code, please cite:

```bibtex
@inproceedings{wu2025sonata,
  title={Sonata: Self-Supervised Learning of Reliable Point Representations},
  author={Wu, Xiaoyang and others},
  booktitle={CVPR},
  year={2025}
}

@inproceedings{nunes2024cvpr,
  title={Scaling Diffusion Models to Real-World 3D LiDAR Scene Completion},
  author={Nunes, Lucas and others},
  booktitle={CVPR},
  year={2024}
}
```
