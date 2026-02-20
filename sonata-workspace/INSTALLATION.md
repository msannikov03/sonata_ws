# Installation Requirements

**Before creating a conda env:** See [DEPENDENCIES_CHECKLIST.md](DEPENDENCIES_CHECKLIST.md) for Python/CUDA constraints. Use **Python 3.9** so `spconv-cu124` installs from PyPI.

## System Prerequisites

- **Python**: **3.9** or 3.10 (3.8 not supported for spconv-cu124)
- **CUDA**: 11.8+ or 12.x (match spconv version; CUDA 13 → use cu124 wheels)
- **PyTorch**: 2.0+

## Core Dependencies

### Deep Learning Framework:
- `torch>=2.0.0`
- `torchvision>=0.15.0` (typically installed with torch)

### Sparse Convolution (replaced MinkowskiEngine):
- `spconv-cu124` (or `spconv-cu118`, `spconv-cu121` depending on your CUDA version)

### Sonata Encoder Dependencies:
- `torch-scatter`
- `flash-attn>=2.0.0` (optional but recommended)
- `huggingface_hub`
- `timm`

### Scientific Computing:
- `numpy>=1.21.0,<1.28` (Open3D requires numpy <2.0; 1.26.4 tested)
- `scipy==1.11.4` (pinned; 1.13.x causes TypeError with Open3D)

### Point Cloud Processing:
- `open3d>=0.16.0`

### Data Processing:
- `pyyaml>=5.4.0` (or `yaml`)

### Training & Logging:
- `tensorboard>=2.8.0`
- `tqdm>=4.60.0`

### Package Management:
- `setuptools` (usually included with Python)

## Installation Order (Recommended)

```bash
# 1. Create conda environment (Python 3.9+ for spconv-cu124)
conda create -n sonata_lidiff python=3.9 -y
conda activate sonata_lidiff

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install spconv (match your CUDA version)
pip install spconv-cu124  # or spconv-cu118, spconv-cu121

# 4. Install torch-scatter (PyG wheel matching PyTorch version; conda can ABI-mismatch)
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VER}.html"

# 5. Install flash-attention (optional but recommended)
pip install flash-attn --no-build-isolation

# 6. Install core dependencies (use requirements.txt for correct numpy/scipy versions)
pip install -r requirements.txt
pip install huggingface_hub timm

# 7. Install project in editable mode
cd /home/didar/Simon_ws/sonata-workspace
pip install -e .
```

## Notes:

- Match the spconv CUDA version to your PyTorch CUDA version (e.g., cu121, cu118, cu124)
- Flash-attention may require compilation; install with `--no-build-isolation` if needed
- The Sonata encoder loads from HuggingFace, so `huggingface_hub` is required
- All other imports (`os`, `sys`, `argparse`, `typing`, `logging`, `datetime`, `pickle`) are standard library modules
- **The code now uses spconv instead of MinkowskiEngine, so MinkowskiEngine is not required.**
