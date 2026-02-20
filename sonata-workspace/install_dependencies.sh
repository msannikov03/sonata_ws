#!/bin/bash
# Installation script for Sonata-LiDiff dependencies
# Run this script with: bash install_dependencies.sh
#
# PREREQUISITE: Create env with Python 3.9+ (spconv-cu124 needs 3.9+)
#   conda create -n sonata_lidiff python=3.9 -y
#   conda activate sonata_lidiff

set -e  # Exit on error

echo "=========================================="
echo "Installing Sonata-LiDiff Dependencies"
echo "=========================================="

# Activate conda environment
echo "Activating conda environment: sonata_lidiff"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sonata_lidiff

# Require Python >=3.9 (spconv-cu124, flash-attn wheels, etc.)
echo "Python version: $(python --version)"
if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
  echo "ERROR: Python 3.9+ required (spconv-cu124 has no PyPI wheels for 3.8)."
  echo "Recreate env: conda create -n sonata_lidiff python=3.9 -y && conda activate sonata_lidiff"
  exit 1
fi

# Step 1: Install PyTorch with CUDA 12.4 support
echo ""
echo "Step 1: Installing PyTorch with CUDA 12.4..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Step 2: Install spconv with CUDA support (requires Python >=3.9 for spconv-cu124)
echo ""
echo "Step 2: Installing spconv-cu124..."
pip install spconv-cu124

# Verify spconv installation
echo "Verifying spconv installation..."
python -c "import spconv; print(f'spconv version: {spconv.__version__}'); print('spconv imported successfully!')" || echo "Warning: spconv import check failed"

# Step 3: Install torch-scatter from PyG wheels (must match PyTorch version; conda build can ABI-mismatch)
echo ""
echo "Step 3: Installing torch-scatter..."
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VER}.html"

# Step 4: Install flash-attention (optional but recommended)
echo ""
echo "Step 4: Installing flash-attention..."
pip install flash-attn --no-build-isolation || echo "Warning: flash-attn installation failed (optional dependency)"

# Step 5: Install Sonata encoder dependencies
echo ""
echo "Step 5: Installing Sonata encoder dependencies..."
pip install huggingface_hub timm

# Step 6: Install remaining requirements from requirements.txt
# (numpy<1.28 + scipy==1.11.4 pinned for Open3D compatibility; scipy 1.13.x causes TypeError)
echo ""
echo "Step 6: Installing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 7: Install project in editable mode
echo ""
echo "Step 7: Installing project in editable mode..."
pip install -e .

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python -c "
import torch
import numpy as np
import scipy
import yaml
import open3d as o3d
import tensorboard
import tqdm
import spconv
print('✓ All core dependencies imported successfully!')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  NumPy: {np.__version__}')
print(f'  SciPy: {scipy.__version__}')
print(f'  Open3D: {o3d.__version__}')
print(f'  spconv: {spconv.__version__}')
"

echo ""
echo "Installation verification complete!"
