# Dependency Checklist (check before creating conda env)

Use this to pick **Python version** and **CUDA** so all packages install without conflicts.

## Python version

| Package | Minimum Python | Notes |
|---------|----------------|--------|
| **spconv-cu124** | **3.9** | PyPI has no wheels for 3.8 |
| PyTorch 2.x | 3.8 | 3.8–3.12 supported |
| torch-scatter | 3.8 | Use **pip install torch-scatter -f** PyG wheel (match PyTorch version; conda can ABI-mismatch) |
| flash-attn | 3.8* | *Prebuilt wheels easier for 3.9–3.12; 2.5.9 last with 3.8 |
| Open3D | 3.8 | 3.8–3.12 |
| h5py, numpy, scipy, etc. | 3.8 | All support 3.8+ |

**Recommendation: Python 3.9** (or 3.10). Satisfies spconv-cu124 and keeps everything else compatible.

## CUDA / PyTorch

| Item | Requirement |
|------|-------------|
| PyTorch | Install `cu124` (or cu121/cu118) to match GPU driver. CUDA 13 systems: use cu124 wheels. |
| spconv | Use `spconv-cu124` (or cu121/cu118) to match PyTorch CUDA build. |
| torch-scatter | Use **pip install torch-scatter -f https://data.pyg.org/whl/torch-{version}.html** (match PyTorch; conda build can ABI-mismatch). |

## Other constraints

| Package | Constraint |
|---------|------------|
| numpy | **&lt;1.28** (Open3D). Use `numpy>=1.21.0,<1.28`. Tested: 1.26.4. |
| scipy | **==1.11.4** (1.13.x causes TypeError with Open3D/scipy.interpolate). |
| setup.py | Currently `python_requires=">=3.9"` when using spconv-cu124. |

## Conda env (recommended)

```bash
# Use Python 3.9 so spconv-cu124 and all others work
conda create -n sonata_lidiff python=3.9 -y
conda activate sonata_lidiff
cd /home/didar/Simon_ws/sonata-workspace
bash install_dependencies.sh
```

## Summary

- **Python: 3.9** (required for spconv-cu124 from PyPI).
- **CUDA: 12.4** (use PyTorch cu124 + spconv-cu124; OK on CUDA 13 driver).
- **numpy: &lt;1.28** (for Open3D; tested 1.26.4).
- **scipy: 1.11.4** (pinned; 1.13.x breaks Open3D import).
