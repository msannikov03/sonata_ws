# `map_from_scans_boost_v2.py` Runbook

This file documents the recommended PowerShell command for running
[`map_from_scans_boost_v2.py`](d:\mag_skoltech\term_3\ML\project\sonata_ws\sonata-workspace\data\map_from_scans_boost_v2.py).

## Main idea

- repo root is `D:\mag_skoltech\term_3\ML\project\sonata_ws`
- all paths below are derived from that root only
- run from `sonata-workspace`
- keep `ICP` enabled
- use GPU voxelization
- save output into `ground_truth`
- remove `_v2` suffix so the output looks like regular `map_from_scans.py`

## Base paths

Use this root:

```text
D:\mag_skoltech\term_3\ML\project\sonata_ws
```

Then define paths like this:

```powershell
$REPO_ROOT = "D:\mag_skoltech\term_3\ML\project\sonata_ws"
$WORKSPACE = Join-Path $REPO_ROOT "sonata-workspace"
$DATASET_ROOT = Join-Path $REPO_ROOT "dataset\SemanticKITTI\dataset"
$SEQ_PATH = Join-Path $DATASET_ROOT "sequences"
```

## Recommended command

Use this as the main command for all SemanticKITTI sequences `00-10`:

```powershell
$REPO_ROOT = "\sonata_ws"
$WORKSPACE = Join-Path $REPO_ROOT "sonata-workspace"
$DATASET_ROOT = Join-Path $REPO_ROOT "dataset\SemanticKITTI\dataset"
$SEQ_PATH = Join-Path $DATASET_ROOT "sequences"

cd $WORKSPACE

python data/map_from_scans_boost_v2.py `
  --path $SEQ_PATH `
  --output $DATASET_ROOT `
  --output_subdir ground_truth `
  --name_suffix "" `
  --gpu_voxelize `
  --backend open3d `
  --voxel_size 0.1 `
  --window_size 17 `
  --accumulation_radius 15 `
  --output_radius 20 `
  --max_gt_points 200000 `
  --icp-reference-n 3 `
  --icp-correction-max 0.6 `
  --icp-max-iter 5 `
  --icp-threshold 1.0 `
  --icp-downsample 0.25 `
  --sor-neighbors 12 `
  --sor-std-ratio 2.0 `
  --ror-nb-points 5 `
  --ror-radius 0.5 `
  --sequences 00 01 02 03 04 05 06 07 08 09 10 `
  --force
```

## Why these parameters

- `--gpu_voxelize` enables Torch voxelization on CUDA when available
- `--backend open3d` keeps ICP and point-cloud filtering in a good working mode
- `--output_subdir ground_truth` matches the usual project layout
- `--name_suffix ""` removes `_v2` from output filenames
- `--window_size 17` gives a 35-frame context window
- `--accumulation_radius 15` is a reasonable local merge radius
- `--output_radius 20` keeps a wider final local map
- `--icp-reference-n 3` keeps the anchor reference stable and not too heavy
- `--icp-correction-max 0.6` limits overly aggressive ICP corrections

## Expected output

After the run, files should appear like this:

```text
$REPO_ROOT\dataset\SemanticKITTI\dataset\ground_truth\00\000000.npz
...
$REPO_ROOT\dataset\SemanticKITTI\dataset\ground_truth\10\xxxxxx.npz
```

Each `.npz` is expected to contain:

```python
points
```

## Quick check

```powershell
$REPO_ROOT = "D:\mag_skoltech\term_3\ML\project\sonata_ws"
$DATASET_ROOT = Join-Path $REPO_ROOT "dataset\SemanticKITTI\dataset"

Get-ChildItem (Join-Path $DATASET_ROOT "ground_truth\00") | Select-Object -First 5
```

## Full help

```powershell
$REPO_ROOT = "D:\mag_skoltech\term_3\ML\project\sonata_ws"
$WORKSPACE = Join-Path $REPO_ROOT "sonata-workspace"

cd $WORKSPACE
python data/map_from_scans_boost_v2.py --help
```
