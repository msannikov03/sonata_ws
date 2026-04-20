#!/bin/bash
# Generate Boost v2 ground truth for all sequences (00-10)
# Uses scipy ICP fallback (no open3d on Python 3.13)
# Estimated time: ~16 hours for all 23k frames

cd /home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace

python3 data/map_from_scans_boost_v2.py \
  -p /home/anywherevla/sonata_ws/dataset/sonata_depth_pro/sequences \
  -o /home/anywherevla/sonata_ws/dataset/sonata_depth_pro \
  --output_subdir ground_truth_v2 \
  -b numpy \
  --name_suffix _v2 \
  --voxel_size 0.1 \
  --window_size 17 \
  --accumulation_radius 15.0 \
  --output_radius 20.0 \
  --max_gt_points 200000 \
  --icp-correction-max 0.15 \
  --icp-reference-n 3 \
  --icp-downsample 0.35 \
  --icp-max-iter 4 \
  --sor-neighbors 10 \
  --sor-std-ratio 2.0 \
  --ror-nb-points 5 \
  --ror-radius 0.5 \
  2>&1 | tee /home/anywherevla/sonata_ws/sonata-workspace-fixed/sonata-workspace/logs/boost_v2_gt_gen.log

echo "Done! $(date)"
