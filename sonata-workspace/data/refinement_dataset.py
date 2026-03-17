"""
Refinement Dataset for Semantic Scene Completion

Provides (coarse_points, dense_points) pairs for training the refinement network.
Coarse = voxelized at coarse_voxel_size (simulates diffusion output).
Dense = voxelized at fine_voxel_size (target).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from .semantickitti import SemanticKITTI


class RefinementDataset(Dataset):
    """
    Wraps SemanticKITTI to produce coarse->dense pairs for refinement.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        coarse_voxel_size: float = 0.1,
        fine_voxel_size: float = 0.05,
        max_points_coarse: int = 4000,
        max_points_fine: int = 12000,
        use_ground_truth_maps: bool = True,
        sequences: List[str] = None,
    ):
        """
        Args:
            root: Dataset root (path to dataset folder)
            split: 'train', 'val', or 'test'
            coarse_voxel_size: Voxel size for coarse input
            fine_voxel_size: Voxel size for dense target
            max_points_coarse: Cap on coarse points
            max_points_fine: Cap on dense points
            use_ground_truth_maps: Use pre-generated GT maps
            sequences: Override split sequences
        """
        self.base_dataset = SemanticKITTI(
            root=root,
            split=split,
            voxel_size=fine_voxel_size,
            max_points=max_points_fine,
            use_ground_truth_maps=use_ground_truth_maps,
            augmentation=(split == "train"),
        )
        self.coarse_voxel_size = coarse_voxel_size
        self.fine_voxel_size = fine_voxel_size
        self.max_points_coarse = max_points_coarse
        self.max_points_fine = max_points_fine

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _voxelize(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Voxelize to unique voxel centers."""
        voxel_coords = np.floor(points / voxel_size).astype(np.int32)
        unique_voxels = np.unique(voxel_coords, axis=0)
        centers = unique_voxels * voxel_size + voxel_size / 2
        return centers.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]

        # Dense = complete_coord (already voxelized at fine_voxel_size)
        dense = sample["complete_coord"].numpy()

        # Coarse = voxelize dense at coarse resolution (simulates diffusion output)
        coarse = self._voxelize(dense, self.coarse_voxel_size)

        # Subsample if needed
        if coarse.shape[0] > self.max_points_coarse:
            idx_sample = np.random.choice(
                coarse.shape[0], self.max_points_coarse, replace=False
            )
            coarse = coarse[idx_sample]

        if dense.shape[0] > self.max_points_fine:
            idx_sample = np.random.choice(
                dense.shape[0], self.max_points_fine, replace=False
            )
            dense = dense[idx_sample]

        return {
            "coarse": torch.from_numpy(coarse).float(),
            "dense": torch.from_numpy(dense).float(),
            "scan_center": sample["scan_center"],
        }


def collate_refinement(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate for refinement (variable sizes -> list or pad)."""
    coarse_list = [b["coarse"] for b in batch]
    dense_list = [b["dense"] for b in batch]

    # Pad to max in batch for batching
    max_coarse = max(c.shape[0] for c in coarse_list)
    max_dense = max(d.shape[0] for d in dense_list)

    coarse_padded = []
    dense_padded = []
    coarse_mask = []

    for c, d in zip(coarse_list, dense_list):
        nc, nd = c.shape[0], d.shape[0]
        pad_c = torch.zeros(max_coarse, 3)
        pad_c[:nc] = c
        coarse_padded.append(pad_c)

        pad_d = torch.zeros(max_dense, 3)
        pad_d[:nd] = d
        dense_padded.append(pad_d)

        mask = torch.zeros(max_coarse, dtype=torch.bool)
        mask[:nc] = True
        coarse_mask.append(mask)

    return {
        "coarse": torch.stack(coarse_padded),
        "dense": torch.stack(dense_padded),
        "coarse_mask": torch.stack(coarse_mask),
        "coarse_lengths": torch.tensor([c.shape[0] for c in coarse_list]),
        "dense_lengths": torch.tensor([d.shape[0] for d in dense_list]),
        "scan_center": torch.stack([b["scan_center"] for b in batch]),
    }
