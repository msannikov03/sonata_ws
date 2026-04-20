"""Thin wrapper: patches build_partial_dict to produce non-negative grid_coord,
then runs train_diffusion_latent.main()."""

import training.train_diffusion_latent as tmod
import torch

_orig_build = tmod.build_partial_dict

def _patched_build(batch, voxel_size_sonata: float) -> dict:
    coord = batch["partial_coord"]
    gc = torch.floor(coord / voxel_size_sonata).long()
    # Shift per batch sample so grid_coord is non-negative (required by Sonata serialization)
    batch_idx = batch["partial_batch"]
    for b in batch_idx.unique():
        mask = batch_idx == b
        gc[mask] -= gc[mask].min(dim=0)[0]
    return {
        "coord": coord,
        "color": batch["partial_color"],
        "normal": batch["partial_normal"],
        "grid_coord": gc,
        "batch": batch_idx,
    }

tmod.build_partial_dict = _patched_build

if __name__ == "__main__":
    tmod.main()
