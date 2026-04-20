"""Shared point cloud ops (sensor frame, KITTI-style origin at 0,0,0)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def crop_lidar_radius(points: np.ndarray, radius_m: float) -> np.ndarray:
    """
    Keep rows within Euclidean distance ``radius_m`` of the origin.

    ``radius_m <= 0`` disables cropping. Used for local-scene training/inference.
    """
    if radius_m <= 0 or points.size == 0:
        return points
    d = np.linalg.norm(np.asarray(points, dtype=np.float64), axis=1)
    return np.asarray(points[d <= radius_m + 1e-5], dtype=np.float32)


def crop_lidar_radius_with_labels(
    points: np.ndarray, labels: np.ndarray, radius_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as ``crop_lidar_radius`` but keeps ``labels`` aligned row-wise."""
    if radius_m <= 0 or points.size == 0:
        return points, labels
    d = np.linalg.norm(np.asarray(points, dtype=np.float64), axis=1)
    m = d <= radius_m + 1e-5
    return points[m], labels[m]
