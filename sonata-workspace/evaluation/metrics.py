"""
Evaluation Metrics for Semantic Scene Completion

Implements standard SSC metrics:
- Completion ratio at different thresholds
- Chamfer distance
- Semantic IoU
- Mean IoU across classes

Added Apr 17 for RA-L paper comparability with LiDiff / ScoreLiDAR / LiNeXt / LiFlow:
- Jensen-Shannon Divergence (BEV voxelized, 0.5m) — LiDiff protocol
- F-score @ 0.1m, 0.2m
- Voxel IoU @ 0.1m, 0.2m (bridges to SSC: MonoScene, VoxFormer, CGFormer)
- Hausdorff-95 (95th percentile directed distance, worst-case geometry)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree


class CompletionMetrics:
    """Metrics for scene completion evaluation."""
    
    def __init__(
        self,
        thresholds: List[float] = [0.5, 0.2, 0.1],  # in cm
        num_classes: int = 20
    ):
        """
        Initialize metrics.
        
        Args:
            thresholds: Distance thresholds in cm for completion metrics
            num_classes: Number of semantic classes
        """
        self.thresholds = [t / 100.0 for t in thresholds]  # Convert to meters
        self.num_classes = num_classes
        
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.completion_counts = {t: 0 for t in self.thresholds}
        self.total_points = 0
        
        self.chamfer_distances = []
        
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
    
    @torch.no_grad()
    def update(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor,
        pred_labels: torch.Tensor = None,
        gt_labels: torch.Tensor = None
    ):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred_points: Predicted complete point cloud (N, 3)
            gt_points: Ground truth complete point cloud (M, 3)
            pred_labels: Predicted semantic labels (N,)
            gt_labels: Ground truth semantic labels (M,)
        """
        # Convert to numpy
        if isinstance(pred_points, torch.Tensor):
            pred_points = pred_points.cpu().numpy()
        if isinstance(gt_points, torch.Tensor):
            gt_points = gt_points.cpu().numpy()
        
        # Completion metrics
        self._update_completion(pred_points, gt_points)
        
        # Chamfer distance
        self._update_chamfer(pred_points, gt_points)
        
        # Semantic metrics
        if pred_labels is not None and gt_labels is not None:
            self._update_semantic(
                pred_points, gt_points, pred_labels, gt_labels
            )
    
    def _update_completion(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray
    ):
        """
        Update completion ratio metrics.
        
        Completion ratio = percentage of GT points with a predicted point
        within threshold distance.
        """
        # Build KD-tree for predicted points
        tree = cKDTree(pred_points)
        
        # Find nearest predicted point for each GT point
        distances, _ = tree.query(gt_points, k=1)
        
        # Count points within each threshold
        for threshold in self.thresholds:
            within_threshold = (distances < threshold).sum()
            self.completion_counts[threshold] += within_threshold
        
        self.total_points += len(gt_points)
    
    def _update_chamfer(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray
    ):
        """Update Chamfer distance."""
        # Build KD-trees
        pred_tree = cKDTree(pred_points)
        gt_tree = cKDTree(gt_points)
        
        # Pred -> GT distances
        dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
        
        # GT -> Pred distances
        dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
        
        # Chamfer distance (symmetric)
        chamfer = (
            dist_pred_to_gt.mean() + dist_gt_to_pred.mean()
        ) / 2.0
        
        self.chamfer_distances.append(chamfer)
    
    def _update_semantic(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        pred_labels: torch.Tensor,
        gt_labels: torch.Tensor
    ):
        """
        Update semantic segmentation metrics.
        
        Matches predicted and GT points, then updates confusion matrix.
        """
        if isinstance(pred_labels, torch.Tensor):
            pred_labels = pred_labels.cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()
        
        # Build KD-tree for predicted points
        tree = cKDTree(pred_points)
        
        # Find nearest predicted point for each GT point
        _, indices = tree.query(gt_points, k=1)
        
        # Get matched labels
        matched_pred_labels = pred_labels[indices]
        
        # Update confusion matrix
        for gt_class in range(self.num_classes):
            mask = gt_labels == gt_class
            if mask.sum() == 0:
                continue
            
            pred_for_class = matched_pred_labels[mask]
            for pred_class in range(self.num_classes):
                count = (pred_for_class == pred_class).sum()
                self.confusion_matrix[gt_class, pred_class] += count
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Completion ratios
        for threshold in self.thresholds:
            ratio = self.completion_counts[threshold] / max(self.total_points, 1)
            # Convert back to cm for reporting
            threshold_cm = threshold * 100
            metrics[f'completion_{threshold_cm:.1f}cm'] = ratio * 100
        
        # Chamfer distance
        if len(self.chamfer_distances) > 0:
            metrics['chamfer_distance'] = np.mean(self.chamfer_distances)
        
        # Semantic metrics
        if self.confusion_matrix.sum() > 0:
            # Per-class IoU
            intersection = np.diag(self.confusion_matrix)
            union = (
                self.confusion_matrix.sum(axis=0) + 
                self.confusion_matrix.sum(axis=1) - 
                intersection
            )
            
            valid_classes = union > 0
            iou_per_class = np.zeros(self.num_classes)
            iou_per_class[valid_classes] = (
                intersection[valid_classes] / union[valid_classes]
            )
            
            # Mean IoU
            metrics['mean_iou'] = iou_per_class[valid_classes].mean()
            
            # Store per-class IoU
            for i in range(self.num_classes):
                if valid_classes[i]:
                    metrics[f'iou_class_{i}'] = iou_per_class[i]
            
            # Overall accuracy
            metrics['overall_accuracy'] = (
                intersection.sum() / self.confusion_matrix.sum()
            )
        
        return metrics
    
    def __str__(self) -> str:
        """String representation of current metrics."""
        metrics = self.compute()
        
        lines = ["Evaluation Metrics:"]
        lines.append("-" * 50)
        
        # Completion metrics
        lines.append("Completion Ratios:")
        for key, value in metrics.items():
            if key.startswith('completion'):
                lines.append(f"  {key}: {value:.2f}%")
        
        # Chamfer distance
        if 'chamfer_distance' in metrics:
            lines.append(f"\nChamfer Distance: {metrics['chamfer_distance']:.4f} m")
        
        # Semantic metrics
        if 'mean_iou' in metrics:
            lines.append(f"\nSemantic Segmentation:")
            lines.append(f"  Mean IoU: {metrics['mean_iou']:.4f}")
            lines.append(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        
        return "\n".join(lines)


def evaluate_scene_completion(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    pred_labels: torch.Tensor = None,
    gt_labels: torch.Tensor = None,
    thresholds: List[float] = [0.5, 0.2, 0.1],
    num_classes: int = 20
) -> Dict[str, float]:
    """
    Evaluate scene completion for a single sample.
    
    Args:
        pred_points: Predicted point cloud
        gt_points: Ground truth point cloud
        pred_labels: Predicted semantic labels
        gt_labels: Ground truth semantic labels
        thresholds: Distance thresholds in cm
        num_classes: Number of semantic classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = CompletionMetrics(thresholds, num_classes)
    metrics.update(pred_points, gt_points, pred_labels, gt_labels)
    return metrics.compute()


# =============================================================================
# Extended metrics for RA-L paper (Apr 17)
# =============================================================================

def _as_np(x) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy float32 array (N, 3)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def jsd_bev(
    pred_pts,
    gt_pts,
    voxel_size: float = 0.5,
    bbox_min: Optional[Tuple[float, float]] = None,
    bbox_max: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Jensen-Shannon Divergence in BEV (top-down, X-Y) between two point clouds.

    Matches the LiDiff / ScoreLiDAR protocol: voxelize to a 2D BEV grid at
    voxel_size resolution, normalize the occupancy histogram into a probability
    distribution over cells, then compute the symmetric JSD.

    Args:
        pred_pts: (N, 3) predicted points
        gt_pts:   (M, 3) ground-truth points
        voxel_size: BEV cell size in meters (LiDiff uses 0.5)
        bbox_min/bbox_max: optional (x_min, y_min) / (x_max, y_max). If None,
            computed from the joint support of the two clouds.

    Returns:
        JSD in [0, log(2)] where 0 means identical distributions.
    """
    pred = _as_np(pred_pts)[:, :2]
    gt = _as_np(gt_pts)[:, :2]
    if len(pred) == 0 or len(gt) == 0:
        return float("nan")

    joint = np.concatenate([pred, gt], axis=0)
    if bbox_min is None:
        bbox_min = (joint[:, 0].min(), joint[:, 1].min())
    if bbox_max is None:
        bbox_max = (joint[:, 0].max(), joint[:, 1].max())

    x_min, y_min = bbox_min
    x_max, y_max = bbox_max
    nx = max(1, int(np.ceil((x_max - x_min) / voxel_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / voxel_size)))

    def _hist(pts):
        ix = np.clip(((pts[:, 0] - x_min) / voxel_size).astype(np.int64), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - y_min) / voxel_size).astype(np.int64), 0, ny - 1)
        flat = ix * ny + iy
        h = np.bincount(flat, minlength=nx * ny).astype(np.float64)
        s = h.sum()
        if s <= 0:
            return None
        return h / s

    p = _hist(pred)
    q = _hist(gt)
    if p is None or q is None:
        return float("nan")

    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(b[mask]))))

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(jsd)


def f_score(pred_pts, gt_pts, threshold: float = 0.2) -> Dict[str, float]:
    """
    F-score at a distance threshold (meters).

    Precision: fraction of predicted points whose nearest GT neighbor is <= threshold.
    Recall:    fraction of GT points whose nearest predicted neighbor is <= threshold.
    F-score:   harmonic mean of precision and recall.

    Returns dict with keys: precision, recall, f_score.
    """
    pred = _as_np(pred_pts)
    gt = _as_np(gt_pts)
    if len(pred) == 0 or len(gt) == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)

    d_p2g, _ = gt_tree.query(pred, k=1)
    d_g2p, _ = pred_tree.query(gt, k=1)

    precision = float((d_p2g <= threshold).mean())
    recall = float((d_g2p <= threshold).mean())
    if precision + recall == 0:
        f = 0.0
    else:
        f = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f}


def voxel_iou(pred_pts, gt_pts, voxel_size: float = 0.2) -> float:
    """
    IoU of occupied voxels between two point clouds.

    Voxelizes both clouds with a shared origin (min of joint support) at
    voxel_size resolution, then IoU = |pred ∩ gt| / |pred ∪ gt|.

    This bridges to SSC literature: MonoScene (11.1 mIoU), VoxFormer (13.4),
    CGFormer (16.6) — though their IoU is semantic per-class; here we report
    geometric occupancy IoU.
    """
    pred = _as_np(pred_pts)
    gt = _as_np(gt_pts)
    if len(pred) == 0 or len(gt) == 0:
        return 0.0

    joint = np.concatenate([pred, gt], axis=0)
    origin = joint.min(axis=0)

    def _vox(pts):
        v = np.floor((pts - origin) / voxel_size).astype(np.int64)
        # Pack to a single int64 key: assume bounded coordinates
        # Use a hashable per-row view via tobytes
        return {tuple(row) for row in v}

    set_p = _vox(pred)
    set_g = _vox(gt)

    inter = len(set_p & set_g)
    union = len(set_p | set_g)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def hausdorff_95(pred_pts, gt_pts) -> float:
    """
    Hausdorff-95 distance: max of the 95th-percentile directed distances
    pred->gt and gt->pred.

    This is the worst-case geometric error (with outliers trimmed), a standard
    robustness metric. Units: meters.
    """
    pred = _as_np(pred_pts)
    gt = _as_np(gt_pts)
    if len(pred) == 0 or len(gt) == 0:
        return float("nan")

    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    d_p2g, _ = gt_tree.query(pred, k=1)
    d_g2p, _ = pred_tree.query(gt, k=1)

    h95_p2g = float(np.percentile(d_p2g, 95))
    h95_g2p = float(np.percentile(d_g2p, 95))
    return max(h95_p2g, h95_g2p)


def chamfer_distance_np(pred_pts, gt_pts) -> float:
    """
    Symmetric Chamfer distance (mean of directed mean Euclidean distances), scipy.
    Linear (meters), provided for offline / CPU-only metrics computation.
    """
    pred = _as_np(pred_pts)
    gt = _as_np(gt_pts)
    if len(pred) == 0 or len(gt) == 0:
        return float("nan")
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    d_p2g, _ = gt_tree.query(pred, k=1)
    d_g2p, _ = pred_tree.query(gt, k=1)
    return float((d_p2g.mean() + d_g2p.mean()) / 2.0)


def chamfer_distance_sq_np(pred_pts, gt_pts) -> float:
    """
    Legacy squared Chamfer distance matching models.refinement_net.chamfer_distance.
    Returned in m^2. Use this to compare against previously reported numbers
    (e.g., teacher v2GT CD 0.039 was squared).
    """
    pred = _as_np(pred_pts)
    gt = _as_np(gt_pts)
    if len(pred) == 0 or len(gt) == 0:
        return float("nan")
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)
    d_p2g, _ = gt_tree.query(pred, k=1)
    d_g2p, _ = pred_tree.query(gt, k=1)
    return float(((d_p2g ** 2).mean() + (d_g2p ** 2).mean()) / 2.0)


def compute_all_metrics(
    pred_pts,
    gt_pts,
    f_thresholds: Tuple[float, ...] = (0.1, 0.2),
    iou_voxel_sizes: Tuple[float, ...] = (0.1, 0.2),
    jsd_voxel_size: float = 0.5,
    jsd_bbox: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Compute the full RA-L metric suite for a single prediction/GT pair.

    Returns a flat dict with keys:
        cd, jsd, f_score@0.1, f_score@0.2, precision@0.1, precision@0.2,
        recall@0.1, recall@0.2, iou@0.1, iou@0.2, hausdorff_95
    """
    out: Dict[str, float] = {}
    out["cd"] = chamfer_distance_np(pred_pts, gt_pts)
    out["cd_sq"] = chamfer_distance_sq_np(pred_pts, gt_pts)

    if jsd_bbox is None:
        out["jsd"] = jsd_bev(pred_pts, gt_pts, voxel_size=jsd_voxel_size)
    else:
        out["jsd"] = jsd_bev(
            pred_pts, gt_pts, voxel_size=jsd_voxel_size,
            bbox_min=jsd_bbox[0], bbox_max=jsd_bbox[1],
        )

    for t in f_thresholds:
        fs = f_score(pred_pts, gt_pts, threshold=t)
        out[f"precision@{t}"] = fs["precision"]
        out[f"recall@{t}"] = fs["recall"]
        out[f"f_score@{t}"] = fs["f_score"]

    for v in iou_voxel_sizes:
        out[f"iou@{v}"] = voxel_iou(pred_pts, gt_pts, voxel_size=v)

    out["hausdorff_95"] = hausdorff_95(pred_pts, gt_pts)
    return out


if __name__ == "__main__":
    # Test metrics
    print("Testing Completion Metrics...")
    
    # Create dummy data
    gt_points = torch.randn(10000, 3)
    
    # Perfect prediction
    pred_points = gt_points + torch.randn_like(gt_points) * 0.001
    
    metrics = CompletionMetrics()
    metrics.update(pred_points, gt_points)
    
    results = metrics.compute()
    print("\nPerfect prediction:")
    print(metrics)
    
    # Imperfect prediction
    pred_points = gt_points + torch.randn_like(gt_points) * 0.1
    
    metrics = CompletionMetrics()
    metrics.update(pred_points, gt_points)
    
    print("\nImperfect prediction:")
    print(metrics)
    
    # With semantic labels
    gt_labels = torch.randint(0, 20, (10000,))
    pred_labels = gt_labels.clone()
    # Add some errors
    error_mask = torch.rand(10000) < 0.1
    pred_labels[error_mask] = torch.randint(0, 20, (error_mask.sum(),))
    
    metrics = CompletionMetrics()
    metrics.update(pred_points, gt_points, pred_labels, gt_labels)

    print("\nWith semantic labels:")
    print(metrics)

    # -----------------------------------------------------------------
    # Sanity checks for the RA-L metric suite
    # -----------------------------------------------------------------
    print("\n=== RA-L metrics sanity checks ===")
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 20, size=(5000, 3)).astype(np.float32)

    print("\n[case 1] identical clouds (expect CD~0, JSD~0, F~1, IoU~1, H95~0):")
    m_id = compute_all_metrics(pts, pts)
    for k, v in m_id.items():
        print(f"  {k}: {v:.6f}")
    assert m_id["cd"] < 1e-6
    assert m_id["jsd"] < 1e-6
    assert m_id["f_score@0.1"] > 0.99
    assert m_id["iou@0.1"] > 0.99
    assert m_id["hausdorff_95"] < 1e-6

    print("\n[case 2] offset by 1m along x (expect CD~1, F@0.1~0, F@0.2~0, H95~1):")
    m_off = compute_all_metrics(pts + np.array([1.0, 0.0, 0.0], np.float32), pts)
    for k, v in m_off.items():
        print(f"  {k}: {v:.6f}")
    assert 0.9 < m_off["cd"] < 1.1
    assert m_off["f_score@0.1"] < 0.05
    assert m_off["f_score@0.2"] < 0.05
    assert 0.9 < m_off["hausdorff_95"] < 1.1

    print("\n[case 3] offset by 0.05m (expect F@0.1~1, F@0.2~1, IoU@0.2 still high):")
    m_small = compute_all_metrics(
        pts + np.array([0.05, 0.0, 0.0], np.float32), pts
    )
    for k, v in m_small.items():
        print(f"  {k}: {v:.6f}")
    assert m_small["f_score@0.1"] > 0.9
    assert m_small["f_score@0.2"] > 0.95

    print("\nAll sanity checks passed.")
