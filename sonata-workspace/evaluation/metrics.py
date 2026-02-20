"""
Evaluation Metrics for Semantic Scene Completion

Implements standard SSC metrics:
- Completion ratio at different thresholds
- Chamfer distance
- Semantic IoU
- Mean IoU across classes
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
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
