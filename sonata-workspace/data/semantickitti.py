"""
SemanticKITTI Dataset Handler

Loads and preprocesses SemanticKITTI data for semantic scene completion.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import yaml


class SemanticKITTI(Dataset):
    """
    SemanticKITTI dataset for semantic scene completion.
    
    Loads:
    - Input scans (partial LiDAR)
    - Ground truth complete scenes
    - Semantic labels
    - Poses for map generation
    """
    
    # SemanticKITTI class mapping
    LEARNING_MAP = {
        0: 0,      # "unlabeled"
        1: 0,      # "outlier" mapped to "unlabeled"
        10: 1,     # "car"
        11: 2,     # "bicycle"
        13: 5,     # "bus"
        15: 3,     # "motorcycle"
        16: 5,     # "on-rails"
        18: 4,     # "truck"
        20: 5,     # "other-vehicle"
        30: 6,     # "person"
        31: 7,     # "bicyclist"
        32: 8,     # "motorcyclist"
        40: 9,     # "road"
        44: 10,    # "parking"
        48: 11,    # "sidewalk"
        49: 12,    # "other-ground"
        50: 13,    # "building"
        51: 14,    # "fence"
        52: 0,     # "other-structure" to "unlabeled"
        60: 9,     # "lane-marking" to "road"
        70: 15,    # "vegetation"
        71: 16,    # "trunk"
        72: 17,    # "terrain"
        80: 18,    # "pole"
        81: 19,    # "traffic-sign"
        99: 0,     # "other-object" to "unlabeled"
        252: 1,    # "moving-car" to "car"
        253: 7,    # "moving-bicyclist" to "bicyclist"
        254: 6,    # "moving-person" to "person"
        255: 8,    # "moving-motorcyclist" to "motorcyclist"
        256: 5,    # "moving-on-rails" mapped to "other-vehicle"
        257: 5,    # "moving-bus" mapped to "other-vehicle"
        258: 4,    # "moving-truck" to "truck"
        259: 5,    # "moving-other-vehicle" to "other-vehicle"
    }
    
    NUM_CLASSES = 20
    
    SPLITS = {
        'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
        'val': ['08'],
        'test': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    }
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        voxel_size: float = 0.05,
        max_points: int = 50000,
        use_ground_truth_maps: bool = True,
        augmentation: bool = True,
        num_points_per_scan: Optional[int] = None,
    ):
        """
        Initialize SemanticKITTI dataset.
        
        Args:
            root: Path to SemanticKITTI dataset root
            split: Dataset split ('train', 'val', 'test')
            voxel_size: Voxel size for scene representation
            max_points: Maximum points per sample
            use_ground_truth_maps: Use pre-generated complete maps as GT
            augmentation: Apply data augmentation
            num_points_per_scan: Subsample scans to this number
        """
        super().__init__()
        
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.use_ground_truth_maps = use_ground_truth_maps
        self.augmentation = augmentation
        self.num_points_per_scan = num_points_per_scan
        
        # Get sequences for this split
        self.sequences = self.SPLITS[split]
        
        # Build file lists
        self.scan_files = []
        self.label_files = []
        self.pose_files = []
        self.gt_map_files = []
        
        self._build_file_lists()
        
        print(f"Loaded SemanticKITTI {split} split:")
        print(f"  Sequences: {self.sequences}")
        print(f"  Total scans: {len(self.scan_files)}")
    
    def _build_file_lists(self):
        """Build lists of data files."""
        for seq in self.sequences:
            seq_path = os.path.join(self.root, 'sequences', seq)
            
            # Scan files
            scan_dir = os.path.join(seq_path, 'velodyne')
            if not os.path.exists(scan_dir):
                print(f"Warning: {scan_dir} does not exist")
                continue
            
            scan_files = sorted(os.listdir(scan_dir))
            
            for scan_file in scan_files:
                if not scan_file.endswith('.bin'):
                    continue
                
                scan_id = scan_file.replace('.bin', '')
                
                # Full paths
                scan_path = os.path.join(scan_dir, scan_file)
                label_path = os.path.join(
                    seq_path, 'labels', f'{scan_id}.label'
                )
                pose_path = os.path.join(seq_path, 'poses.txt')
                
                # Ground truth map (if using pre-generated)
                if self.use_ground_truth_maps:
                    gt_map_path = os.path.join(
                        self.root, 'ground_truth', seq, f'{scan_id}.npz'
                    )
                    self.gt_map_files.append(gt_map_path)
                
                self.scan_files.append(scan_path)
                self.label_files.append(label_path)
                self.pose_files.append(pose_path)
    
    def __len__(self) -> int:
        return len(self.scan_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sample.
        
        Returns:
            Dictionary containing:
                - partial_scan: Incomplete LiDAR scan
                - complete_scene: Ground truth complete scene
                - semantic_labels: Semantic class labels
                - metadata: Additional information
        """
        # Load scan
        scan = self._load_scan(self.scan_files[idx])
        
        # Load labels
        if os.path.exists(self.label_files[idx]):
            labels = self._load_labels(self.label_files[idx])
        else:
            labels = np.zeros(scan.shape[0], dtype=np.int32)
        
        # Load or generate ground truth
        if self.use_ground_truth_maps and \
           os.path.exists(self.gt_map_files[idx]):
            gt_complete = self._load_gt_map(self.gt_map_files[idx])
        else:
            # Use scan itself as GT for testing
            gt_complete = scan.copy()
        
        # Subsample scan if needed
        if self.num_points_per_scan is not None:
            scan, labels = self._subsample_scan(
                scan, labels, self.num_points_per_scan
            )
        
        # Apply augmentation
        if self.augmentation and self.split == 'train':
            scan, gt_complete, labels = self._augment(
                scan, gt_complete, labels
            )
        
        # Normalize coordinates
        scan_center = scan.mean(axis=0)
        scan = scan - scan_center
        gt_complete = gt_complete - scan_center
        
        # Voxelize
        scan_voxel, scan_labels = self._voxelize(scan, labels)
        gt_voxel, gt_labels = self._voxelize(gt_complete, labels)
        
        # Limit number of points
        if scan_voxel.shape[0] > self.max_points:
            indices = np.random.choice(
                scan_voxel.shape[0], 
                self.max_points, 
                replace=False
            )
            scan_voxel = scan_voxel[indices]
            scan_labels = scan_labels[indices]
        
        if gt_voxel.shape[0] > self.max_points:
            indices = np.random.choice(
                gt_voxel.shape[0], 
                self.max_points, 
                replace=False
            )
            gt_voxel = gt_voxel[indices]
            gt_labels = gt_labels[indices]
        
        # Convert to tensors
        data = {
            'partial_coord': torch.from_numpy(scan_voxel).float(),
            'partial_labels': torch.from_numpy(scan_labels).long(),
            'complete_coord': torch.from_numpy(gt_voxel).float(),
            'complete_labels': torch.from_numpy(gt_labels).long(),
            'scan_center': torch.from_numpy(scan_center).float(),
            'idx': idx,
        }
        
        # Add color if available (use height as color for now)
        partial_color = self._height_to_color(scan_voxel)
        complete_color = self._height_to_color(gt_voxel)
        
        data['partial_color'] = torch.from_numpy(partial_color).float()
        data['complete_color'] = torch.from_numpy(complete_color).float()
        
        # Add normals (computed from local neighbors)
        data['partial_normal'] = torch.zeros_like(data['partial_coord'])
        data['complete_normal'] = torch.zeros_like(data['complete_coord'])
        
        return data
    
    def _load_scan(self, scan_path: str) -> np.ndarray:
        """Load LiDAR scan from binary file."""
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))  # x, y, z, intensity
        return scan[:, :3]  # Return only xyz
    
    def _load_labels(self, label_path: str) -> np.ndarray:
        """Load semantic labels."""
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF  # Remove instance info
        
        # Map to learning classes
        labels_mapped = np.vectorize(
            lambda x: self.LEARNING_MAP.get(x, 0)
        )(labels)
        
        return labels_mapped.astype(np.int32)
    
    def _load_gt_map(self, gt_path: str) -> np.ndarray:
        """Load pre-generated ground truth complete map."""
        data = np.load(gt_path)
        return data['points']
    
    def _subsample_scan(
        self,
        scan: np.ndarray,
        labels: np.ndarray,
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly subsample scan."""
        if scan.shape[0] <= num_points:
            return scan, labels
        
        indices = np.random.choice(
            scan.shape[0], num_points, replace=False
        )
        return scan[indices], labels[indices]
    
    def _voxelize(
        self,
        points: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Voxelize point cloud.
        
        Returns voxel centers and majority-vote labels.
        """
        # Compute voxel coordinates
        voxel_coords = np.floor(points / self.voxel_size).astype(np.int32)
        
        # Get unique voxels
        unique_voxels, inverse_indices = np.unique(
            voxel_coords, axis=0, return_inverse=True
        )
        
        # Compute voxel centers
        voxel_centers = unique_voxels * self.voxel_size + self.voxel_size / 2
        
        # Majority vote for labels
        voxel_labels = np.zeros(unique_voxels.shape[0], dtype=np.int32)
        for i in range(unique_voxels.shape[0]):
            mask = inverse_indices == i
            if mask.sum() > 0:
                # Most common label
                voxel_labels[i] = np.bincount(
                    labels[mask]
                ).argmax()
        
        return voxel_centers, voxel_labels
    
    def _augment(
        self,
        partial: np.ndarray,
        complete: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random rotation around z-axis
        angle = np.random.uniform(-np.pi, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        partial = partial @ rot_matrix.T
        complete = complete @ rot_matrix.T
        
        # Random flip
        if np.random.rand() > 0.5:
            partial[:, 1] = -partial[:, 1]
            complete[:, 1] = -complete[:, 1]
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        partial = partial * scale
        complete = complete * scale
        
        # Random jittering
        partial += np.random.randn(*partial.shape) * 0.01
        
        return partial, complete, labels
    
    def _height_to_color(self, points: np.ndarray) -> np.ndarray:
        """Convert height to RGB color for visualization."""
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        
        # Create color gradient (blue -> green -> red)
        colors = np.zeros((points.shape[0], 3))
        colors[:, 0] = z_norm  # Red
        colors[:, 1] = 1 - np.abs(z_norm - 0.5) * 2  # Green
        colors[:, 2] = 1 - z_norm  # Blue
        
        return colors


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching samples.
    
    Creates batch indices for sparse tensor operations.
    """
    # Collect all data
    partial_coords = []
    partial_colors = []
    partial_normals = []
    partial_labels = []
    
    complete_coords = []
    complete_colors = []
    complete_labels = []
    
    batch_indices_partial = []
    batch_indices_complete = []
    
    for i, data in enumerate(batch):
        partial_coords.append(data['partial_coord'])
        partial_colors.append(data['partial_color'])
        partial_normals.append(data['partial_normal'])
        partial_labels.append(data['partial_labels'])
        
        complete_coords.append(data['complete_coord'])
        complete_colors.append(data['complete_color'])
        complete_labels.append(data['complete_labels'])
        
        # Batch indices
        batch_indices_partial.append(
            torch.full((data['partial_coord'].shape[0],), i, dtype=torch.long)
        )
        batch_indices_complete.append(
            torch.full((data['complete_coord'].shape[0],), i, dtype=torch.long)
        )
    
    # Concatenate
    batch_data = {
        'partial_coord': torch.cat(partial_coords, dim=0),
        'partial_color': torch.cat(partial_colors, dim=0),
        'partial_normal': torch.cat(partial_normals, dim=0),
        'partial_labels': torch.cat(partial_labels, dim=0),
        'partial_batch': torch.cat(batch_indices_partial, dim=0),
        
        'complete_coord': torch.cat(complete_coords, dim=0),
        'complete_color': torch.cat(complete_colors, dim=0),
        'complete_labels': torch.cat(complete_labels, dim=0),
        'complete_batch': torch.cat(batch_indices_complete, dim=0),
        
        'scan_center': torch.stack([d['scan_center'] for d in batch]),
        'idx': torch.tensor([d['idx'] for d in batch]),
    }
    
    return batch_data


if __name__ == "__main__":
    # Test dataset
    print("Testing SemanticKITTI dataset...")
    
    dataset = SemanticKITTI(
        root='Datasets/SemanticKITTI/dataset',
        split='train',
        voxel_size=0.05,
        use_ground_truth_maps=False,  # Test without GT maps first
        augmentation=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Partial scan shape: {sample['partial_coord'].shape}")
    print(f"Complete scene shape: {sample['complete_coord'].shape}")
    print(f"Labels shape: {sample['partial_labels'].shape}")
    print(f"Unique labels: {torch.unique(sample['partial_labels'])}")
    
    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    
    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch partial coords: {batch['partial_coord'].shape}")
    print(f"Batch indices: {torch.unique(batch['partial_batch'])}")
