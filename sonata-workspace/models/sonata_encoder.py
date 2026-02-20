"""
Sonata Encoder Wrapper for Semantic Scene Completion

Integrates the pre-trained Sonata (Point Transformer V3) encoder
with the LiDiff diffusion framework for scene completion.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download
import pickle


class SonataEncoder(nn.Module):
    """
    Wrapper for Sonata encoder with hierarchical feature extraction.
    
    The encoder provides multi-scale point features from 5 hierarchical levels,
    which are used as conditional information for the diffusion process.
    """
    
    def __init__(
        self,
        pretrained: str = "facebook/sonata",
        freeze: bool = True,
        enable_flash: bool = True,
        custom_config: Optional[Dict] = None,
        feature_levels: List[int] = [0, 1, 2, 3, 4],
    ):
        """
        Initialize Sonata encoder.
        
        Args:
            pretrained: Hugging Face model ID or local path
            freeze: Whether to freeze encoder weights
            enable_flash: Enable flash attention (requires installation)
            custom_config: Custom configuration overrides
            feature_levels: Which hierarchical levels to extract features from
        """
        super().__init__()
        
        self.feature_levels = feature_levels
        self.freeze = freeze
        
        # Load pre-trained Sonata model
        self.encoder = self._load_pretrained(
            pretrained, enable_flash, custom_config
        )
        
        # Feature dimensions at each level
        self.feature_dims = {
            0: 384,  # Input level
            1: 384,  # After first pooling
            2: 384,  # After second pooling
            3: 384,  # After third pooling
            4: 384,  # After fourth pooling
        }
        
        if freeze:
            self._freeze_encoder()
        
        # Feature projection layers for each level
        self.projections = nn.ModuleDict({
            str(level): nn.Linear(self.feature_dims[level], 256)
            for level in feature_levels
        })
        
    def _load_pretrained(
        self, 
        model_id: str,
        enable_flash: bool,
        custom_config: Optional[Dict]
    ):
        """Load pre-trained Sonata model."""
        try:
            # Try loading from Hugging Face
            if custom_config is None:
                custom_config = {}
            
            if not enable_flash:
                custom_config['enable_flash'] = False
                custom_config['enc_patch_size'] = [512] * 5  # Reduce patch size
            
            # Import Sonata model
            try:
                from sonata.model import PointTransformerV3
                model = PointTransformerV3.from_pretrained(
                    model_id, **custom_config
                )
            except ImportError:
                print("Sonata not installed as package, trying direct import...")
                # Add fallback for standalone mode
                import sys
                sys.path.insert(0, './sonata')
                from model import PointTransformerV3
                model = PointTransformerV3.from_pretrained(
                    model_id, **custom_config
                )
                
            return model
            
        except Exception as e:
            print(f"Error loading Sonata model: {e}")
            raise
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Sonata encoder frozen.")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze = False
        print("Sonata encoder unfrozen for fine-tuning.")
    
    def forward(
        self, 
        point_dict: Dict[str, torch.Tensor],
        return_all_levels: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Sonata encoder.
        
        Args:
            point_dict: Dictionary containing point cloud data
                - coord: (N, 3) point coordinates
                - color: (N, 3) RGB colors
                - normal: (N, 3) surface normals
                - grid_coord: (N, 3) grid coordinates
                - batch: (N,) batch indices (optional)
            return_all_levels: Return features from all hierarchical levels
            
        Returns:
            Dictionary containing:
                - features: Multi-level point features
                - coords: Coordinates at each level
                - point: Final encoded point cloud
        """
        # Encode point cloud through hierarchical levels
        with torch.set_grad_enabled(not self.freeze):
            point = self.encoder(point_dict)
        
        if not return_all_levels:
            # Return only final level features
            return {
                'features': {4: point.feat},
                'coords': {4: point.coord},
                'point': point
            }
        
        # Collect features from all hierarchical levels
        features = {}
        coords = {}
        
        # Navigate back through pooling hierarchy
        current_point = point
        level = 4
        
        while level >= 0:
            if level in self.feature_levels:
                # Project features to common dimension
                projected_feat = self.projections[str(level)](current_point.feat)
                features[level] = projected_feat
                coords[level] = current_point.coord
            
            # Move to parent level if available
            if hasattr(current_point, 'pooling_parent') and level > 0:
                current_point = current_point.pooling_parent
                level -= 1
            else:
                break
        
        return {
            'features': features,
            'coords': coords,
            'point': point
        }
    
    def get_feature_dim(self, level: int = 4) -> int:
        """Get feature dimension at specific level."""
        return 256  # After projection
    
    def map_features_to_original(
        self,
        features: torch.Tensor,
        point_dict: Dict[str, torch.Tensor],
        inverse: torch.Tensor
    ) -> torch.Tensor:
        """
        Map hierarchical features back to original point cloud.
        
        Args:
            features: Features at encoded level
            point_dict: Original point cloud dictionary
            inverse: Inverse mapping from grid sampling
            
        Returns:
            Features mapped to original point cloud
        """
        # Map from encoded level back through hierarchy
        mapped_feat = features[inverse]
        return mapped_feat


class ConditionalFeatureExtractor(nn.Module):
    """
    Extracts and processes conditional features from Sonata encoder
    for the diffusion process.
    """
    
    def __init__(
        self,
        encoder: SonataEncoder,
        feature_levels: List[int] = [2, 3, 4],
        fusion_type: str = "concat"  # "concat", "attention", "hierarchical"
    ):
        """
        Initialize conditional feature extractor.
        
        Args:
            encoder: Sonata encoder instance
            feature_levels: Which levels to use for conditioning
            fusion_type: How to fuse multi-level features
        """
        super().__init__()
        
        self.encoder = encoder
        self.feature_levels = feature_levels
        self.fusion_type = fusion_type
        
        # Feature fusion
        if fusion_type == "concat":
            self.out_dim = len(feature_levels) * 256
        elif fusion_type == "attention":
            self.attention_fusion = MultiLevelAttentionFusion(
                feature_levels, feature_dim=256
            )
            self.out_dim = 256
        elif fusion_type == "hierarchical":
            self.hierarchical_fusion = HierarchicalFusion(
                feature_levels, feature_dim=256
            )
            self.out_dim = 256
    
    def forward(
        self,
        point_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Extract conditional features.
        
        Returns:
            - Fused conditional features
            - Feature dictionary with all levels
        """
        # Get multi-level features from encoder
        encoder_output = self.encoder(point_dict, return_all_levels=True)
        features = encoder_output['features']
        
        # Select desired levels
        selected_features = [
            features[level] for level in self.feature_levels
            if level in features
        ]
        
        # Fuse features
        if self.fusion_type == "concat":
            # Simple concatenation
            fused = torch.cat(selected_features, dim=-1)
        elif self.fusion_type == "attention":
            # Attention-based fusion
            fused = self.attention_fusion(selected_features)
        elif self.fusion_type == "hierarchical":
            # Hierarchical fusion with skip connections
            fused = self.hierarchical_fusion(selected_features)
        
        return fused, encoder_output


class MultiLevelAttentionFusion(nn.Module):
    """Attention-based fusion of multi-level features."""
    
    def __init__(self, feature_levels: List[int], feature_dim: int = 256):
        super().__init__()
        
        num_levels = len(feature_levels)
        
        # Learnable attention weights for each level
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * num_levels, num_levels),
            nn.Softmax(dim=-1)
        )
        
        self.projection = nn.Linear(feature_dim * num_levels, feature_dim)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-level features with attention.
        
        Args:
            features: List of feature tensors from different levels
            
        Returns:
            Fused features
        """
        # Concatenate all features
        concat_feat = torch.cat(features, dim=-1)
        
        # Compute attention weights
        weights = self.attention(concat_feat)  # (N, num_levels)
        
        # Weighted sum
        stacked = torch.stack(features, dim=1)  # (N, num_levels, D)
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (N, D)
        
        # Final projection
        fused = self.projection(torch.cat([weighted, concat_feat], dim=-1))
        
        return fused


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion with progressive refinement."""
    
    def __init__(self, feature_levels: List[int], feature_dim: int = 256):
        super().__init__()
        
        self.num_levels = len(feature_levels)
        
        # Progressive fusion blocks
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
            for _ in range(self.num_levels - 1)
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Hierarchically fuse features from coarse to fine.
        
        Args:
            features: List of features [coarse -> fine]
            
        Returns:
            Fused features
        """
        # Start with coarsest level
        fused = features[0]
        
        # Progressively refine with finer levels
        for i in range(1, len(features)):
            fused = self.fusion_blocks[i-1](
                torch.cat([fused, features[i]], dim=-1)
            )
        
        return fused


if __name__ == "__main__":
    # Test encoder
    print("Testing Sonata Encoder...")
    
    # Create dummy input
    batch_size = 2
    num_points = 4096
    
    point_dict = {
        'coord': torch.randn(num_points * batch_size, 3).cuda(),
        'color': torch.randn(num_points * batch_size, 3).cuda(),
        'normal': torch.randn(num_points * batch_size, 3).cuda(),
        'grid_coord': torch.randn(num_points * batch_size, 3).cuda(),
        'batch': torch.cat([
            torch.full((num_points,), i) for i in range(batch_size)
        ]).cuda(),
    }
    
    # Initialize encoder
    encoder = SonataEncoder(
        pretrained="facebook/sonata",
        freeze=True,
        enable_flash=False  # Disable for testing
    ).cuda()
    
    # Forward pass
    output = encoder(point_dict, return_all_levels=True)
    
    print(f"Encoder output keys: {output.keys()}")
    print(f"Feature levels: {output['features'].keys()}")
    for level, feat in output['features'].items():
        print(f"  Level {level}: {feat.shape}")
    
    # Test conditional feature extractor
    cond_extractor = ConditionalFeatureExtractor(
        encoder, 
        feature_levels=[2, 3, 4],
        fusion_type="attention"
    ).cuda()
    
    cond_features, encoder_out = cond_extractor(point_dict)
    print(f"\nConditional features shape: {cond_features.shape}")
    print(f"Output dimension: {cond_extractor.out_dim}")
