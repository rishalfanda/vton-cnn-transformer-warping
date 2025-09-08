"""
Warping Module untuk Virtual Try-On
Bab IV 4.3.4 - Geometric Transformation Module
Implementasi TPS, Flow-based, dan Neural Warping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class TPSWarping(nn.Module):
    """
    Thin-Plate Spline Warping
    Bab IV 4.3.4.1 - TPS-based Geometric Transformation
    """
    def __init__(self, grid_size: int = 5, reg_factor: float = 0.0):
        super().__init__()
        self.grid_size = grid_size
        self.reg_factor = reg_factor
        
        # Create regular grid
        self.grid = self._create_grid(grid_size)
    
    def _create_grid(self, size: int) -> torch.Tensor:
        """Create regular grid points"""
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return grid
    
    def _compute_distance(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between points"""
        diff = p1.unsqueeze(1) - p2.unsqueeze(0)
        dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-6)
        return dist
    
    def _solve_system(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> torch.Tensor:
        """Solve TPS interpolation system"""
        B, N, _ = source_points.shape
        
        # Compute kernel matrix K
        K = self._compute_distance(source_points, source_points)
        K = K ** 2 * torch.log(K + 1e-6)
        
        # Add regularization
        K = K + self.reg_factor * torch.eye(N, device=K.device)
        
        # Build system matrix
        P = torch.cat([
            torch.ones(B, N, 1, device=source_points.device),
            source_points
        ], dim=2)
        
        # Zero padding
        zeros = torch.zeros(B, 3, 3, device=source_points.device)
        
        # Full system matrix
        A_upper = torch.cat([K, P], dim=2)
        A_lower = torch.cat([P.transpose(1, 2), zeros], dim=2)
        A = torch.cat([A_upper, A_lower], dim=1)
        
        # Right hand side
        b = torch.cat([target_points, torch.zeros(B, 3, 2, device=target_points.device)], dim=1)
        
        # Solve system
        weights = torch.linalg.solve(A, b)
        
        return weights
    
    def forward(
        self,
        image: torch.Tensor,
        source_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply TPS warping to image
        
        Args:
            image: (B, C, H, W) image to warp
            source_points: (B, N, 2) source control points
            target_points: (B, N, 2) target control points
        """
        B, C, H, W = image.shape
        
        # Solve TPS system
        weights = self._solve_system(source_points, target_points)
        
        # Create dense grid
        grid_x = torch.linspace(-1, 1, W, device=image.device)
        grid_y = torch.linspace(-1, 1, H, device=image.device)
        mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing='xy')
        grid = torch.stack([mesh_x, mesh_y], dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Compute TPS transformation for each grid point
        grid_flat = grid.view(B, -1, 2)
        
        # Compute distances to control points
        dist = self._compute_distance(grid_flat, source_points)
        K = dist ** 2 * torch.log(dist + 1e-6)
        
        # Apply weights
        P = torch.cat([
            torch.ones(B, H*W, 1, device=image.device),
            grid_flat
        ], dim=2)
        
        features = torch.cat([K, P], dim=2)
        warped_grid = torch.matmul(features, weights[:, :, :2])
        warped_grid = warped_grid.view(B, H, W, 2)
        
        # Sample image with warped grid
        warped_image = F.grid_sample(
            image,
            warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped_image

class FlowWarping(nn.Module):
    """
    Flow-based Warping Network
    Bab IV 4.3.4.2 - Optical Flow Warping
    """
    def __init__(
        self,
        in_channels: int = 6,  # concatenated source and target
        hidden_channels: int = 128,
        refinement_stages: int = 3
    ):
        super().__init__()
        self.refinement_stages = refinement_stages
        
        # Flow estimation network
        self.flow_estimator = nn.ModuleList()
        
        for stage in range(refinement_stages):
            stage_in_channels = in_channels if stage == 0 else in_channels + 2  # Add previous flow
            
            self.flow_estimator.append(
                nn.Sequential(
                    # Encoder
                    nn.Conv2d(stage_in_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(hidden_channels, hidden_channels*2, 3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_channels*2),
                    nn.ReLU(inplace=True),
                    
                    # Bottleneck
                    nn.Conv2d(hidden_channels*2, hidden_channels*2, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels*2),
                    nn.ReLU(inplace=True),
                    
                    # Decoder
                    nn.ConvTranspose2d(hidden_channels*2, hidden_channels, 4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(hidden_channels, hidden_channels//2, 4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_channels//2),
                    nn.ReLU(inplace=True),
                    
                    # Flow prediction
                    nn.Conv2d(hidden_channels//2, 2, 3, padding=1)
                )
            )
    
    def warp_image(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp image using flow field"""
        B, C, H, W = image.shape
        
        # Create identity grid
        grid_x = torch.linspace(-1, 1, W, device=image.device)
        grid_y = torch.linspace(-1, 1, H, device=image.device)
        mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing='xy')
        grid = torch.stack([mesh_x, mesh_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Normalize flow to [-1, 1] range
        flow_norm = flow.clone()
        flow_norm[:, 0] = flow_norm[:, 0] / (W / 2)
        flow_norm[:, 1] = flow_norm[:, 1] / (H / 2)
        
        # Apply flow to grid
        warped_grid = grid + flow_norm
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Sample image
        warped = F.grid_sample(
            image,
            warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped
    
    def forward(
        self,
        source: torch.Tensor,
        target_shape: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate flow and warp source image
        
        Args:
            source: source cloth image
            target_shape: target body shape/mask
        
        Returns:
            warped: warped source image
            flow: estimated flow field
        """
        B, C, H, W = source.shape
        
        # Concatenate inputs
        x = torch.cat([source, target_shape], dim=1)
        
        # Multi-stage refinement
        flow = None
        for stage, estimator in enumerate(self.flow_estimator):
            if stage > 0 and flow is not None:
                # Warp source with current flow
                warped_source = self.warp_image(source, flow)
                x = torch.cat([warped_source, target_shape, flow], dim=1)
            
            # Estimate flow refinement
            flow_delta = estimator(x)
            
            # Accumulate flow
            if flow is None:
                flow = flow_delta
            else:
                # Upsample previous flow if needed
                if flow.shape[2:] != flow_delta.shape[2:]:
                    flow = F.interpolate(
                        flow,
                        size=flow_delta.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                flow = flow + flow_delta
        
        # Final warping
        warped = self.warp_image(source, flow)
        
        return warped, flow

class NeuralWarping(nn.Module):
    """
    Neural Warping dengan Feedback dari Segmentasi
    Bab IV 4.3.4.3 - Advanced Neural Warping
    """
    def __init__(
        self,
        segmentation_channels: int = 20,
        cloth_channels: int = 3,
        hidden_channels: int = 128,
        use_feedback: bool = True
    ):
        super().__init__()
        self.use_feedback = use_feedback
        
        # Feature extraction
        self.cloth_encoder = nn.Sequential(
            nn.Conv2d(cloth_channels, hidden_channels//2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//2, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.segmentation_encoder = nn.Sequential(
            nn.Conv2d(segmentation_channels, hidden_channels//2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//2, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Correlation layer
        self.correlation = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Warping decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels//2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels//2, 2, 3, padding=1)  # Output flow
        )
        
        # Feedback refinement network
        if use_feedback:
            self.feedback_refine = nn.Sequential(
                nn.Conv2d(cloth_channels + segmentation_channels + 2, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 2, 3, padding=1)
            )
    
    def forward(
        self,
        cloth: torch.Tensor,
        segmentation: torch.Tensor,
        pose: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Neural warping with segmentation feedback
        
        Args:
            cloth: source cloth image
            segmentation: target body segmentation
            pose: optional pose information
        
        Returns:
            warped: warped cloth
            flow: transformation flow
        """
        # Extract features
        cloth_feat = self.cloth_encoder(cloth)
        seg_feat = self.segmentation_encoder(segmentation)
        
        # Correlate features
        combined = torch.cat([cloth_feat, seg_feat], dim=1)
        corr = self.correlation(combined)
        
        # Decode to flow
        flow = self.decoder(corr)
        
        # Warp cloth
        warped = self._warp_with_flow(cloth, flow)
        
        # Feedback refinement
        if self.use_feedback and self.training:
            # Concatenate warped result with segmentation
            feedback_input = torch.cat([warped, segmentation, flow], dim=1)
            flow_refine = self.feedback_refine(feedback_input)
            flow = flow + flow_refine
            warped = self._warp_with_flow(cloth, flow)
        
        return warped, flow
    
    def _warp_with_flow(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Apply flow field to warp image"""
        B, C, H, W = image.shape
        
        # Create meshgrid
        grid_x = torch.linspace(-1, 1, W, device=image.device)
        grid_y = torch.linspace(-1, 1, H, device=image.device)
        mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing='xy')
        grid = torch.stack([mesh_x, mesh_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Normalize flow
        flow_norm = flow.clone()
        flow_norm[:, 0] = flow_norm[:, 0] / (W / 2)
        flow_norm[:, 1] = flow_norm[:, 1] / (H / 2)
        
        # Apply flow
        warped_grid = grid + flow_norm
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Sample
        warped = F.grid_sample(
            image,
            warped_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped