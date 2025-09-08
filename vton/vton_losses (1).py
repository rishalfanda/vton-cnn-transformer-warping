"""
Loss Functions untuk VTON Pipeline
Bab IV 4.3.3 - Boundary-aware Loss dan Fungsi Loss Lainnya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt

class DiceLoss(nn.Module):
    """
    Dice Loss untuk segmentasi
    Bab IV 4.3.3.1 - Dice Coefficient Loss
    """
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) - predicted logits
            target: (B, H, W) - ground truth labels
        """
        B, C, H, W = pred.shape
        
        # Convert to probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Calculate dice per class
        dice_loss = 0
        for c in range(C):
            pred_c = pred[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice).mean()
        
        dice_loss /= C
        
        return dice_loss

class BoundaryLoss(nn.Module):
    """
    Boundary-aware Loss untuk presisi tepi segmentasi
    Bab IV 4.3.3.2 - Boundary Enhancement
    """
    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def _compute_distance_map(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute distance transform map untuk boundary weighting
        """
        B, H, W = mask.shape
        dist_maps = torch.zeros_like(mask, dtype=torch.float32)
        
        for b in range(B):
            mask_np = mask[b].cpu().numpy()
            
            # Compute distance from boundaries
            dist_pos = distance_transform_edt(mask_np)
            dist_neg = distance_transform_edt(1 - mask_np)
            dist = dist_pos + dist_neg
            
            # Normalize dan apply exponential weighting
            dist = torch.from_numpy(dist).float()
            weight_map = 1 + self.theta * torch.exp(-dist / self.theta0)
            
            dist_maps[b] = weight_map.to(mask.device)
        
        return dist_maps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) - predicted logits
            target: (B, H, W) - ground truth labels
        """
        B, C, H, W = pred.shape
        
        # Compute boundary weights
        boundary_weights = self._compute_distance_map(target)
        
        # Weighted cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * boundary_weights
        
        return weighted_loss.mean()

class CombinedSegmentationLoss(nn.Module):
    """
    Combined Loss untuk Segmentasi
    Bab IV 4.3.3.3 - Multi-component Loss Function
    Kombinasi Dice + CrossEntropy + Boundary
    """
    def __init__(
        self,
        dice_weight: float = 0.4,
        ce_weight: float = 0.3,
        boundary_weight: float = 0.3,
        use_boundary: bool = True
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary
        
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.boundary_loss = BoundaryLoss() if use_boundary else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred: (B, C, H, W) - predicted logits
            target: (B, H, W) - ground truth labels
        
        Returns:
            total_loss: combined loss value
            loss_dict: dictionary of individual losses for logging
        """
        loss_dict = {}
        
        # Dice loss
        dice = self.dice_loss(pred, target)
        loss_dict['dice_loss'] = dice.item()
        
        # Cross entropy loss
        ce = self.ce_loss(pred, target)
        loss_dict['ce_loss'] = ce.item()
        
        # Combine losses
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        # Boundary loss (optional)
        if self.use_boundary and self.boundary_loss is not None:
            boundary = self.boundary_loss(pred, target)
            loss_dict['boundary_loss'] = boundary.item()
            total_loss += self.boundary_weight * boundary
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

class WarpingLoss(nn.Module):
    """
    Loss function untuk warping module
    Bab IV 4.3.4.3 - Warping Optimization
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.2,
        smooth_weight: float = 0.1
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.smooth_weight = smooth_weight
        
        # VGG features untuk perceptual loss
        self.vgg_features = self._build_vgg_features()
    
    def _build_vgg_features(self) -> nn.Module:
        """Build VGG feature extractor untuk perceptual loss"""
        from torchvision import models
        vgg = models.vgg19(pretrained=True).features
        
        # Freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract specific layers
        layers = []
        layer_indices = [3, 8, 17, 26, 35]  # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
        
        return nn.ModuleList([vgg[:idx+1] for idx in layer_indices])
    
    def _perceptual_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate perceptual loss using VGG features"""
        loss = 0
        
        for layer in self.vgg_features:
            pred_features = layer(pred)
            target_features = layer(target)
            loss += F.l1_loss(pred_features, target_features)
        
        return loss / len(self.vgg_features)
    
    def _smoothness_loss(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Calculate smoothness loss untuk flow field
        Mendorong spatial coherence dalam warping
        """
        # Gradient in x direction
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        # Gradient in y direction  
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        
        # L2 norm of gradients
        smooth_loss = torch.mean(dx**2) + torch.mean(dy**2)
        
        return smooth_loss
    
    def forward(
        self,
        warped: torch.Tensor,
        target: torch.Tensor,
        flow: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            warped: warped cloth image
            target: target cloth on person
            flow: optional flow field for smoothness constraint
        """
        loss_dict = {}
        
        # L1 loss
        l1 = F.l1_loss(warped, target)
        loss_dict['l1_loss'] = l1.item()
        
        # Perceptual loss
        perceptual = self._perceptual_loss(warped, target)
        loss_dict['perceptual_loss'] = perceptual.item()
        
        # Total loss
        total_loss = self.l1_weight * l1 + self.perceptual_weight * perceptual
        
        # Smoothness loss if flow is provided
        if flow is not None and self.smooth_weight > 0:
            smooth = self._smoothness_loss(flow)
            loss_dict['smooth_loss'] = smooth.item()
            total_loss += self.smooth_weight * smooth
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

class SSIMLoss(nn.Module):
    """
    SSIM Loss untuk quality assessment
    Bab IV 4.4.3 - Structural Similarity
    """
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size()[1]
        
        if channel != self.channel:
            self.window = self._create_window(self.window_size, channel)
            self.channel = channel
        
        if img1.is_cuda:
            self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)
        
        return self._ssim(img1, img2, self.window, self.window_size, channel, self.size_average)
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        size_average: bool = True
    ) -> torch.Tensor:
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return 1 - ssim_map.mean()  # Return as loss (1 - SSIM)
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)