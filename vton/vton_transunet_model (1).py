"""
TransUNet Implementation
Bab IV 4.3.2 - Hybrid CNN-Transformer untuk Segmentasi
Kombinasi CNN encoder dengan Transformer bottleneck untuk capture local dan global features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math
from torch.utils.checkpoint import checkpoint

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self Attention module
    Bab IV 4.3.2.1 - Transformer Component
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(B, N, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_size).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.out(x)
        x = self.dropout(x)
        
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block dengan MHA dan FFN
    Bab IV 4.3.2.2 - Transformer Block Architecture
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout)
        )
    
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x)
        return self._forward(x)

class CNNEncoder(nn.Module):
    """
    CNN Encoder untuk extract local features
    Bab IV 4.3.2.3 - CNN Backbone
    """
    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # Build encoder layers
        prev_channels = in_channels
        for out_channels in encoder_channels:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.pools.append(nn.MaxPool2d(2, 2))
            prev_channels = out_channels
        
        self.encoder_channels = encoder_channels
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            features.append(x)
            x = pool(x)
        
        return features

class TransUNet(nn.Module):
    """
    TransUNet: Hybrid CNN-Transformer untuk Segmentasi
    Bab IV 4.3.2 - Model Utama
    
    Architecture:
    1. CNN encoder untuk local features
    2. Transformer bottleneck untuk global context
    3. CNN decoder dengan skip connections
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 384),
        in_channels: int = 3,
        num_classes: int = 20,
        encoder_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 32],
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        patch_size: int = 16,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # CNN Encoder
        self.encoder = CNNEncoder(in_channels, encoder_channels)
        
        # Patch embedding untuk transformer input
        self.patch_embed = nn.Conv2d(
            encoder_channels[-1],
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                mlp_dim,
                dropout,
                use_checkpoint
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # Reshape back to image
        self.reshape = nn.Linear(hidden_size, encoder_channels[-1] * patch_size * patch_size)
        
        # CNN Decoder dengan skip connections
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Build decoder layers
        prev_channels = encoder_channels[-1]
        for i, out_channels in enumerate(decoder_channels):
            # Account for skip connection
            skip_channels = encoder_channels[-(i+2)] if i < len(encoder_channels)-1 else 0
            in_ch = prev_channels + skip_channels
            
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.upsamples.append(
                nn.ConvTranspose2d(prev_channels, prev_channels, 2, stride=2)
            )
            prev_channels = out_channels
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # CNN Encoder dengan skip connections
        encoder_features = self.encoder(x)
        
        # Ambil deepest feature untuk transformer
        x = encoder_features[-1]
        
        # Patch embedding
        x = self.patch_embed(x)  # B, hidden_size, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, hidden_size
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks untuk global context
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back to spatial dimensions
        x = self.reshape(x)  # B, num_patches, C*P*P
        x = x.transpose(1, 2).view(B, encoder_features[-1].shape[1], 
                                   H // self.patch_size, W // self.patch_size)
        
        # Upsample to original encoder resolution
        x = F.interpolate(x, size=encoder_features[-1].shape[2:], mode='bilinear', align_corners=False)
        
        # CNN Decoder dengan skip connections
        for i, (decoder, upsample) in enumerate(zip(self.decoders, self.upsamples)):
            # Upsample
            x = upsample(x)
            
            # Skip connection
            if i < len(encoder_features) - 1:
                skip_feature = encoder_features[-(i+2)]
                x = torch.cat([x, skip_feature], dim=1)
            
            # Decode
            x = decoder(x)
        
        # Final segmentation
        x = self.seg_head(x)
        
        # Upsample to original size if needed
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


class UNet(nn.Module):
    """
    Standard U-Net untuk baseline comparison
    Bab IV 4.5.1 - Baseline Model
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 20,
        encoder_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 32]
    ):
        super().__init__()
        
        # Encoder
        self.encoder = CNNEncoder(in_channels, encoder_channels)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(encoder_channels[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels[-1] * 2, encoder_channels[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(encoder_channels[-1] * 2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        prev_channels = encoder_channels[-1] * 2
        for i, out_channels in enumerate(decoder_channels):
            # Account for skip connection
            skip_channels = encoder_channels[-(i+1)] if i < len(encoder_channels) else 0
            in_ch = prev_channels + skip_channels
            
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.upsamples.append(
                nn.ConvTranspose2d(prev_channels, prev_channels, 2, stride=2)
            )
            prev_channels = out_channels
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoder_features = self.encoder(x)
        
        # Bottleneck
        x = self.bottleneck(encoder_features[-1])
        
        # Decoder with skip connections
        for i, (decoder, upsample) in enumerate(zip(self.decoders, self.upsamples)):
            x = upsample(x)
            
            if i < len(encoder_features):
                skip_feature = encoder_features[-(i+1)]
                x = torch.cat([x, skip_feature], dim=1)
            
            x = decoder(x)
        
        # Final segmentation
        x = self.seg_head(x)
        
        return x