# transunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=768, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = h + x
        h = x
        x = self.norm2(x)
        x = h + self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=8, emb_dim=768, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


# --- TransUNet-Light ---
class TransUNetLight(nn.Module):
    def __init__(self, img_size=(512, 384), in_channels=22, num_classes=20,
                 hidden_size=256, num_layers=2, num_heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()

        # CNN encoder (lebih kecil)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Transformer bottleneck
        self.linear_proj = nn.Linear(128, hidden_size)
        self.transformer = TransformerEncoder(num_layers, hidden_size, num_heads, mlp_dim, dropout)
        self.proj_back = nn.Linear(hidden_size, 128)

        # Decoder
        self.up1 = DecoderBlock(128, 64)
        self.up2 = DecoderBlock(64, 32)
        self.up3 = DecoderBlock(32, 16)

        self.final = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        feat = self.pool3(x3)  # (B,128,H/8,W/8)

        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B,N,C)
        tokens = self.linear_proj(tokens)         # (B,N,Hid)
        trans = self.transformer(tokens)          # (B,N,Hid)
        trans = self.proj_back(trans)             # (B,N,128)
        trans = trans.transpose(1, 2).reshape(B, 128, H, W)

        # Decoder
        d1 = F.interpolate(trans, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.up1(d1)
        d2 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.up2(d2)
        d3 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.up3(d3)

        return self.final(d3)


# --- Quick test ---
if __name__ == "__main__":
    model = TransUNetLight(in_channels=22, num_classes=20)
    x = torch.randn(2, 22, 512, 384)
    y = model(x)
    print("Output:", y.shape)  # (2,20,512,384)
