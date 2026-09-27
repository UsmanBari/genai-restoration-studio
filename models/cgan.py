"""
Conditional Generative Adversarial Network (cGAN) Architecture for Task 4.
Style-Conditioned Face-to-Sketch Synthesis on FS2K Dataset.

Components:
  1. StyleConditionedUNetGenerator: U-Net architecture with full skip connections at all levels,
     conditioned on learned style embeddings (Styles 0, 1, 2) and generating 128x128 RGB sketches in [-1, 1].
  2. ConditionalPatchGANDiscriminator: 70x70 receptive field PatchGAN classifying local real vs fake
     patches, conditioned on input photo and style category.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleConditionedUNetGenerator(nn.Module):
    """
    U-Net Generator with Full Skip Connections and Style Conditioning.
    
    Args:
        in_channels: Input photograph channels (default: 3).
        out_channels: Output sketch channels (default: 3).
        num_styles: Number of style categories in FS2K (default: 3).
        emb_dim: Dimension of learned categorical style embedding (default: 32).
        emb_channels: Projected spatial channels for style conditioning (default: 16).
        base_channels: Number of base encoder filters (default: 64).
        dropout_rate: Dropout rate in bottleneck decoder layers (default: 0.5).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_styles: int = 3,
        emb_dim: int = 32,
        emb_channels: int = 16,
        base_channels: int = 64,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_styles = num_styles
        self.emb_dim = emb_dim
        self.emb_channels = emb_channels
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate

        # 1. Learned Categorical Style Embedding & Spatial Projection
        self.style_embedding = nn.Embedding(num_styles, emb_dim)
        self.style_proj = nn.Linear(emb_dim, emb_channels)

        # 2. Encoder Layers (128x128 -> 4x4)
        c = base_channels
        # Enc1: 128 -> 64
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels + emb_channels, c, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Enc2: 64 -> 32
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Enc3: 32 -> 16
        self.enc3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Enc4: 16 -> 8
        self.enc4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Enc5 (Bottleneck): 8 -> 4
        self.enc5 = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. Decoder Layers with Full Skip Connections (4x4 -> 128x128)
        # Dec5: 4 -> 8 (input: c*8 -> output: c*8, skip concat with enc4 -> c*16)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True)
        )
        # Dec4: 8 -> 16 (input: c*16 -> output: c*4, skip concat with enc3 -> c*8)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(c * 16, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True)
        )
        # Dec3: 16 -> 32 (input: c*8 -> output: c*2, skip concat with enc2 -> c*4)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.ReLU(inplace=True)
        )
        # Dec2: 32 -> 64 (input: c*4 -> output: c, skip concat with enc1 -> c*2)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c, affine=True),
            nn.ReLU(inplace=True)
        )
        # Dec1: 64 -> 128 (input: c*2 -> output: out_channels)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

    def _get_spatial_style_map(self, style_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Projects style category ID to spatial feature map (B, emb_channels, H, W)."""
        # Ensure 1D tensor of style indices
        if style_ids.dim() > 1:
            style_ids = style_ids.squeeze()
        if style_ids.dim() == 0:
            style_ids = style_ids.unsqueeze(0)
            
        emb = self.style_embedding(style_ids)          # (B, emb_dim)
        proj = self.style_proj(emb)                    # (B, emb_channels)
        spatial_map = proj.view(-1, self.emb_channels, 1, 1).expand(-1, -1, height, width)
        return spatial_map

    def forward(self, photo: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of style-conditioned U-Net Generator.
        
        Args:
            photo: Input photograph tensor (B, 3, 128, 128) in [-1.0, 1.0].
            style_ids: Style category indices (B,) with values in {0, 1, 2}.
            
        Returns:
            Generated sketch tensor (B, 3, 128, 128) in [-1.0, 1.0].
        """
        b, _, h, w = photo.shape
        style_map = self._get_spatial_style_map(style_ids, h, w)
        x_in = torch.cat([photo, style_map], dim=1)  # (B, 3 + emb_channels, 128, 128)

        # Encoder forward pass with skip activations saved
        e1 = self.enc1(x_in)   # (B, c, 64, 64)
        e2 = self.enc2(e1)     # (B, 2c, 32, 32)
        e3 = self.enc3(e2)     # (B, 4c, 16, 16)
        e4 = self.enc4(e3)     # (B, 8c, 8, 8)
        e5 = self.enc5(e4)     # (B, 8c, 4, 4) - Bottleneck

        # Decoder forward pass with full U-Net skip connection concatenations
        d5 = self.dec5(e5)                # (B, 8c, 8, 8)
        d5_cat = torch.cat([d5, e4], dim=1)  # (B, 16c, 8, 8)

        d4 = self.dec4(d5_cat)            # (B, 4c, 16, 16)
        d4_cat = torch.cat([d4, e3], dim=1)  # (B, 8c, 16, 16)

        d3 = self.dec3(d4_cat)            # (B, 2c, 32, 32)
        d3_cat = torch.cat([d3, e2], dim=1)  # (B, 4c, 32, 32)

        d2 = self.dec2(d3_cat)            # (B, c, 64, 64)
        d2_cat = torch.cat([d2, e1], dim=1)  # (B, 2c, 64, 64)

        out_sketch = self.dec1(d2_cat)    # (B, 3, 128, 128) in [-1.0, 1.0]
        return out_sketch


class ConditionalPatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Conditional Discriminator.
    Determines whether local 70x70 image patches are real or generated sketches,
    conditioned on the input facial photograph and categorical sketch style.
    
    Input:
        photo: (B, 3, 128, 128)
        sketch: (B, 3, 128, 128) [real target or generated sketch]
        style_ids: (B,) categorical style indices
    Output:
        logits: (B, 1, 14, 14) patch prediction grid.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_styles: int = 3,
        emb_dim: int = 32,
        emb_channels: int = 16,
        base_channels: int = 64
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_styles = num_styles
        self.emb_dim = emb_dim
        self.emb_channels = emb_channels
        self.base_channels = base_channels

        # Learned Style Embedding & Spatial Projection
        self.style_embedding = nn.Embedding(num_styles, emb_dim)
        self.style_proj = nn.Linear(emb_dim, emb_channels)

        c = base_channels
        total_in_channels = in_channels * 2 + emb_channels  # 3 (photo) + 3 (sketch) + emb_channels

        # Layer 1: 128 -> 64 (no norm on first layer per standard pix2pix)
        self.layer1 = nn.Sequential(
            nn.Conv2d(total_in_channels, c, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 2: 64 -> 32
        self.layer2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 3: 32 -> 16
        self.layer3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 4: 16 -> 15 (stride 1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Final Patch Prediction Logits: 15 -> 14 (stride 1)
        self.out_conv = nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1, bias=True)

    def _get_spatial_style_map(self, style_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Projects style category ID to spatial feature map (B, emb_channels, H, W)."""
        if style_ids.dim() > 1:
            style_ids = style_ids.squeeze()
        if style_ids.dim() == 0:
            style_ids = style_ids.unsqueeze(0)
            
        emb = self.style_embedding(style_ids)
        proj = self.style_proj(emb)
        return proj.view(-1, self.emb_channels, 1, 1).expand(-1, -1, height, width)

    def forward(self, photo: torch.Tensor, sketch: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Conditional PatchGAN Discriminator.
        
        Args:
            photo: Input photograph (B, 3, 128, 128) in [-1.0, 1.0].
            sketch: Real or generated sketch (B, 3, 128, 128) in [-1.0, 1.0].
            style_ids: Style category indices (B,) with values in {0, 1, 2}.
            
        Returns:
            Patch logits (B, 1, 14, 14).
        """
        b, _, h, w = photo.shape
        style_map = self._get_spatial_style_map(style_ids, h, w)
        d_in = torch.cat([photo, sketch, style_map], dim=1)  # (B, 6 + emb_channels, 128, 128)

        x = self.layer1(d_in)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.out_conv(x)  # (B, 1, 14, 14)
        return logits
