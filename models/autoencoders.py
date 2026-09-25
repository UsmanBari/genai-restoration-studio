"""
Universal Multi-Corruption Denoising Autoencoder Architecture (Task 1).

Design Constraints:
- Progressive spatial downsampling via strided convolutions down to a compressed bottleneck.
- No unrestricted skip connections that bypass the bottleneck undamaged (true information compression).
- Symmetrical decoder with transposed convolutions / upsampling to reconstruct 128x128 RGB.
- One shared set of weights across Clean, Salt-and-Pepper, Gaussian Blur, and Occlusion.
"""

from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class _DummyModule:
        def __init__(self, *args, **kwargs):
            pass
    class _DummyNN:
        Module = _DummyModule
    torch = None
    nn = _DummyNN()
    F = None


class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv2d -> BatchNorm2d -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """Upsampling Block: ConvTranspose2d -> BatchNorm2d -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_bn: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UniversalAutoencoder(nn.Module):
    """
    Universal Convolutional Denoising Autoencoder for 128x128 RGB images.
    
    Architecture:
      Input: (B, 3, 128, 128)
      Enc1: Conv stride 2 -> (B, C, 64, 64)
      Enc2: Conv stride 2 -> (B, 2C, 32, 32)
      Enc3: Conv stride 2 -> (B, 4C, 16, 16)
      Enc4: Conv stride 2 -> (B, 8C, 8, 8)
      Enc5: Conv stride 2 -> (B, 8C, 4, 4)
      Bottleneck: 1x1 Conv compression to (B, bottleneck_dim, 4, 4)
                  [NO skip connections around this bottleneck]
      Dec5: Deconv stride 2 -> (B, 8C, 8, 8)
      Dec4: Deconv stride 2 -> (B, 4C, 16, 16)
      Dec3: Deconv stride 2 -> (B, 2C, 32, 32)
      Dec2: Deconv stride 2 -> (B, C, 64, 64)
      Dec1: Deconv stride 2 -> (B, C, 128, 128)
      Out:  1x1 Conv -> Sigmoid -> (B, 3, 128, 128)

    Total compression ratio at bottleneck (for C=64, bottleneck_dim=256):
      Input pixels: 3 * 128 * 128 = 49,152 values
      Bottleneck:   256 * 4 * 4 = 4,096 values (12x spatial-feature compression)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        bottleneck_dim: int = 256,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate

        c = base_channels

        # Encoder: 128x128 -> 4x4
        self.enc1 = ConvBlock(in_channels, c, stride=2, use_bn=False, dropout_rate=dropout_rate)       # 128 -> 64
        self.enc2 = ConvBlock(c, c * 2, stride=2, use_bn=True, dropout_rate=dropout_rate)             # 64 -> 32
        self.enc3 = ConvBlock(c * 2, c * 4, stride=2, use_bn=True, dropout_rate=dropout_rate)         # 32 -> 16
        self.enc4 = ConvBlock(c * 4, c * 8, stride=2, use_bn=True, dropout_rate=dropout_rate)         # 16 -> 8
        self.enc5 = ConvBlock(c * 8, c * 8, stride=2, use_bn=True, dropout_rate=dropout_rate)         # 8 -> 4

        # Genuine compressed bottleneck (linear projection to bottleneck_dim channels at 4x4)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(c * 8, bottleneck_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(bottleneck_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bottleneck_dim, c * 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder: 4x4 -> 128x128
        self.dec5 = DeconvBlock(c * 8, c * 8, stride=2, use_bn=True, dropout_rate=dropout_rate)       # 4 -> 8
        self.dec4 = DeconvBlock(c * 8, c * 4, stride=2, use_bn=True, dropout_rate=dropout_rate)       # 8 -> 16
        self.dec3 = DeconvBlock(c * 4, c * 2, stride=2, use_bn=True, dropout_rate=dropout_rate)       # 16 -> 32
        self.dec2 = DeconvBlock(c * 2, c, stride=2, use_bn=True, dropout_rate=dropout_rate)           # 32 -> 64
        self.dec1 = DeconvBlock(c, c, stride=2, use_bn=True, dropout_rate=dropout_rate)               # 64 -> 128

        # Output projection to RGB image in [0.0, 1.0]
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder down to bottleneck latent representation."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        z = self.bottleneck_conv(e5)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder from bottleneck latent to reconstructed image."""
        d5 = self.dec5(z)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        out = self.out_conv(d1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass: Corrupted Input -> Clean Target Reconstruction."""
        z = self.encode(x)
        out = self.decode(z)
        return out
