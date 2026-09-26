"""
Corruption Classifier Architecture for Task 2.
Lightweight convolutional neural network for 4-way corruption classification:
    0: Clean
    1: Salt & Pepper
    2: Gaussian Blur
    3: Rectangular Occlusion
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.corruptions import CORRUPTION_NAMES, NAME_TO_CLASS


class ConvBlock(nn.Module):
    """Standard Conv2d -> BatchNorm2d -> LeakyReLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CorruptionClassifier(nn.Module):
    """
    Lightweight 4-stage convolutional classifier for 128x128 RGB images.
    
    Architecture:
      Stage 1: 128x128x3   -> Conv(3, C)    -> Pool -> 64x64xC
      Stage 2: 64x64xC     -> Conv(C, 2C)   -> Pool -> 32x32x2C
      Stage 3: 32x32x2C    -> Conv(2C, 4C)  -> Pool -> 16x16x4C
      Stage 4: 16x16x4C    -> Conv(4C, 8C)  -> Pool -> 8x8x8C
      Head:    GlobalAvgPool -> Flatten(8C) -> Dropout(p) -> Linear(8C, 4)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 32,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Stage 1: 128 -> 64
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, c1),
            ConvBlock(c1, c1),
            nn.MaxPool2d(2, 2)
        )

        # Stage 2: 64 -> 32
        self.stage2 = nn.Sequential(
            ConvBlock(c1, c2),
            ConvBlock(c2, c2),
            nn.MaxPool2d(2, 2)
        )

        # Stage 3: 32 -> 16
        self.stage3 = nn.Sequential(
            ConvBlock(c2, c3),
            ConvBlock(c3, c3),
            nn.MaxPool2d(2, 2)
        )

        # Stage 4: 16 -> 8
        self.stage4 = nn.Sequential(
            ConvBlock(c3, c4),
            ConvBlock(c4, c4),
            nn.MaxPool2d(2, 2)
        )

        # Global Average Pooling and classification head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.classifier = nn.Linear(c4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized logits.
        Input: (B, 3, 128, 128)
        Output: (B, 4) logits
        """
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.gap(h)
        h = torch.flatten(h, 1)
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper returning predicted class indices and class probabilities.
        Returns:
            preds: (B,) int64 tensor with class IDs in {0, 1, 2, 3}
            probs: (B, 4) float32 tensor with softmax probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
