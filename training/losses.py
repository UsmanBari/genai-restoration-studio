"""
Differentiable loss functions for image restoration.
Implements:
  - Differentiable Structural Similarity Index (SSIM)
  - Combined Restoration Loss: L = alpha * L1 + (1 - alpha) * (1 - SSIM)
"""

import math
from typing import Tuple, Dict, Any

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
        L1Loss = _DummyModule
    torch = None
    nn = _DummyNN()
    F = None


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 2D Gaussian window kernel for SSIM calculation."""
    def gaussian(size, sigma):
        gauss = torch.exp(torch.tensor([-(x - size // 2) ** 2 / float(2 * sigma ** 2) for x in range(size)]))
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_tensor(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:
    """
    Computes Structural Similarity Index (SSIM) between two image batches (B, C, H, W) in [0.0, 1.0].
    """
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device).type(img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """SSIM Loss module: (1.0 - SSIM(x, x_hat))."""

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim_tensor(img1, img2, self.window_size)


class RestorationLoss(nn.Module):
    """
    Combined restoration loss function:
      L = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))
      
    Args:
      alpha: Trade-off weighting between pixel-level L1 error and structural SSIM.
    """

    def __init__(self, alpha: float = 0.8, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=window_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1_val = self.l1_loss(pred, target)
        ssim_val = ssim_tensor(pred, target)
        ssim_loss_val = 1.0 - ssim_val

        total_loss = self.alpha * l1_val + (1.0 - self.alpha) * ssim_loss_val

        breakdown = {
            'loss': total_loss.item(),
            'l1': l1_val.item(),
            'ssim': ssim_val.item(),
            'ssim_loss': ssim_loss_val.item()
        }
        return total_loss, breakdown
