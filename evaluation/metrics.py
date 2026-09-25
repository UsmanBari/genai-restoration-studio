"""
Evaluation metrics for image restoration: PSNR, SSIM, L1 error.
Operates on PyTorch tensors or NumPy arrays in range [0.0, 1.0].
"""

import math
from typing import Dict, Tuple, Union
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr


def compute_psnr(img_pred: np.ndarray, img_target: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    Images should be NumPy arrays (H, W, C) in [0.0, 1.0] or [0, 255].
    """
    data_range = 255.0 if img_pred.max() > 1.0 else 1.0
    mse = np.mean((img_pred.astype(np.float64) - img_target.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def compute_ssim(img_pred: np.ndarray, img_target: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM).
    Images should be NumPy arrays (H, W, C).
    """
    data_range = 255.0 if img_pred.max() > 1.0 else 1.0
    channel_axis = 2 if len(img_pred.shape) == 3 else None
    return float(skimage_ssim(
        img_target,
        img_pred,
        data_range=data_range,
        channel_axis=channel_axis
    ))


def compute_l1(img_pred: np.ndarray, img_target: np.ndarray) -> float:
    """Computes Mean Absolute Error (L1 distance)."""
    data_range = 255.0 if img_pred.max() > 1.0 else 1.0
    l1 = np.mean(np.abs(img_pred.astype(np.float64) - img_target.astype(np.float64)))
    return float(l1 / data_range)


def evaluate_image_pair(pred_np: np.ndarray, target_np: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a single prediction and clean target pair (H, W, C) in [0.0, 1.0].
    Returns dict: {'psnr': float, 'ssim': float, 'l1': float}.
    """
    # Ensure clipping to [0.0, 1.0]
    pred_clamped = np.clip(pred_np, 0.0, 1.0)
    target_clamped = np.clip(target_np, 0.0, 1.0)

    psnr_val = compute_psnr(pred_clamped, target_clamped)
    ssim_val = compute_ssim(pred_clamped, target_clamped)
    l1_val = compute_l1(pred_clamped, target_clamped)

    return {
        'psnr': round(psnr_val, 4),
        'ssim': round(ssim_val, 4),
        'l1': round(l1_val, 6)
    }
