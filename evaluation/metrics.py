"""
Evaluation metrics for image restoration: PSNR, SSIM, L1 error.
Operates on PyTorch tensors or NumPy arrays in range [0.0, 1.0].
"""

import math
from typing import Dict, Tuple, Union, Any, Optional
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr


def _ensure_0_1(img: np.ndarray) -> np.ndarray:
    """Safely converts any image array (H, W, C) from [-1, 1] or [0, 255] to [0.0, 1.0]."""
    arr = img.astype(np.float64)
    if arr.min() < -0.05:
        # Array is in [-1.0, 1.0] from Tanh
        arr = np.clip((arr * 0.5) + 0.5, 0.0, 1.0)
    elif arr.max() > 1.05:
        # Array is in [0, 255]
        arr = np.clip(arr / 255.0, 0.0, 1.0)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return arr


def compute_psnr(img_pred: np.ndarray, img_target: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB on [0.0, 1.0] range.
    Returns 100.0 dB for identical images (MSE < 1e-12).
    """
    p = _ensure_0_1(img_pred)
    t = _ensure_0_1(img_target)
    mse = float(np.mean((p - t) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def compute_ssim(img_pred: np.ndarray, img_target: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) on [0.0, 1.0] range.
    Returns 1.0 for identical images.
    """
    p = _ensure_0_1(img_pred)
    t = _ensure_0_1(img_target)
    if np.allclose(p, t, atol=1e-6):
        return 1.0
    channel_axis = 2 if len(p.shape) == 3 else None
    return float(skimage_ssim(
        t,
        p,
        data_range=1.0,
        channel_axis=channel_axis
    ))


def compute_l1(img_pred: np.ndarray, img_target: np.ndarray) -> float:
    """Computes Mean Absolute Error (L1 distance) on [0.0, 1.0] range."""
    p = _ensure_0_1(img_pred)
    t = _ensure_0_1(img_target)
    return float(np.mean(np.abs(p - t)))


def evaluate_image_pair(pred_np: np.ndarray, target_np: np.ndarray) -> Dict[str, float]:
    """
    Evaluates a single prediction and clean target pair (H, W, C) in [0.0, 1.0].
    Returns dict: {'psnr': float, 'ssim': float, 'l1': float}.
    """
    psnr_val = compute_psnr(pred_np, target_np)
    ssim_val = compute_ssim(pred_np, target_np)
    l1_val = compute_l1(pred_np, target_np)

    return {
        'psnr': round(psnr_val, 4),
        'ssim': round(ssim_val, 4),
        'l1': round(l1_val, 6)
    }


def calculate_psnr(pred: Union[np.ndarray, Any], target: Union[np.ndarray, Any]) -> float:
    """Computes PSNR from NumPy array or PyTorch Tensor."""
    if hasattr(pred, "detach"):
        import torch
        p_t = pred.detach().float()
        t_t = target.detach().float()
        if p_t.min() < -0.05:
            p_t = torch.clamp((p_t * 0.5) + 0.5, 0.0, 1.0)
            t_t = torch.clamp((t_t * 0.5) + 0.5, 0.0, 1.0)
        elif p_t.max() > 1.05:
            p_t = torch.clamp(p_t / 255.0, 0.0, 1.0)
            t_t = torch.clamp(t_t / 255.0, 0.0, 1.0)
        else:
            p_t = torch.clamp(p_t, 0.0, 1.0)
            t_t = torch.clamp(t_t, 0.0, 1.0)
        mse = float(torch.mean((p_t - t_t) ** 2).item())
        if mse <= 1e-12:
            return 100.0
        return float(10.0 * math.log10(1.0 / mse))
    return compute_psnr(pred, target)


def calculate_ssim(pred: Union[np.ndarray, Any], target: Union[np.ndarray, Any]) -> float:
    """Computes SSIM from NumPy array or PyTorch Tensor batch."""
    if hasattr(pred, "permute"):
        p_np = pred.detach().permute(0, 2, 3, 1).cpu().numpy()
        t_np = target.detach().permute(0, 2, 3, 1).cpu().numpy()
        return float(np.mean([compute_ssim(p_np[i], t_np[i]) for i in range(len(p_np))]))
    return compute_ssim(pred, target)
