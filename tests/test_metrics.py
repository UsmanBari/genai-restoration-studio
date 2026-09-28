"""
Unit tests for evaluation metrics (PSNR, SSIM, L1 error).
Verifies:
  1. Identical images return 100.0 dB PSNR, 1.0 SSIM, 0.0 L1.
  2. [-1, 1] and [0, 1] input ranges produce consistent mathematical results.
  3. All-white vs all-black images produce correct boundary metrics.
  4. PyTorch Tensor and NumPy array calculations are mathematically identical.
"""

import numpy as np
import pytest
import torch

from evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    compute_l1,
    evaluate_image_pair,
    calculate_psnr,
    calculate_ssim,
    _ensure_0_1,
)


def test_metrics_identical_images():
    """Identical images must yield 100.0 dB PSNR, 1.0 SSIM, and 0.0 L1 error."""
    img = np.random.RandomState(42).uniform(0.0, 1.0, (128, 128, 3)).astype(np.float32)

    psnr = compute_psnr(img, img)
    ssim = compute_ssim(img, img)
    l1 = compute_l1(img, img)

    assert psnr == 100.0, f"Expected 100.0 dB for identical images, got {psnr}"
    assert ssim == 1.0, f"Expected 1.0 SSIM for identical images, got {ssim}"
    assert l1 == 0.0, f"Expected 0.0 L1 for identical images, got {l1}"


def test_metrics_range_handling_minus1_1_vs_0_1():
    """Tensors/arrays in [-1, 1] (from Tanh) and [0, 1] must produce consistent metrics."""
    rng = np.random.RandomState(123)
    img_01_a = rng.uniform(0.0, 1.0, (64, 64, 3)).astype(np.float32)
    img_01_b = rng.uniform(0.0, 1.0, (64, 64, 3)).astype(np.float32)

    # Convert to [-1, 1]
    img_m11_a = (img_01_a * 2.0) - 1.0
    img_m11_b = (img_01_b * 2.0) - 1.0

    psnr_01 = compute_psnr(img_01_a, img_01_b)
    psnr_m11 = compute_psnr(img_m11_a, img_m11_b)

    ssim_01 = compute_ssim(img_01_a, img_01_b)
    ssim_m11 = compute_ssim(img_m11_a, img_m11_b)

    l1_01 = compute_l1(img_01_a, img_01_b)
    l1_m11 = compute_l1(img_m11_a, img_m11_b)

    assert abs(psnr_01 - psnr_m11) < 1e-4, f"PSNR mismatch between [0,1] ({psnr_01}) and [-1,1] ({psnr_m11})"
    assert abs(ssim_01 - ssim_m11) < 1e-4, f"SSIM mismatch between [0,1] ({ssim_01}) and [-1,1] ({ssim_m11})"
    assert abs(l1_01 - l1_m11) < 1e-4, f"L1 mismatch between [0,1] ({l1_01}) and [-1,1] ({l1_m11})"


def test_metrics_black_vs_white():
    """All-black (0.0) vs all-white (1.0) must produce MSE=1.0 -> PSNR=0.0 dB, L1=1.0."""
    black = np.zeros((64, 64, 3), dtype=np.float32)
    white = np.ones((64, 64, 3), dtype=np.float32)

    psnr = compute_psnr(black, white)
    l1 = compute_l1(black, white)
    ssim = compute_ssim(black, white)

    assert abs(psnr - 0.0) < 1e-4, f"Expected 0.0 dB for black vs white, got {psnr}"
    assert abs(l1 - 1.0) < 1e-4, f"Expected 1.0 L1 for black vs white, got {l1}"
    assert ssim < 0.1, f"Expected near-zero SSIM for black vs white, got {ssim}"


def test_torch_tensor_metrics_equivalence():
    """calculate_psnr with PyTorch tensor matches compute_psnr with NumPy array."""
    t_a = torch.rand(1, 3, 64, 64)
    t_b = torch.rand(1, 3, 64, 64)

    np_a = t_a[0].permute(1, 2, 0).numpy()
    np_b = t_b[0].permute(1, 2, 0).numpy()

    psnr_t = calculate_psnr(t_a, t_b)
    psnr_np = compute_psnr(np_a, np_b)

    assert abs(psnr_t - psnr_np) < 1e-3, f"Torch PSNR ({psnr_t}) differs from NumPy PSNR ({psnr_np})"
