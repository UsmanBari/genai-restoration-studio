"""
Corruption pipeline for Oxford-IIIT Pet dataset.
Generates programmatic corruptions at runtime (training) and deterministically (val/test).

Classes:
    0: Clean (no corruption)
    1: Salt-and-Pepper noise
    2: Gaussian blur
    3: Rectangular occlusion
"""

import math
import random
import numpy as np
from PIL import Image, ImageFilter
import cv2

CORRUPTION_NAMES = {
    0: "clean",
    1: "salt_and_pepper",
    2: "gaussian_blur",
    3: "rectangular_occlusion"
}

NAME_TO_CLASS = {v: k for k, v in CORRUPTION_NAMES.items()}


def apply_salt_and_pepper(img_np: np.ndarray, prob: float, rng=None) -> np.ndarray:
    """
    Apply salt-and-pepper noise to an RGB image (values in [0, 255] or [0, 1]).
    prob: fraction of total pixels to corrupt.
    Selected pixels are replaced with 0 (black) or 255 (white) with equal probability.
    """
    if rng is None:
        rng = np.random.default_rng()

    corrupted = img_np.copy()
    h, w = corrupted.shape[:2]
    num_pixels = h * w
    num_corrupt = int(num_pixels * prob)

    if num_corrupt == 0:
        return corrupted

    # Pick random pixel indices
    indices = rng.choice(num_pixels, size=num_corrupt, replace=False)
    ys = indices // w
    xs = indices % w

    # Equal probability of salt (max) or pepper (min)
    max_val = 255 if corrupted.dtype == np.uint8 or corrupted.max() > 1.0 else 1.0
    min_val = 0.0

    salt_mask = rng.random(num_corrupt) < 0.5
    pepper_mask = ~salt_mask

    if len(corrupted.shape) == 3:
        corrupted[ys[salt_mask], xs[salt_mask], :] = max_val
        corrupted[ys[pepper_mask], xs[pepper_mask], :] = min_val
    else:
        corrupted[ys[salt_mask], xs[salt_mask]] = max_val
        corrupted[ys[pepper_mask], xs[pepper_mask]] = min_val

    return corrupted


def apply_gaussian_blur(img_np: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur using OpenCV.
    kernel_size must be an odd positive integer (e.g. 3, 5, 7).
    sigma: Gaussian kernel standard deviation.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return blurred


def generate_occlusion_rectangles(img_h: int, img_w: int, num_rects: int, target_coverage: float, rng=None):
    """
    Generate non-overlapping or partially overlapping rectangular bounding boxes
    whose combined area roughly matches target_coverage.
    Returns list of dicts: [{'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}]
    """
    if rng is None:
        rng = np.random.default_rng()

    total_pixels = img_h * img_w
    target_pixels = total_pixels * target_coverage
    pixels_per_rect = target_pixels / num_rects

    rects = []
    for _ in range(num_rects):
        # Target aspect ratio between 0.5 and 2.0
        aspect = rng.uniform(0.5, 2.0)
        rect_w = int(math.sqrt(pixels_per_rect * aspect))
        rect_h = int(math.sqrt(pixels_per_rect / aspect))

        rect_w = max(4, min(img_w - 2, rect_w))
        rect_h = max(4, min(img_h - 2, rect_h))

        max_x = max(0, img_w - rect_w)
        max_y = max(0, img_h - rect_h)

        x1 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y1 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x2 = min(img_w, x1 + rect_w)
        y2 = min(img_h, y1 + rect_h)

        rects.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    return rects


def apply_rectangular_occlusion(img_np: np.ndarray, rects: list) -> np.ndarray:
    """
    Apply black rectangular occlusions specified by list of bounding boxes.
    """
    corrupted = img_np.copy()
    min_val = 0
    for r in rects:
        x1, y1, x2, y2 = r['x1'], r['y1'], r['x2'], r['y2']
        if len(corrupted.shape) == 3:
            corrupted[y1:y2, x1:x2, :] = min_val
        else:
            corrupted[y1:y2, x1:x2] = min_val
    return corrupted


def apply_random_corruption_runtime(img_np: np.ndarray, rng=None):
    """
    Training runtime corruption sampler.
    Picks one of 4 conditions with EQUAL probability (0.25 each):
      0: Clean
      1: Salt-and-pepper (prob in uniform(0.02, 0.15))
      2: Gaussian blur (kernel in {3, 5, 7}, sigma in uniform(0.5, 2.5))
      3: Rectangular occlusion (1 to 3 black rectangles, total coverage 10%-35%)

    Returns:
      corrupted_img_np: np.ndarray
      label: int (0, 1, 2, 3)
      params: dict
    """
    if rng is None:
        rng = np.random.default_rng()

    label = int(rng.integers(0, 4))
    params = {'type': CORRUPTION_NAMES[label]}

    if label == 0:  # Clean
        return img_np.copy(), label, params

    elif label == 1:  # Salt and Pepper
        prob = float(rng.uniform(0.02, 0.15))
        params['prob'] = prob
        corrupted = apply_salt_and_pepper(img_np, prob, rng=rng)
        return corrupted, label, params

    elif label == 2:  # Gaussian blur
        k_options = [3, 5, 7]
        kernel_size = int(rng.choice(k_options))
        sigma = float(rng.uniform(0.5, 2.5))
        params['kernel_size'] = kernel_size
        params['sigma'] = sigma
        corrupted = apply_gaussian_blur(img_np, kernel_size, sigma)
        return corrupted, label, params

    elif label == 3:  # Rectangular occlusion
        num_rects = int(rng.integers(1, 4))  # 1, 2, or 3
        coverage = float(rng.uniform(0.10, 0.35))
        h, w = img_np.shape[:2]
        rects = generate_occlusion_rectangles(h, w, num_rects, coverage, rng=rng)
        params['num_rects'] = num_rects
        params['coverage'] = coverage
        params['rectangles'] = rects
        corrupted = apply_rectangular_occlusion(img_np, rects)
        return corrupted, label, params

    return img_np.copy(), 0, params


def apply_deterministic_corruption(img_np: np.ndarray, meta: dict) -> np.ndarray:
    """
    Applies deterministic corruption according to metadata entry stored in manifest.
    """
    corr_type = meta.get('corruption_type', 'clean')
    if corr_type == 'clean' or meta.get('label', 0) == 0:
        return img_np.copy()
    elif corr_type == 'salt_and_pepper' or meta.get('label') == 1:
        prob = meta.get('prob', 0.05)
        # Use stored seed if present for exact pixel reproducibility
        seed = meta.get('seed')
        rng = np.random.default_rng(seed) if seed is not None else None
        return apply_salt_and_pepper(img_np, prob, rng=rng)
    elif corr_type == 'gaussian_blur' or meta.get('label') == 2:
        k = int(meta.get('kernel_size', 5))
        sigma = float(meta.get('sigma', 1.5))
        return apply_gaussian_blur(img_np, k, sigma)
    elif corr_type == 'rectangular_occlusion' or meta.get('label') == 3:
        rects = meta.get('rectangles', [])
        if not rects:
            # Reconstruct from coverage and seed if rects missing
            seed = meta.get('seed', 42)
            rng = np.random.default_rng(seed)
            num_rects = int(meta.get('num_rects', 1))
            coverage = float(meta.get('coverage', 0.15))
            h, w = img_np.shape[:2]
            rects = generate_occlusion_rectangles(h, w, num_rects, coverage, rng=rng)
        return apply_rectangular_occlusion(img_np, rects)
    return img_np.copy()
