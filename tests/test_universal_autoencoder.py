"""
Unit tests for Task 1: Universal Autoencoder, Loss, Metrics, and ONNX Runner.
"""

import pytest
import numpy as np
from PIL import Image

from models.onnx_runner import UniversalRestorationONNXRunner
from evaluation.metrics import compute_psnr, compute_ssim, compute_l1, evaluate_image_pair


def test_evaluation_metrics():
    # Identical images
    img1 = np.full((128, 128, 3), 0.5, dtype=np.float32)
    img2 = np.full((128, 128, 3), 0.5, dtype=np.float32)
    res = evaluate_image_pair(img1, img2)
    assert res['psnr'] >= 99.0
    assert res['ssim'] >= 0.99
    assert res['l1'] == 0.0

    # Slight difference
    img3 = img1.copy()
    img3[0, 0] = 0.6
    res2 = evaluate_image_pair(img1, img3)
    assert res2['psnr'] > 30.0
    assert res2['l1'] > 0.0


def test_onnx_runner_preprocessing():
    # Test preprocessor and postprocessor without loading physical file
    class DummyRunner(UniversalRestorationONNXRunner):
        def _load_session(self):
            pass

    runner = DummyRunner.__new__(DummyRunner)
    test_img = Image.new('RGB', (200, 150), color=(100, 150, 200))
    tensor_np = runner.preprocess_image(test_img, target_size=(128, 128))
    assert tensor_np.shape == (1, 3, 128, 128)
    assert tensor_np.dtype == np.float32
    assert tensor_np.min() >= 0.0 and tensor_np.max() <= 1.0

    reconstructed_img = runner.postprocess_array(tensor_np)
    assert reconstructed_img.size == (128, 128)
    assert reconstructed_img.mode == 'RGB'
