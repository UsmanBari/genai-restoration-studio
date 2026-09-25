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


def test_universal_autoencoder_architecture_shapes():
    """Sanity test verifying UniversalAutoencoder encoder, bottleneck, gated skip, decoder, and output shapes."""
    try:
        import torch
        from models.autoencoders import UniversalAutoencoder
    except ImportError:
        pytest.skip("PyTorch not installed in this environment")

    base_channels = 48
    bottleneck_dim = 128
    model = UniversalAutoencoder(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        bottleneck_dim=bottleneck_dim,
        dropout_rate=0.2
    )
    model.eval()

    dummy_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        # Test encode
        z, e_init = model.encode(dummy_input)
        assert z.shape == (2, bottleneck_dim, 8, 8), f"Expected (2, {bottleneck_dim}, 8, 8), got {z.shape}"
        assert e_init.shape == (2, base_channels, 128, 128), f"Expected (2, {base_channels}, 128, 128), got {e_init.shape}"

        # Test decode with gated skip
        recon_gated = model.decode(z, e_init)
        assert recon_gated.shape == (2, 3, 128, 128)

        # Test decode without skip (fallback)
        recon_standalone = model.decode(z)
        assert recon_standalone.shape == (2, 3, 128, 128)

        # Test end-to-end forward
        output = model(dummy_input)
        assert output.shape == (2, 3, 128, 128)
        assert output.min() >= 0.0 and output.max() <= 1.0

        # Test learned gate computation and range [0.0, 1.0]
        gate = model.get_gate(dummy_input)
        assert gate.shape == (2, base_channels, 128, 128)
        assert gate.min() >= 0.0, f"Gate min below 0: {gate.min()}"
        assert gate.max() <= 1.0, f"Gate max above 1: {gate.max()}"


