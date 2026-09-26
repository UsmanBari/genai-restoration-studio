"""
Unit Tests for Task 3: Soft Mixture-of-Experts (SoftMoERestorationNetwork).
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np

from models.moe import SoftMoERestorationNetwork
from models.classifiers import CorruptionClassifier
from models.autoencoders import UniversalAutoencoder
from training.losses import SoftMoECompositeLoss
from models.onnx_export import export_to_onnx, verify_onnx_numerical_equivalence


def test_soft_moe_forward_shapes():
    """Verify forward pass output shapes and probability normalization."""
    model = SoftMoERestorationNetwork(
        gate_base_channels=16,
        specialist_base_channels=16,
        specialist_bottleneck_dim=32,
        temperature=1.0
    )
    model.eval()

    dummy_input = torch.rand(3, 3, 128, 128)
    reconstructed, weights, logits = model(dummy_input)

    assert reconstructed.shape == (3, 3, 128, 128), f"Expected (3, 3, 128, 128), got {reconstructed.shape}"
    assert weights.shape == (3, 4), f"Expected (3, 4), got {weights.shape}"
    assert logits.shape == (3, 4), f"Expected (3, 4), got {logits.shape}"

    # Weights must be non-negative and sum to 1.0 along expert dimension
    assert torch.all(weights >= 0.0), "Routing weights must be non-negative"
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-5), "Routing weights must sum to 1.0 per sample"


def test_temperature_scaling():
    """Verify temperature controls sharpness of softmax distribution."""
    model = SoftMoERestorationNetwork(gate_base_channels=16, specialist_base_channels=16, specialist_bottleneck_dim=32)
    dummy_input = torch.rand(2, 3, 128, 128)

    # Sharp temperature
    model.set_temperature(0.2)
    weights_sharp, _ = model.get_gating_weights(dummy_input)

    # Soft temperature
    model.set_temperature(5.0)
    weights_soft, _ = model.get_gating_weights(dummy_input)

    # Sharp distribution should have higher maximum per-sample weight than soft distribution
    max_sharp = weights_sharp.max(dim=1).values
    max_soft = weights_soft.max(dim=1).values
    assert torch.all(max_sharp >= max_soft), "Low temperature must produce sharper/more concentrated distribution"


def test_freeze_unfreeze_experts():
    """Verify freeze_experts and unfreeze_experts properly toggle gradient tracking."""
    model = SoftMoERestorationNetwork(gate_base_channels=16, specialist_base_channels=16, specialist_bottleneck_dim=32)

    # Initially all require grad
    for specialist in (model.specialist_sp, model.specialist_blur, model.specialist_occlusion):
        assert all(p.requires_grad for p in specialist.parameters())

    # Freeze experts (Phase 1)
    model.freeze_experts()
    for specialist in (model.specialist_sp, model.specialist_blur, model.specialist_occlusion):
        assert all(not p.requires_grad for p in specialist.parameters())
    # Gate remains trainable
    assert all(p.requires_grad for p in model.gate.parameters())

    # Unfreeze experts (Phase 2)
    model.unfreeze_experts()
    for specialist in (model.specialist_sp, model.specialist_blur, model.specialist_occlusion):
        assert all(p.requires_grad for p in specialist.parameters())


def test_composite_loss_backward():
    """Verify SoftMoECompositeLoss computes all 4 terms and backpropagates valid gradients."""
    criterion = SoftMoECompositeLoss(
        lambda_recon=0.8,
        lambda_class=0.2,
        lambda_balance=0.1,
        lambda_entropy=0.01,
        alpha=0.90
    )

    model = SoftMoERestorationNetwork(
        gate_base_channels=16,
        specialist_base_channels=16,
        specialist_bottleneck_dim=32
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    corrupted = torch.rand(4, 3, 128, 128)
    clean = torch.rand(4, 3, 128, 128)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    optimizer.zero_grad()
    reconstructed, weights, logits = model(corrupted)
    loss, breakdown = criterion(reconstructed, clean, logits, weights, labels)

    assert loss.item() > 0.0, "Total loss must be positive"
    assert "l_recon" in breakdown and "l_class" in breakdown
    assert "l_balance" in breakdown and "l_entropy" in breakdown
    assert breakdown["l_balance"] >= 0.0
    assert breakdown["l_entropy"] >= 0.0

    loss.backward()

    # Check gradients exist across both gate and specialist parameters
    gate_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gate.parameters())
    sp_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.specialist_sp.parameters())
    assert gate_has_grad, "Gate parameters must receive non-zero gradients"
    assert sp_has_grad, "Specialist parameters must receive non-zero gradients"


def test_soft_moe_onnx_export_and_parity():
    """Verify SoftMoERestorationNetwork exports cleanly to ONNX with multi-output numerical parity."""
    model = SoftMoERestorationNetwork(
        gate_base_channels=16,
        specialist_base_channels=16,
        specialist_bottleneck_dim=32
    )
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "test_soft_moe.onnx")
        export_to_onnx(
            model=model,
            onnx_output_path=onnx_path,
            input_shape=(1, 3, 128, 128),
            opset_version=18,
            input_names=["corrupted_image"],
            output_names=["restored_image", "routing_weights", "logits"]
        )

        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 100 * 1024, "ONNX file must be valid size"

        # Verify numerical equivalence on sample batch
        sample_batch = torch.rand(2, 3, 128, 128)
        verification = verify_onnx_numerical_equivalence(
            model=model,
            onnx_path=onnx_path,
            sample_batch=sample_batch,
            atol=1e-3,
            rtol=1e-2
        )
        assert verification["equivalent"], f"ONNX numerical equivalence failed: max diff = {verification['max_abs_diff']}"
