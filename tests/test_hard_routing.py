"""
Unit tests for Task 2: Hard Routing Restoration Pipeline.
"""

import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_hard_routing_identity_bypass_on_clean():
    """
    Verifies that clean inputs routed to Class 0 experience exact identity bypass
    with zero distortion (output == input).
    """
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed in test environment")

    from models.hard_router import HardRoutingRestorationPipeline

    pipeline = HardRoutingRestorationPipeline()

    clean_input = torch.rand(2, 3, 128, 128)
    oracle_labels = torch.tensor([0, 0], dtype=torch.long)

    # Route via oracle
    restored_oracle = pipeline.route_oracle(clean_input, oracle_labels)
    assert torch.equal(restored_oracle, clean_input), "Clean images must bypass without alteration in Oracle routing."


def test_hard_routing_mixed_batch_dispatch():
    """
    Verifies that a mixed batch containing clean, S&P, blur, and occlusion
    correctly invokes the designated specialist or identity bypass for each element.
    """
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed in test environment")

    from models.autoencoders import UniversalAutoencoder
    from models.classifiers import CorruptionClassifier
    from models.hard_router import HardRoutingRestorationPipeline

    sp_spec = UniversalAutoencoder(base_channels=16, bottleneck_dim=64)
    blur_spec = UniversalAutoencoder(base_channels=16, bottleneck_dim=64)
    occ_spec = UniversalAutoencoder(base_channels=16, bottleneck_dim=64)

    specialists = {
        'salt_and_pepper': sp_spec,
        'gaussian_blur': blur_spec,
        'rectangular_occlusion': occ_spec
    }

    classifier = CorruptionClassifier(base_channels=16)
    pipeline = HardRoutingRestorationPipeline(classifier=classifier, specialists=specialists)

    mixed_input = torch.rand(4, 3, 128, 128)
    mixed_labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    # Test Oracle routing on mixed batch
    restored_oracle = pipeline.route_oracle(mixed_input, mixed_labels)
    assert restored_oracle.shape == (4, 3, 128, 128)
    assert torch.equal(restored_oracle[0:1], mixed_input[0:1]), "Sample 0 (clean) must be exact identity."
    assert not torch.equal(restored_oracle[1:2], mixed_input[1:2]), "Sample 1 (S&P) must be transformed by specialist."

    # Test Predicted routing on mixed batch
    restored_pred, pred_classes, probs = pipeline.route_predicted(mixed_input)
    assert restored_pred.shape == (4, 3, 128, 128)
    assert pred_classes.shape == (4,)
    assert probs.shape == (4, 4)
