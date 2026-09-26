"""
Unit tests for Task 2: Corruption Classifier and Specialist Anti-Contamination.
"""

import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_classifier_forward_and_predict():
    """Verifies CorruptionClassifier forward pass shapes and probability normalization."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed in test environment")

    from models.classifiers import CorruptionClassifier

    batch_size = 4
    model = CorruptionClassifier(in_channels=3, num_classes=4, base_channels=16, dropout_rate=0.2)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 128, 128)
    logits = model(dummy_input)
    assert logits.shape == (batch_size, 4), f"Expected shape ({batch_size}, 4), got {logits.shape}"

    preds, probs = model.predict(dummy_input)
    assert preds.shape == (batch_size,)
    assert probs.shape == (batch_size, 4)
    # Verify probability distribution
    prob_sums = probs.sum(dim=1).numpy()
    np.testing.assert_allclose(prob_sums, np.ones(batch_size), atol=1e-5)


def test_classifier_backward_step():
    """Verifies loss computation and gradient backpropagation for CorruptionClassifier."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed in test environment")

    from models.classifiers import CorruptionClassifier

    model = CorruptionClassifier(in_channels=3, num_classes=4, base_channels=16, dropout_rate=0.1)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.randn(4, 3, 128, 128)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(dummy_input)
    loss = criterion(logits, targets)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient after backward pass"


def test_specialist_batch_anti_contamination_assertion():
    """
    Verifies that train_one_epoch_specialist raises AssertionError if a batch
    contains any contaminated labels (e.g. Clean or Occlusion samples in Gaussian Blur specialist).
    """
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed in test environment")

    from models.autoencoders import UniversalAutoencoder
    from training.losses import RestorationLoss
    from training.trainer_specialist import train_one_epoch_specialist

    model = UniversalAutoencoder(base_channels=16, bottleneck_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = RestorationLoss(alpha=0.8)

    # Construct a contaminated dummy batch for "gaussian_blur" (expected label 2)
    contaminated_batch = {
        'corrupted': torch.randn(4, 3, 128, 128),
        'clean': torch.randn(4, 3, 128, 128),
        'label': torch.tensor([2, 2, 0, 2], dtype=torch.long)  # Label 0 (clean) is a contaminant!
    }

    class DummyLoader:
        def __iter__(self):
            return iter([contaminated_batch])
        def __len__(self):
            return 1

    with pytest.raises(AssertionError, match="CRITICAL CORRUPTION CONTAMINATION DETECTED"):
        train_one_epoch_specialist(
            model=model,
            dataloader=DummyLoader(),
            optimizer=optimizer,
            criterion=criterion,
            scaler=None,
            target_corruption="gaussian_blur",
            device="cpu"
        )


def test_dataset_single_corruption_mode_filtering():
    """
    Verifies that OxfordPetDataset with corruption_mode yields 100% targeted single-corruption samples.
    """
    import os
    from data.oxford_pet import OxfordPetDataset

    os.environ["ALLOW_DUMMY_DATASET_IMAGES"] = "1"
    manifest_dir = "configs/manifests"

    for mode, expected_label in [
        ("salt_and_pepper", 1),
        ("gaussian_blur", 2),
        ("rectangular_occlusion", 3)
    ]:
        ds = OxfordPetDataset(
            manifest_path=manifest_dir,
            images_dir="data/raw/OxfordPet/images_128x128",
            split="train",
            corruption_mode=mode
        )
        # Sample 10 items from dataset and verify label
        for i in range(min(10, len(ds))):
            sample = ds[i]
            assert sample['label'] == expected_label, (
                f"Expected label {expected_label} for mode '{mode}', got {sample['label']}"
            )
