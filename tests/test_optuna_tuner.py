"""
Unit and smoke tests for training/optuna_tuner.py dataset loading logic.
"""

import os
import pytest
from data.oxford_pet import OxfordPetDataset, get_oxford_dataloaders, resolve_manifest_path


def test_optuna_dataset_resolution_from_manifest_dir():
    manifest_dir = "configs/manifests"
    images_dir = "data/raw/OxfordPet/images_128x128"

    train_path = resolve_manifest_path(manifest_dir, "oxford_train_manifest.json")
    val_path = resolve_manifest_path(manifest_dir, "oxford_val_manifest.json")

    assert os.path.exists(train_path)
    assert os.path.exists(val_path)

    train_ds = OxfordPetDataset(manifest_path=train_path, images_dir=images_dir, split='train')
    val_ds = OxfordPetDataset(manifest_path=val_path, images_dir=images_dir, split='val')

    assert len(train_ds) == 2944, f"Expected 2944 train items, got {len(train_ds)}"
    assert len(val_ds) == 736, f"Expected 736 val items, got {len(val_ds)}"


def test_optuna_dataset_resolution_with_subpath():
    manifest_dir = "configs/manifests"
    images_dir = "data/raw/OxfordPet/images_128x128"

    # Test passing os.path.join(manifest_dir, 'oxford_train_manifest.json')
    train_file = os.path.join(manifest_dir, 'oxford_train_manifest.json')
    val_file = os.path.join(manifest_dir, 'oxford_val_manifest.json')

    train_ds = OxfordPetDataset(manifest_path=train_file, images_dir=images_dir, split='train')
    val_ds = OxfordPetDataset(manifest_path=val_file, images_dir=images_dir, split='val')

    assert len(train_ds) == 2944
    assert len(val_ds) == 736
