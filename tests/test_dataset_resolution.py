"""
Smoke tests for OxfordPetDataset and FS2KDataset manifest resolution and loading.
Catches any path or zero-length dataset issues locally.
"""

import os
import pytest
from data.oxford_pet import OxfordPetDataset, resolve_manifest_path
from data.fs2k import FS2KDataset


def test_oxford_pet_dataset_instantiation():
    manifest_dir = "configs/manifests"
    images_dir = "data/raw/OxfordPet/images_128x128"

    # Test directory-based instantiation
    train_ds = OxfordPetDataset(manifest_path=manifest_dir, images_dir=images_dir, split='train')
    val_ds = OxfordPetDataset(manifest_path=manifest_dir, images_dir=images_dir, split='val')
    test_ds = OxfordPetDataset(manifest_path=manifest_dir, images_dir=images_dir, split='test')

    assert len(train_ds) == 2944, f"Expected 2944 train items, got {len(train_ds)}"
    assert len(val_ds) == 736, f"Expected 736 val items, got {len(val_ds)}"
    assert len(test_ds) == 3669, f"Expected 3669 test items, got {len(test_ds)}"

    # Test direct file path instantiation
    direct_train_path = os.path.join(manifest_dir, "oxford_train_manifest.json")
    train_ds_direct = OxfordPetDataset(manifest_path=direct_train_path, images_dir=images_dir, split='train')
    assert len(train_ds_direct) == 2944


def test_fs2k_dataset_instantiation():
    manifest_dir = "configs/manifests"
    fs2k_root = "data/raw/FS2K"

    train_ds = FS2KDataset(manifest_path=manifest_dir, fs2k_root=fs2k_root, split='train')
    val_ds = FS2KDataset(manifest_path=manifest_dir, fs2k_root=fs2k_root, split='val')
    test_ds = FS2KDataset(manifest_path=manifest_dir, fs2k_root=fs2k_root, split='test')

    assert len(train_ds) + len(val_ds) == 1058, f"Expected 1058 total train+val items"
    assert len(train_ds) in [898, 899], f"Unexpected train count: {len(train_ds)}"
    assert len(val_ds) in [159, 160], f"Unexpected val count: {len(val_ds)}"
    assert len(test_ds) == 1046, f"Expected 1046 test items, got {len(test_ds)}"


def test_invalid_manifest_path_raises():
    with pytest.raises(FileNotFoundError):
        resolve_manifest_path("non_existent_folder_xyz_123", "dummy_manifest.json")
