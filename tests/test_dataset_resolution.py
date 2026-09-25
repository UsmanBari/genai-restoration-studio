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


def test_capitalized_breed_name_resolution(tmp_path):
    """Verifies that _load_image looks for exact capitalized filenames (e.g. Bengal_96.jpg)."""
    from PIL import Image
    import json
    from data.oxford_pet import OxfordPetDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()

    # Create real image with capitalized name (Bengal_96.jpg)
    img_file = img_dir / "Bengal_96.jpg"
    Image.new('RGB', (128, 128), color=(200, 100, 50)).save(str(img_file))

    # Manifest with exact capitalized name
    manifest_data = [{
        'image_id': 'Bengal_96',
        'filename': 'Bengal_96.jpg',
        'class_id': 6,
        'species': 1,
        'breed_id': 6,
        'split_source': 'trainval',
        'split': 'train'
    }]
    m_path = manifest_dir / "oxford_train_manifest.json"
    with open(str(m_path), 'w') as f:
        json.dump(manifest_data, f)

    ds = OxfordPetDataset(manifest_path=str(manifest_dir), images_dir=str(img_dir), split='train')
    assert len(ds) == 1
    sample = ds[0]
    assert sample['clean'].shape == (3, 128, 128)

    # Test case-variant resilience (if manifest had lowercased 'bengal_96.jpg', it resolves 'Bengal_96.jpg' on disk)
    lower_manifest = [{
        'image_id': 'bengal_96',
        'filename': 'bengal_96.jpg',
        'class_id': 6,
        'species': 1,
        'breed_id': 6,
        'split_source': 'trainval',
        'split': 'train'
    }]
    with open(str(m_path), 'w') as f:
        json.dump(lower_manifest, f)

    ds_lower = OxfordPetDataset(manifest_path=str(manifest_dir), images_dir=str(img_dir), split='train')
    sample_lower = ds_lower[0]
    assert sample_lower['clean'].shape == (3, 128, 128)

