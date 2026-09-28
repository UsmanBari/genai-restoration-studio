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
    assert len(train_ds) == 898, f"Unexpected train count: {len(train_ds)}"
    assert len(val_ds) == 160, f"Unexpected val count: {len(val_ds)}"
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


def test_fs2k_uppercase_jpg_and_png_extension_resolution(tmp_path):
    """Verifies that FS2KDataset resolves uppercase .JPG photo (e.g. photo3/image0449.JPG), .png sketch, and preserves distinct folder boundaries."""
    from PIL import Image
    import json
    from data.fs2k import FS2KDataset

    fs2k_root = tmp_path / "FS2K"
    
    # Subfolder 1: photo1 (.jpg photo, .jpg sketch)
    photo1_dir = fs2k_root / "photo" / "photo1"
    sketch1_dir = fs2k_root / "sketch" / "sketch1"
    photo1_dir.mkdir(parents=True)
    sketch1_dir.mkdir(parents=True)
    
    photo1_file = photo1_dir / "image0449.jpg"
    Image.new('RGB', (128, 128), color=(100, 150, 200)).save(str(photo1_file))
    sketch1_file = sketch1_dir / "sketch0449.jpg"
    Image.new('RGB', (128, 128), color=(30, 30, 30)).save(str(sketch1_file))

    # Subfolder 3: photo3 (.JPG photo, .png sketch)
    photo3_dir = fs2k_root / "photo" / "photo3"
    sketch3_dir = fs2k_root / "sketch" / "sketch3"
    photo3_dir.mkdir(parents=True)
    sketch3_dir.mkdir(parents=True)

    photo3_file = photo3_dir / "image0449.JPG"
    Image.new('RGB', (128, 128), color=(200, 120, 60)).save(str(photo3_file))
    sketch3_file = sketch3_dir / "sketch0449.png"
    Image.new('RGB', (128, 128), color=(70, 70, 70)).save(str(sketch3_file))

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()

    manifest_data = [
        {
            'image_name': 'photo1/image0449',
            'style': 0,
            'split': 'train'
        },
        {
            'image_name': 'photo3/image0449',
            'style': 2,
            'split': 'train'
        }
    ]
    m_path = manifest_dir / "fs2k_train_manifest.json"
    with open(str(m_path), 'w') as f:
        json.dump(manifest_data, f)

    ds = FS2KDataset(manifest_path=str(manifest_dir), fs2k_root=str(fs2k_root), split='train')
    assert len(ds) == 2

    # Verify item 0 (photo1/image0449.jpg)
    p0_path, s0_path = ds._resolve_paths(ds.items[0])
    assert os.path.basename(p0_path) == "image0449.jpg"
    assert os.path.basename(s0_path) == "sketch0449.jpg"
    assert "photo1" in p0_path
    sample0 = ds[0]
    assert sample0['photo'].shape == (3, 128, 128)
    assert sample0['sketch'].shape == (3, 128, 128)
    assert sample0['style'] == 0
    assert float(sample0['photo'].std()) > 0.0

    # Verify item 1 (photo3/image0449.JPG)
    p1_path, s1_path = ds._resolve_paths(ds.items[1])
    assert os.path.basename(p1_path) == "image0449.JPG"
    assert os.path.basename(s1_path) == "sketch0449.png"
    assert "photo3" in p1_path
    sample1 = ds[1]
    assert sample1['photo'].shape == (3, 128, 128)
    assert sample1['sketch'].shape == (3, 128, 128)
    assert sample1['style'] == 2
    assert float(sample1['photo'].std()) > 0.0
    assert float(sample1['sketch'].std()) > 0.0

