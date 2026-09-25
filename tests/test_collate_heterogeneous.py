"""
Unit test verifying heterogeneous corruption batching with collate_oxford.
Directly tests mixing Clean, S&P, Blur, and Occlusion samples in a single batch
with mismatched metadata dictionary keys.
"""

import pytest
import numpy as np
from data.oxford_pet import collate_oxford, OxfordPetDataset


def test_collate_heterogeneous_corruption_types():
    # Construct 4 sample items with different metadata fields matching real manifests
    sample_clean = {
        'corrupted': np.full((3, 128, 128), 0.5, dtype=np.float32),
        'clean': np.full((3, 128, 128), 0.5, dtype=np.float32),
        'label': 0,
        'metadata': {
            'image_id': 'img_clean',
            'corruption_type': 'clean',
            'severity_tier': 0,
            'seed': 42
        }
    }

    sample_sp = {
        'corrupted': np.full((3, 128, 128), 0.5, dtype=np.float32),
        'clean': np.full((3, 128, 128), 0.5, dtype=np.float32),
        'label': 1,
        'metadata': {
            'image_id': 'img_sp',
            'corruption_type': 'salt_and_pepper',
            'severity_tier': 2,
            'prob': 0.08,
            'seed': 43
        }
    }

    sample_blur = {
        'corrupted': np.full((3, 128, 128), 0.5, dtype=np.float32),
        'clean': np.full((3, 128, 128), 0.5, dtype=np.float32),
        'label': 2,
        'metadata': {
            'image_id': 'img_blur',
            'corruption_type': 'gaussian_blur',
            'severity_tier': 3,
            'kernel_size': 7,
            'sigma': 2.5,
            'seed': 44
        }
    }

    sample_occlusion = {
        'corrupted': np.full((3, 128, 128), 0.5, dtype=np.float32),
        'clean': np.full((3, 128, 128), 0.5, dtype=np.float32),
        'label': 3,
        'metadata': {
            'image_id': 'img_occ',
            'corruption_type': 'rectangular_occlusion',
            'severity_tier': 1,
            'num_rects': 1,
            'coverage': 0.10,
            'rectangles': [{'x1': 10, 'y1': 10, 'x2': 40, 'y2': 40}],
            'seed': 45
        }
    }

    batch_input = [sample_clean, sample_sp, sample_blur, sample_occlusion]

    # Run collate_oxford
    collated = collate_oxford(batch_input)

    assert 'corrupted' in collated
    assert 'clean' in collated
    assert 'label' in collated
    assert 'metadata' in collated

    assert len(collated['corrupted']) == 4
    assert len(collated['metadata']) == 4

    # Verify heterogeneous metadata preserved without KeyError
    assert collated['metadata'][0]['corruption_type'] == 'clean'
    assert 'kernel_size' not in collated['metadata'][0]

    assert collated['metadata'][1]['corruption_type'] == 'salt_and_pepper'
    assert collated['metadata'][1]['prob'] == 0.08

    assert collated['metadata'][2]['corruption_type'] == 'gaussian_blur'
    assert collated['metadata'][2]['kernel_size'] == 7
    assert collated['metadata'][2]['sigma'] == 2.5

    assert collated['metadata'][3]['corruption_type'] == 'rectangular_occlusion'
    assert collated['metadata'][3]['num_rects'] == 1
    assert len(collated['metadata'][3]['rectangles']) == 1


def test_test_manifest_iteration_with_collate():
    manifest_dir = "configs/manifests"
    images_dir = "data/raw/OxfordPet/images_128x128"

    test_ds = OxfordPetDataset(
        manifest_path=manifest_dir,
        images_dir=images_dir,
        split='test'
    )

    # Simulate batching 16 items from test set (which mixes clean, sp, blur, occlusion)
    first_16 = [test_ds[i] for i in range(16)]
    collated = collate_oxford(first_16)

    assert len(collated['metadata']) == 16
    assert len(collated['corrupted']) == 16
    types_in_batch = set(m['corruption_type'] for m in collated['metadata'])
    # In test manifest, tiers cycle every sample so all 4 corruption types are in the first 16 samples!
    assert len(types_in_batch) == 4, f"Expected all 4 corruption types in batch, got {types_in_batch}"
