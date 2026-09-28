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


def test_collate_fs2k_batch_16_with_benchmark():
    """
    Directly tests batch_size 16 on FS2K dataset with non-scalar annotation fields
    (skin_color [2], lip_color [3], eye_color [3]) and executes run_cgan_benchmark.
    Verifies no IndexError or collate mismatch occurs.
    """
    import os
    import json
    import tempfile
    import shutil
    import torch
    from PIL import Image
    from data.fs2k import FS2KDataset, collate_fs2k, get_fs2k_dataloaders
    from models.cgan import StyleConditionedUNetGenerator
    from evaluation.benchmark_cgan import run_cgan_benchmark

    tmp_dir = tempfile.mkdtemp(prefix="fs2k_collate_test_")
    try:
        manifest_dir = os.path.join(tmp_dir, "manifests")
        data_root = os.path.join(tmp_dir, "FS2K")
        os.makedirs(manifest_dir, exist_ok=True)
        os.makedirs(os.path.join(data_root, "photo", "photo1"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "sketch", "sketch1"), exist_ok=True)

        items = []
        for i in range(16):
            st = i % 3
            p_name = f"image_{i:04d}.jpg"
            s_name = f"sketch_{i:04d}.jpg"
            p_full = os.path.join(data_root, "photo", "photo1", p_name)
            s_full = os.path.join(data_root, "sketch", "sketch1", s_name)

            # Create dummy images
            Image.fromarray((np.random.RandomState(i).uniform(0, 255, (128, 128, 3))).astype(np.uint8)).save(p_full)
            Image.fromarray((np.random.RandomState(i + 100).uniform(0, 255, (128, 128, 3))).astype(np.uint8)).save(s_full)

            # Realistic non-scalar list attributes
            items.append({
                "image_name": f"photo1/image_{i:04d}",
                "style": st,
                "skin_color": [156, 137],
                "lip_color": [197, 125, 109],
                "eye_color": [42, 33, 29],
                "gender": 1 if i % 2 == 0 else 0,
                "hair": 2,
                "split": "test"
            })

        test_manifest = os.path.join(manifest_dir, "fs2k_test_manifest.json")
        with open(test_manifest, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)

        # 1. Test collate_fs2k on batch of 16
        ds = FS2KDataset(manifest_path=test_manifest, fs2k_root=data_root, split='test', normalize_gan=True)
        assert len(ds) == 16
        sample_batch_list = [ds[i] for i in range(16)]
        collated = collate_fs2k(sample_batch_list)

        assert collated['photo'].shape == (16, 3, 128, 128)
        assert collated['sketch'].shape == (16, 3, 128, 128)
        assert collated['style'].shape == (16,)
        assert len(collated['image_name']) == 16
        assert len(collated['metadata']) == 16
        # Check value range is [-1, 1]
        assert collated['photo'].min() >= -1.0 and collated['photo'].max() <= 1.0

        # 2. Test DataLoader execution with batch_size=16
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fs2k)
        batch = next(iter(loader))
        assert batch['photo'].shape == (16, 3, 128, 128)
        assert len(batch['metadata']) == 16

        # 3. Test run_cgan_benchmark on this dataset
        net_g = StyleConditionedUNetGenerator(base_channels=16, emb_dim=8, emb_channels=4)
        net_g.eval()
        eval_out = os.path.join(tmp_dir, "eval_out")
        bench_res = run_cgan_benchmark(
            generator=net_g,
            manifest_path=test_manifest,
            fs2k_root=data_root,
            device="cpu",
            output_dir=eval_out,
            num_visualizations=4
        )

        assert 'summary' in bench_res
        assert bench_res['summary']['total_test_samples'] == 16
        assert 'baselines' in bench_res
        assert os.path.exists(os.path.join(eval_out, "cgan_benchmark_results.json"))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

