"""
Manifest generator for Oxford-IIIT Pet and FS2K datasets.
Creates deterministic, reproducible JSON manifests for train, validation, and test sets.
"""

import os
import json
import math
import random
import numpy as np
from typing import List, Dict, Any

from data.corruptions import generate_occlusion_rectangles, CORRUPTION_NAMES


def create_oxford_pet_manifests(
    image_entries: List[Dict[str, Any]],
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    img_h: int = 128,
    img_w: int = 128
):
    """
    Splits Oxford-IIIT Pet trainval entries 80% train / 20% val with seed 42.
    Test set entries are kept separate.
    Generates deterministic corruption parameters for val and test manifests.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Separate trainval entries vs test entries if annotated
    trainval_entries = [e for e in image_entries if e.get('split_source') == 'trainval']
    test_entries = [e for e in image_entries if e.get('split_source') == 'test']

    if not trainval_entries and not test_entries:
        # If no split_source provided, split all provided entries
        all_shuffled = list(image_entries)
        rng.shuffle(all_shuffled)
        n_train = int(len(all_shuffled) * train_ratio)
        train_list = all_shuffled[:n_train]
        val_raw_list = all_shuffled[n_train:]
        test_list = []
    else:
        rng.shuffle(trainval_entries)
        n_train = int(len(trainval_entries) * train_ratio)
        train_list = trainval_entries[:n_train]
        val_raw_list = trainval_entries[n_train:]
        test_list = list(test_entries)

    # 1. Train manifest (Clean references, corruptions generated on-the-fly at runtime)
    train_manifest = []
    for idx, item in enumerate(train_list):
        entry = dict(item)
        entry['split'] = 'train'
        train_manifest.append(entry)

    # 2. Validation manifest (Deterministic corruptions across all 4 types)
    val_manifest = []
    for idx, item in enumerate(val_raw_list):
        entry = dict(item)
        entry['split'] = 'val'
        # Cycle or sample corruption condition deterministically
        corr_label = idx % 4
        corr_type = CORRUPTION_NAMES[corr_label]
        item_seed = seed + idx * 17
        item_rng = np.random.default_rng(item_seed)

        entry['corruption_type'] = corr_type
        entry['label'] = corr_label
        entry['seed'] = item_seed

        if corr_label == 1:  # S&P
            entry['prob'] = float(item_rng.uniform(0.02, 0.15))
        elif corr_label == 2:  # Blur
            entry['kernel_size'] = int(item_rng.choice([3, 5, 7]))
            entry['sigma'] = float(item_rng.uniform(0.5, 2.5))
        elif corr_label == 3:  # Occlusion
            num_rects = int(item_rng.integers(1, 4))
            coverage = float(item_rng.uniform(0.10, 0.35))
            rects = generate_occlusion_rectangles(img_h, img_w, num_rects, coverage, rng=item_rng)
            entry['num_rects'] = num_rects
            entry['coverage'] = coverage
            entry['rectangles'] = rects

        val_manifest.append(entry)

    # 3. Test manifest (Deterministic with THREE FIXED SEVERITY TIERS per corruption type + clean)
    test_manifest = []
    fixed_tiers = [
        {'type': 'clean', 'label': 0, 'tier': 0},
        # S&P tiers: 0.03, 0.08, 0.15
        {'type': 'salt_and_pepper', 'label': 1, 'tier': 1, 'prob': 0.03},
        {'type': 'salt_and_pepper', 'label': 1, 'tier': 2, 'prob': 0.08},
        {'type': 'salt_and_pepper', 'label': 1, 'tier': 3, 'prob': 0.15},
        # Blur tiers: (3, 0.7), (5, 1.5), (7, 2.5)
        {'type': 'gaussian_blur', 'label': 2, 'tier': 1, 'kernel_size': 3, 'sigma': 0.7},
        {'type': 'gaussian_blur', 'label': 2, 'tier': 2, 'kernel_size': 5, 'sigma': 1.5},
        {'type': 'gaussian_blur', 'label': 2, 'tier': 3, 'kernel_size': 7, 'sigma': 2.5},
        # Occlusion tiers: ~10% (1 rect), ~20% (2 rects), ~35% (3 rects)
        {'type': 'rectangular_occlusion', 'label': 3, 'tier': 1, 'num_rects': 1, 'coverage': 0.10},
        {'type': 'rectangular_occlusion', 'label': 3, 'tier': 2, 'num_rects': 2, 'coverage': 0.20},
        {'type': 'rectangular_occlusion', 'label': 3, 'tier': 3, 'num_rects': 3, 'coverage': 0.35},
    ]

    for idx, item in enumerate(test_list):
        entry = dict(item)
        entry['split'] = 'test'
        tier_cfg = fixed_tiers[idx % len(fixed_tiers)]
        entry['corruption_type'] = tier_cfg['type']
        entry['label'] = tier_cfg['label']
        entry['severity_tier'] = tier_cfg['tier']
        item_seed = seed + 10000 + idx * 13
        entry['seed'] = item_seed
        item_rng = np.random.default_rng(item_seed)

        if tier_cfg['type'] == 'salt_and_pepper':
            entry['prob'] = tier_cfg['prob']
        elif tier_cfg['type'] == 'gaussian_blur':
            entry['kernel_size'] = tier_cfg['kernel_size']
            entry['sigma'] = tier_cfg['sigma']
        elif tier_cfg['type'] == 'rectangular_occlusion':
            entry['num_rects'] = tier_cfg['num_rects']
            entry['coverage'] = tier_cfg['coverage']
            rects = generate_occlusion_rectangles(img_h, img_w, tier_cfg['num_rects'], tier_cfg['coverage'], rng=item_rng)
            entry['rectangles'] = rects

        test_manifest.append(entry)

    # Save to disk
    train_path = os.path.join(output_dir, 'oxford_train_manifest.json')
    val_path = os.path.join(output_dir, 'oxford_val_manifest.json')
    test_path = os.path.join(output_dir, 'oxford_test_manifest.json')

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_manifest, f, indent=2)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_manifest, f, indent=2)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_manifest, f, indent=2)

    return train_path, val_path, test_path


def create_fs2k_manifests(
    anno_train_items: List[Dict[str, Any]],
    anno_test_items: List[Dict[str, Any]],
    output_dir: str,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Creates FS2K manifests based on official anno_train.json and anno_test.json.
    From official train (1058 items), reserves 15% as validation stratified by the 'style' field (0/1/2).
    Official test set (1046 items) is untouched.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    # Group official train items by style (0, 1, 2)
    by_style = {}
    for item in anno_train_items:
        style = item.get('style', 0)
        by_style.setdefault(style, []).append(item)

    train_split = []
    val_split = []

    for style, items in sorted(by_style.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = int(math.ceil(len(shuffled) * val_ratio))
        val_items = shuffled[:n_val]
        train_items = shuffled[n_val:]

        for it in train_items:
            entry = dict(it)
            entry['split'] = 'train'
            train_split.append(entry)

        for it in val_items:
            entry = dict(it)
            entry['split'] = 'val'
            val_split.append(entry)

    test_split = []
    for it in anno_test_items:
        entry = dict(it)
        entry['split'] = 'test'
        test_split.append(entry)

    train_path = os.path.join(output_dir, 'fs2k_train_manifest.json')
    val_path = os.path.join(output_dir, 'fs2k_val_manifest.json')
    test_path = os.path.join(output_dir, 'fs2k_test_manifest.json')

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_split, f, indent=2)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_split, f, indent=2)
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_split, f, indent=2)

    return train_path, val_path, test_path
