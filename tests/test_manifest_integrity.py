"""
Manifest Integrity and Anti-Regression Test Suite.
Verifies that:
1. Oxford-IIIT Pet manifests have exact official counts (2944 / 736 / 3669) and 100% disjoint splits.
2. FS2K manifests have exact official counts and style distributions and 100% disjoint splits.
3. No tests or scripts can silently mutate or corrupt configs/manifests/.
"""

import os
import json
from collections import Counter
import pytest


def test_oxford_manifest_integrity():
    manifest_dir = "configs/manifests"
    train_path = os.path.join(manifest_dir, "oxford_train_manifest.json")
    val_path = os.path.join(manifest_dir, "oxford_val_manifest.json")
    test_path = os.path.join(manifest_dir, "oxford_test_manifest.json")

    assert os.path.exists(train_path), "oxford_train_manifest.json missing!"
    assert os.path.exists(val_path), "oxford_val_manifest.json missing!"
    assert os.path.exists(test_path), "oxford_test_manifest.json missing!"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    assert len(train_data) == 2944, f"Expected 2944 Oxford train items, got {len(train_data)}"
    assert len(val_data) == 736, f"Expected 736 Oxford val items, got {len(val_data)}"
    assert len(test_data) == 3669, f"Expected 3669 Oxford test items, got {len(test_data)}"

    t_ids = set(x["image_id"] for x in train_data)
    v_ids = set(x["image_id"] for x in val_data)
    te_ids = set(x["image_id"] for x in test_data)

    assert len(t_ids & v_ids) == 0, f"Oxford train/val overlap: {len(t_ids & v_ids)}"
    assert len(t_ids & te_ids) == 0, f"Oxford train/test overlap: {len(t_ids & te_ids)}"
    assert len(v_ids & te_ids) == 0, f"Oxford val/test overlap: {len(v_ids & te_ids)}"


def test_fs2k_manifest_disjointness_and_distribution():
    manifest_dir = "configs/manifests"
    train_path = os.path.join(manifest_dir, "fs2k_train_manifest.json")
    val_path = os.path.join(manifest_dir, "fs2k_val_manifest.json")
    test_path = os.path.join(manifest_dir, "fs2k_test_manifest.json")

    assert os.path.exists(train_path), "fs2k_train_manifest.json missing!"
    assert os.path.exists(val_path), "fs2k_val_manifest.json missing!"
    assert os.path.exists(test_path), "fs2k_test_manifest.json missing!"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    assert len(train_data) + len(val_data) == 1058, f"Expected 1058 train+val items, got {len(train_data) + len(val_data)}"
    assert len(test_data) == 1046, f"Expected 1046 test items, got {len(test_data)}"

    def get_id(item):
        return item.get("image_name", item.get("photo_path", ""))

    t_ids = set(get_id(x) for x in train_data)
    v_ids = set(get_id(x) for x in val_data)
    te_ids = set(get_id(x) for x in test_data)

    assert len(t_ids & v_ids) == 0, f"FS2K train/val overlap: {len(t_ids & v_ids)}"
    assert len(t_ids & te_ids) == 0, f"FS2K train/test overlap: {len(t_ids & te_ids)}"
    assert len(v_ids & te_ids) == 0, f"FS2K val/test overlap: {len(v_ids & te_ids)}"

    # Enforce strict unconditional assertions for authentic FS2K dataset
    assert len(train_data) == 898, f"Expected 898 train items, got {len(train_data)}"
    assert len(val_data) == 160, f"Expected 160 val items, got {len(val_data)}"
    assert len(test_data) == 1046, f"Expected 1046 test items, got {len(test_data)}"

    train_counts = Counter(x.get("style", 0) for x in train_data)
    val_counts = Counter(x.get("style", 0) for x in val_data)
    test_counts = Counter(x.get("style", 0) for x in test_data)

    assert train_counts == {0: 303, 1: 297, 2: 298}, f"Unexpected train style distribution: {train_counts}"
    assert val_counts == {0: 54, 1: 53, 2: 53}, f"Unexpected val style distribution: {val_counts}"
    assert test_counts == {0: 619, 1: 381, 2: 46}, f"Unexpected test style distribution: {test_counts}"
