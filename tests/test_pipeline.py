"""
Unit tests for data corruptions, manifests, and backend endpoints.
"""

import os
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from data.corruptions import (
    apply_salt_and_pepper,
    apply_gaussian_blur,
    apply_rectangular_occlusion,
    apply_random_corruption_runtime,
    apply_deterministic_corruption
)
from data.manifest_generator import create_oxford_pet_manifests, create_fs2k_manifests
from backend.main import app


def test_runtime_corruptions():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    for _ in range(50):
        corr, lbl, params = apply_random_corruption_runtime(img)
        assert corr.shape == (128, 128, 3)
        assert lbl in [0, 1, 2, 3]
        assert corr.dtype == np.uint8


def test_deterministic_corruptions():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    # S&P
    sp_meta = {'corruption_type': 'salt_and_pepper', 'prob': 0.08, 'seed': 42}
    res1 = apply_deterministic_corruption(img, sp_meta)
    res2 = apply_deterministic_corruption(img, sp_meta)
    assert np.array_equal(res1, res2)

    # Blur
    blur_meta = {'corruption_type': 'gaussian_blur', 'kernel_size': 5, 'sigma': 1.5}
    b1 = apply_deterministic_corruption(img, blur_meta)
    b2 = apply_deterministic_corruption(img, blur_meta)
    assert np.array_equal(b1, b2)


def test_oxford_manifest_generation(tmp_path):
    entries = []
    for i in range(100):
        entries.append({
            'image_id': f'img_{i}',
            'filename': f'img_{i}.jpg',
            'split_source': 'trainval' if i < 80 else 'test'
        })
    out_dir = str(tmp_path / 'manifests')
    t_p, v_p, te_p = create_oxford_pet_manifests(entries, out_dir, train_ratio=0.8, seed=42)

    with open(t_p) as f:
        t = json.load(f)
    with open(v_p) as f:
        v = json.load(f)
    with open(te_p) as f:
        te = json.load(f)

    assert len(t) == 64
    assert len(v) == 16
    assert len(te) == 20


def test_fs2k_stratified_manifest_generation(tmp_path):
    train_items = [{'image_name': f'photo1/img_{i}', 'style': i % 3} for i in range(100)]
    test_items = [{'image_name': f'photo1/img_{i+100}', 'style': i % 3} for i in range(50)]

    out_dir = str(tmp_path / 'fs2k_manifests')
    t_p, v_p, te_p = create_fs2k_manifests(train_items, test_items, out_dir, val_ratio=0.15, seed=42)

    with open(t_p) as f:
        t = json.load(f)
    with open(v_p) as f:
        v = json.load(f)
    with open(te_p) as f:
        te = json.load(f)

    assert len(t) + len(v) == 100
    assert len(te) == 50


def test_backend_endpoints():
    client = TestClient(app)

    # Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Dummy image upload test
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (128, 128), color='red').save(buf, format='PNG')
    buf.seek(0)

    # Universal restoration stub
    resp = client.post("/universal-restoration", files={"file": ("test.png", buf, "image/png")})
    assert resp.status_code == 200
    assert resp.json()["task"] == "universal-restoration"

    buf.seek(0)
    # Hard-routing stub
    resp = client.post("/hard-routing", files={"file": ("test.png", buf, "image/png")})
    assert resp.status_code == 200
    assert resp.json()["task"] == "hard-routing"

    buf.seek(0)
    # Soft mixture stub
    resp = client.post("/soft-mixture", files={"file": ("test.png", buf, "image/png")})
    assert resp.status_code == 200
    assert resp.json()["task"] == "soft-mixture"

    buf.seek(0)
    # Face to sketch stub
    resp = client.post("/face-to-sketch", files={"file": ("test.png", buf, "image/png")}, data={"style": "1"})
    assert resp.status_code == 200
    assert resp.json()["task"] == "face-to-sketch"
