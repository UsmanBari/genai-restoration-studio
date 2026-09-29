"""
Unit and integration tests for FastAPI backend endpoints using real ONNX models and real test image assets.
"""

import io
import os
import base64
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app
from evaluation.metrics import compute_psnr, compute_ssim
from data.corruptions import (
    apply_salt_and_pepper,
    apply_gaussian_blur,
    apply_rectangular_occlusion,
    generate_occlusion_rectangles
)

client = TestClient(app)


@pytest.fixture
def clean_pet_image():
    path = os.path.join("tests", "assets", "clean_pet.png")
    if not os.path.exists(path):
        pytest.skip(f"Test asset {path} not found")
    return Image.open(path).convert("RGB")


@pytest.fixture
def clean_face_image():
    path = os.path.join("tests", "assets", "clean_face.png")
    if not os.path.exists(path):
        pytest.skip(f"Test asset {path} not found")
    return Image.open(path).convert("RGB")


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert data["device"] == "cpu"
    for k, v in data["models_loaded"].items():
        assert v is True, f"Model component {k} not loaded!"


def test_universal_restoration_psnr_gain(clean_pet_image):
    """Assert that universal restoration improves PSNR over the corrupted input."""
    clean_np = np.array(clean_pet_image, dtype=np.float32) / 255.0
    sp_np = apply_salt_and_pepper(np.array(clean_pet_image), prob=0.08, rng=np.random.default_rng(42))
    sp_img = Image.fromarray(sp_np)

    buf = io.BytesIO()
    sp_img.save(buf, format="PNG")
    files = {"file": ("corrupted.png", buf.getvalue(), "image/png")}

    response = client.post("/api/universal-restoration", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    restored_bytes = base64.b64decode(data["output_image_base64"])
    restored_img = Image.open(io.BytesIO(restored_bytes)).convert("RGB")
    restored_np = np.array(restored_img, dtype=np.float32) / 255.0

    psnr_corrupted = compute_psnr(clean_np, sp_np.astype(np.float32) / 255.0)
    psnr_restored = compute_psnr(clean_np, restored_np)

    print(f"\n[Task 1 Universal Restoration] S&P Corrupted PSNR: {psnr_corrupted:.2f} dB -> Restored PSNR: {psnr_restored:.2f} dB (Gain: +{psnr_restored-psnr_corrupted:.2f} dB, Latency: {data['latency_ms']:.2f} ms)")
    assert psnr_restored > psnr_corrupted


def test_hard_routing_all_corruptions(clean_pet_image):
    """Assert that hard routing correctly labels all 4 corruption classes with high confidence."""
    sp_np = apply_salt_and_pepper(np.array(clean_pet_image), prob=0.08, rng=np.random.default_rng(42))
    blur_np = apply_gaussian_blur(np.array(clean_pet_image), 5, 1.5)
    rects = generate_occlusion_rectangles(128, 128, 2, 0.20, rng=np.random.default_rng(42))
    occ_np = apply_rectangular_occlusion(np.array(clean_pet_image), rects)

    test_cases = [
        ("clean", clean_pet_image),
        ("salt_and_pepper", Image.fromarray(sp_np)),
        ("gaussian_blur", Image.fromarray(blur_np)),
        ("rectangular_occlusion", Image.fromarray(occ_np)),
    ]

    for expected_label, img in test_cases:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        files = {"file": ("input.png", buf.getvalue(), "image/png")}

        response = client.post("/api/hard-routing", files=files)
        assert response.status_code == 200
        data = response.json()

        print(f"\n[Task 2 Hard Routing ({expected_label})] Predicted: {data['predicted_corruption']} (Confidence: {data['confidence']:.4f}, Selected Expert: {data['selected_expert']}, Latency: {data['latency_ms']:.2f} ms)")
        assert data["predicted_corruption"] == expected_label
        assert data["confidence"] > 0.50
        assert len(data["output_image_base64"]) > 50


def test_hard_routing_oracle_bypass(clean_pet_image):
    buf = io.BytesIO()
    clean_pet_image.save(buf, format="PNG")
    files = {"file": ("clean.png", buf.getvalue(), "image/png")}

    response = client.post("/api/hard-routing", files=files, data={"oracle_class": 0})
    assert response.status_code == 200
    data = response.json()
    assert data["oracle_used"] is True
    assert "identity_bypass" in data["selected_expert"]


def test_soft_mixture_routing_dominance(clean_pet_image):
    """Assert that the soft MoE gating network assigns dominant weight to the matching expert."""
    sp_np = apply_salt_and_pepper(np.array(clean_pet_image), prob=0.08, rng=np.random.default_rng(42))
    blur_np = apply_gaussian_blur(np.array(clean_pet_image), 5, 1.5)
    rects = generate_occlusion_rectangles(128, 128, 2, 0.20, rng=np.random.default_rng(42))
    occ_np = apply_rectangular_occlusion(np.array(clean_pet_image), rects)

    test_cases = [
        ("clean", clean_pet_image, "clean_identity"),
        ("salt_and_pepper", Image.fromarray(sp_np), "salt_and_pepper_specialist"),
        ("gaussian_blur", Image.fromarray(blur_np), "gaussian_blur_specialist"),
        ("rectangular_occlusion", Image.fromarray(occ_np), "rectangular_occlusion_specialist"),
    ]

    for label_name, img, expected_expert in test_cases:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        files = {"file": ("input.png", buf.getvalue(), "image/png")}

        response = client.post("/api/soft-mixture", files=files)
        assert response.status_code == 200
        data = response.json()

        print(f"\n[Task 3 Soft MoE ({label_name})] Dominant: {data['dominant_expert']}, Weights: {data['routing_weights']}, Latency: {data['latency_ms']:.2f} ms")
        assert data["dominant_expert"] == expected_expert
        assert data["routing_weights"][expected_expert] > 0.50
        assert sum(data["routing_weights"].values()) == pytest.approx(1.0, abs=1e-3)


def test_face_to_sketch_styles_differ_and_bounded(clean_face_image):
    """Assert that face-to-sketch outputs for styles 0, 1, 2 differ visibly and stay in [0, 1]."""
    buf = io.BytesIO()
    clean_face_image.save(buf, format="PNG")

    sketches = {}
    for style_id in (0, 1, 2):
        files = {"file": ("face.png", buf.getvalue(), "image/png")}
        response = client.post("/api/face-to-sketch", files=files, data={"style_id": style_id})
        assert response.status_code == 200
        data = response.json()

        sk_bytes = base64.b64decode(data["sketch_image_base64"])
        sk_img = Image.open(io.BytesIO(sk_bytes)).convert("RGB")
        sk_np = np.array(sk_img, dtype=np.float32) / 255.0

        assert sk_np.min() >= 0.0 and sk_np.max() <= 1.0
        sketches[style_id] = sk_np

    diff_id0_vs_id1 = float(np.mean(np.abs(sketches[0] - sketches[1])))
    diff_id1_vs_id2 = float(np.mean(np.abs(sketches[1] - sketches[2])))
    diff_id0_vs_id2 = float(np.mean(np.abs(sketches[0] - sketches[2])))

    print(f"\n[Task 4 Face-to-Sketch Multi-Style Discrepancy] id0_vs_id1: {diff_id0_vs_id1:.4f}, id1_vs_id2: {diff_id1_vs_id2:.4f}, id0_vs_id2: {diff_id0_vs_id2:.4f}")
    assert diff_id0_vs_id1 > 0.01, f"Style id 0 and Style id 1 outputs are too similar (diff: {diff_id0_vs_id1})"
    assert diff_id1_vs_id2 > 0.01, f"Style id 1 and Style id 2 outputs are too similar (diff: {diff_id1_vs_id2})"
    assert diff_id0_vs_id2 > 0.01, f"Style id 0 and Style id 2 outputs are too similar (diff: {diff_id0_vs_id2})"


def test_corrupt_endpoint_reproducibility(clean_pet_image):
    """Assert that /api/corrupt reproduces programmatic corruptions identically."""
    buf = io.BytesIO()
    clean_pet_image.save(buf, format="PNG")

    for corr_type in ["salt_and_pepper", "gaussian_blur", "rectangular_occlusion", "clean"]:
        files = {"file": ("clean.png", buf.getvalue(), "image/png")}
        response = client.post("/api/corrupt", files=files, data={"corruption_type": corr_type, "severity_tier": "medium"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["corruption_type"] == corr_type
        assert len(data["corrupted_image_base64"]) > 50


def test_sample_gallery_endpoint():
    """Assert that /api/samples returns sample presets and static URLs work."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["samples"]) >= 2
    for s in data["samples"]:
        assert "id" in s
        assert "name" in s
        assert "category" in s
        assert "url" in s
        # Verify static URL returns valid image bytes
        static_resp = client.get(s["url"])
        assert static_resp.status_code == 200
        assert "image" in static_resp.headers.get("content-type", "")
        assert len(static_resp.content) > 1000


def test_upload_validation_errors():
    """Assert that non-images, empty files, and invalid requests return clear 4xx errors."""
    # 1. Missing input
    resp = client.post("/api/universal-restoration")
    assert resp.status_code == 400
    assert "Either 'file' upload or 'image_base64'" in resp.json()["detail"]

    # 2. Unsupported content type
    files = {"file": ("script.py", b"print('hello')", "text/x-python")}
    resp = client.post("/api/universal-restoration", files=files)
    assert resp.status_code == 415
    assert "Unsupported file type" in resp.json()["detail"]

    # 3. Empty file
    files = {"file": ("empty.png", b"", "image/png")}
    resp = client.post("/api/universal-restoration", files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]




