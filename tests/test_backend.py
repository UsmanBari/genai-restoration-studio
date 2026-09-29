"""
Unit and integration tests for FastAPI backend endpoints using real ONNX models.
"""

import io
import base64
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


@pytest.fixture
def sample_pil_image():
    np.random.seed(42)
    arr = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_image_b64(sample_image_bytes):
    return base64.b64encode(sample_image_bytes).decode("utf-8")


def test_health_check_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert data["device"] == "cpu"
    for k, v in data["models_loaded"].items():
        assert v is True, f"Model component {k} not loaded!"


def test_universal_restoration_endpoint(sample_image_bytes):
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/universal-restoration", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "universal_restoration"
    assert data["status"] == "success"
    assert len(data["output_image_base64"]) > 50
    assert data["latency_ms"] > 0


def test_universal_restoration_b64(sample_image_b64):
    response = client.post("/api/universal-restoration", data={"image_base64": sample_image_b64})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_hard_routing_endpoint(sample_image_bytes):
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/hard-routing", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "hard_routing"
    assert data["status"] == "success"
    assert data["predicted_corruption"] in ["clean", "salt_and_pepper", "gaussian_blur", "rectangular_occlusion"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 4
    assert len(data["output_image_base64"]) > 50


def test_hard_routing_oracle_mode(sample_image_b64):
    response = client.post("/api/hard-routing", data={"image_base64": sample_image_b64, "oracle_class": 0})
    assert response.status_code == 200
    data = response.json()
    assert data["oracle_used"] is True
    assert "identity_bypass" in data["selected_expert"]


def test_soft_mixture_endpoint(sample_image_bytes):
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/soft-mixture", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "soft_mixture_of_experts"
    assert data["status"] == "success"
    assert len(data["routing_weights"]) == 4
    weights_sum = sum(data["routing_weights"].values())
    assert pytest.approx(weights_sum, abs=1e-3) == 1.0
    assert data["dominant_expert"] in data["routing_weights"]


def test_face_to_sketch_endpoint(sample_image_bytes):
    for style_id in (0, 1, 2):
        files = {"file": ("face.png", sample_image_bytes, "image/png")}
        response = client.post("/api/face-to-sketch", files=files, data={"style_id": style_id})
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "face_to_sketch"
        assert data["status"] == "success"
        assert data["style_id"] == style_id
        assert f"Style {style_id + 1}" in data["style_name"]
        assert len(data["sketch_image_base64"]) > 50


def test_corrupt_endpoint_presets(sample_image_bytes):
    for corr in ["salt_and_pepper", "gaussian_blur", "rectangular_occlusion", "clean"]:
        files = {"file": ("clean.png", sample_image_bytes, "image/png")}
        response = client.post("/api/corrupt", files=files, data={"corruption_type": corr, "severity_tier": "medium"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["corruption_type"] == corr
        assert len(data["corrupted_image_base64"]) > 50
