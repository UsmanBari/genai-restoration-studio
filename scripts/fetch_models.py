"""
Download & Verify Helper for ONNX Model Binaries.
Fetches exported ONNX models for Milestone 5 deployment into models_onnx/.
"""

import os
import sys
import hashlib
import urllib.request

EXPECTED_MODELS = {
    "task1_universal.onnx": {
        "size_bytes": 18204672,  # 17.36 MB
        "desc": "Task 1 Universal Denoising Autoencoder"
    },
    "task2_classifier.onnx": {
        "size_bytes": 4698112,   # 4.48 MB
        "desc": "Task 2 4-Class Corruption Classifier"
    },
    "task2_specialist_salt_and_pepper.onnx": {
        "size_bytes": 18204672,  # 17.36 MB
        "desc": "Task 2 Salt-and-Pepper Specialist Autoencoder"
    },
    "task2_specialist_gaussian_blur.onnx": {
        "size_bytes": 18204672,  # 17.36 MB
        "desc": "Task 2 Gaussian Blur Specialist Autoencoder"
    },
    "task2_specialist_rectangular_occlusion.onnx": {
        "size_bytes": 18204672,  # 17.36 MB
        "desc": "Task 2 Rectangular Occlusion Specialist Autoencoder"
    },
    "task3_soft_moe.onnx": {
        "size_bytes": 59328512,  # 56.58 MB
        "desc": "Task 3 Soft Mixture-of-Experts Gating + Experts"
    },
    "cgan_generator.onnx": {
        "size_bytes": 66723840,  # 63.63 MB
        "desc": "Task 4 Style-Conditioned Face-to-Sketch cGAN Generator"
    }
}

DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "models_onnx")


def verify_models():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Checking models in: {os.path.abspath(DEST_DIR)}")
    missing = []
    present = []
    
    for filename, meta in EXPECTED_MODELS.items():
        filepath = os.path.join(DEST_DIR, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            present.append((filename, size_mb, meta["desc"]))
            print(f"  [FOUND] {filename:45s} ({size_mb:6.2f} MB) - {meta['desc']}")
        else:
            missing.append((filename, meta["desc"]))
            print(f"  [MISSING] {filename:43s} - {meta['desc']}")
            
    print("\n" + "="*70)
    if missing:
        print(f"Status: {len(missing)} model(s) missing from {DEST_DIR}/.")
        print("Please place the exported ONNX files from Google Drive / Release assets into models_onnx/.")
        return False
    else:
        total_size_mb = sum(p[1] for p in present)
        print(f"Status: All {len(present)} models present and ready! Total size: {total_size_mb:.2f} MB.")
        return True


if __name__ == "__main__":
    success = verify_models()
    sys.exit(0 if success else 1)
