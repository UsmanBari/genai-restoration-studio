"""
Download & Verify Helper for ONNX Model Binaries.
Fetches exported ONNX models for Milestone 5 deployment from GitHub Release assets into models_onnx/.
"""

import os
import sys
import argparse
import urllib.request
import hashlib
import time

RELEASE_TAG = "milestone-5-complete"
REPO_OWNER = "UsmanBari"
REPO_NAME = "genai-restoration-studio"
BASE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{RELEASE_TAG}"

EXPECTED_MODELS = {
    "task1_universal.onnx": {
        "size_bytes": 18207156,  # 17.36 MB
        "sha256": "55053304122f46486e7d5e2290b99625f8710bee23a0987efd2e6a7c962fc941",
        "desc": "Task 1 Universal Denoising Autoencoder",
        "url": f"{BASE_URL}/task1_universal.onnx"
    },
    "task2_classifier.onnx": {
        "size_bytes": 4698438,   # 4.48 MB
        "sha256": "739089fa8a7e155c3b2646093f7568302fd4eee1d2493f7ebefaaa0de7ad312f",
        "desc": "Task 2 4-Class Corruption Classifier",
        "url": f"{BASE_URL}/task2_classifier.onnx"
    },
    "task2_specialist_salt_and_pepper.onnx": {
        "size_bytes": 18207156,  # 17.36 MB
        "sha256": "e698d9e031f744ca8f72f79b0ccb3927c8cfe4bcac7879bda1b01a8286ff63b1",
        "desc": "Task 2 Salt-and-Pepper Specialist Autoencoder",
        "url": f"{BASE_URL}/task2_specialist_salt_and_pepper.onnx"
    },
    "task2_specialist_gaussian_blur.onnx": {
        "size_bytes": 18207156,  # 17.36 MB
        "sha256": "71de9626c5127ee20d18a879dfc3ead049070880c0d4e440b00a253ce0177bef",
        "desc": "Task 2 Gaussian Blur Specialist Autoencoder",
        "url": f"{BASE_URL}/task2_specialist_gaussian_blur.onnx"
    },
    "task2_specialist_rectangular_occlusion.onnx": {
        "size_bytes": 18207156,  # 17.36 MB
        "sha256": "bf7b0505970c433400c1c483eebc87d17b342d3341f213975cc6e79be5b0c463",
        "desc": "Task 2 Rectangular Occlusion Specialist Autoencoder",
        "url": f"{BASE_URL}/task2_specialist_rectangular_occlusion.onnx"
    },
    "task3_soft_moe.onnx": {
        "size_bytes": 59330355,  # 56.58 MB
        "sha256": "a30741c4be5d32d7c058b4fce6e2e20a8396bfe0607298d7606a87455729c388",
        "desc": "Task 3 Soft Mixture-of-Experts Gating + Experts",
        "url": f"{BASE_URL}/task3_soft_moe.onnx"
    },
    "cgan_generator.onnx": {
        "size_bytes": 66719645,  # 63.63 MB
        "sha256": "31ad36d9f560c555a0931aa57fea3b247ec5bc0ef90a9e1e7697de54f8d3b6b9",
        "desc": "Task 4 Style-Conditioned Face-to-Sketch cGAN Generator",
        "url": f"{BASE_URL}/cgan_generator.onnx"
    }
}


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    sys.stdout.write(f"\r  Downloading... {progress_size / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB [{percent}%] ({speed} KB/s)")
    sys.stdout.flush()


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def fetch_and_verify_models(dest_dir: str = None, force_download: bool = False) -> bool:
    if dest_dir is None:
        dest_dir = os.path.join(os.path.dirname(__file__), "..", "models_onnx")
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"\nTarget directory: {dest_dir}")
    print(f"Release source:   {BASE_URL}\n" + "="*75)

    all_success = True

    for filename, meta in EXPECTED_MODELS.items():
        filepath = os.path.join(dest_dir, filename)
        expected_size = meta["size_bytes"]
        expected_sha = meta.get("sha256", "").lower()
        desc = meta["desc"]
        url = meta["url"]

        needs_download = force_download or not os.path.exists(filepath) or os.path.getsize(filepath) != expected_size

        if not needs_download:
            actual_size = os.path.getsize(filepath)
            actual_sha = compute_file_sha256(filepath)
            if actual_sha == expected_sha:
                size_mb = actual_size / (1024 * 1024)
                print(f"[OK] {filename:45s} ({size_mb:6.2f} MB) - SHA256 Verified")
                continue
            else:
                needs_download = True

        print(f"[FETCHING] {filename} ({expected_size / (1024 * 1024):.2f} MB) - {desc}")
        try:
            urllib.request.urlretrieve(url, filepath, _reporthook)
            print()  # newline after progress bar
            actual_size = os.path.getsize(filepath)
            actual_sha = compute_file_sha256(filepath)

            if actual_size == expected_size and actual_sha == expected_sha:
                print(f"  --> Verified: {actual_size:,} bytes | SHA256: {actual_sha[:16]}... Match!")
            else:
                print(f"  --> Warning: Verification failed! Size: {actual_size:,} vs {expected_size:,}, SHA: {actual_sha[:8]} vs {expected_sha[:8]}")
                all_success = False
        except Exception as e:
            print(f"\n  --> Download failed from {url}: {e}")
            all_success = False

    print("="*75)
    if all_success:
        total_mb = sum(meta["size_bytes"] for meta in EXPECTED_MODELS.values()) / (1024 * 1024)
        print(f"All 7 ONNX models are verified and ready ({total_mb:.2f} MB total).\n")
    else:
        print("Some models could not be fetched or verified.\n")
    return all_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and verify ONNX model binaries from GitHub Releases.")
    parser.add_argument("--dest-dir", type=str, default=None, help="Destination directory (default: models_onnx/)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    success = fetch_and_verify_models(dest_dir=args.dest_dir, force_download=args.force)
    sys.exit(0 if success else 1)
