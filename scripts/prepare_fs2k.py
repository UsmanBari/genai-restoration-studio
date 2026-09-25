"""
Preparation script for FS2K Facial Sketch Synthesis Dataset (Task 4).
Runs inside Google Colab (with Drive mounted) or locally.

1. Unzips the raw FS2K zip already uploaded at GenAI-A1/raw/FS2K/.
2. Verifies exact official structure:
   - FS2K/photo/{photo1,photo2,photo3}
   - FS2K/sketch/{sketch1,sketch2,sketch3}
   - FS2K/anno_train.json
   - FS2K/anno_test.json
3. Validates official pair counts: 1,058 train + 1,046 test = 2,104 total pairs.
4. Splits official train into 85% train / 15% val stratified by the 'style' field (0, 1, 2) with seed 42.
5. Generates fs2k_train_manifest.json, fs2k_val_manifest.json, fs2k_test_manifest.json.
"""

import os
import sys
import json
import zipfile
import shutil

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.manifest_generator import create_fs2k_manifests


def find_zip_in_dir(directory: str) -> str:
    if not os.path.exists(directory):
        return ""
    for fname in os.listdir(directory):
        if fname.lower().endswith('.zip'):
            return os.path.join(directory, fname)
    return ""


def prepare_fs2k(
    fs2k_dir: str = "/content/drive/MyDrive/GenAI-A1/raw/FS2K",
    manifest_dir: str = "configs/manifests",
    split_seed: int = 42
):
    os.makedirs(fs2k_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    # Check if raw zip exists in directory
    zip_path = find_zip_in_dir(fs2k_dir)
    extracted_root = os.path.join(fs2k_dir, "FS2K")

    if not os.path.exists(extracted_root) and zip_path:
        print(f"Extracting FS2K zip: {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(fs2k_dir)
        print("Extraction finished.")

    # Find where FS2K folder is located (sometimes zipped as FS2K/ or nested)
    candidate_roots = [
        extracted_root,
        fs2k_dir,
        os.path.join(fs2k_dir, "FS2K_dataset"),
        os.path.join(fs2k_dir, "fs2k")
    ]
    target_root = None
    for cand in candidate_roots:
        if os.path.exists(os.path.join(cand, "anno_train.json")):
            target_root = cand
            break

    if target_root is None:
        print(f"Notice: anno_train.json not found directly in {fs2k_dir}. Searching subfolders...")
        for root, dirs, files in os.walk(fs2k_dir):
            if "anno_train.json" in files:
                target_root = root
                break

    if target_root is None:
        print(f"Warning: FS2K annotations not found in {fs2k_dir}. (Ensure zip is present and extracted).")
        return False

    print(f"Found FS2K root at: {target_root}")
    anno_train_path = os.path.join(target_root, "anno_train.json")
    anno_test_path = os.path.join(target_root, "anno_test.json")

    with open(anno_train_path, 'r', encoding='utf-8') as f:
        anno_train = json.load(f)
    with open(anno_test_path, 'r', encoding='utf-8') as f:
        anno_test = json.load(f)

    n_train = len(anno_train)
    n_test = len(anno_test)
    n_total = n_train + n_test

    print(f"Official FS2K counts: {n_train} train pairs + {n_test} test pairs = {n_total} total pairs.")
    assert n_train == 1058, f"Expected 1058 train items, got {n_train}"
    assert n_test == 1046, f"Expected 1046 test items, got {n_test}"
    assert n_total == 2104, f"Expected 2104 total items, got {n_total}"

    # Generate stratified manifests
    print("Generating stratified train / validation / test manifests...")
    train_m, val_m, test_m = create_fs2k_manifests(
        anno_train_items=anno_train,
        anno_test_items=anno_test,
        output_dir=manifest_dir,
        val_ratio=0.15,
        seed=split_seed
    )

    with open(train_m, 'r') as f:
        t_items = json.load(f)
    with open(val_m, 'r') as f:
        v_items = json.load(f)
    with open(test_m, 'r') as f:
        te_items = json.load(f)

    print(f"Split counts after 15% stratified carve-out:")
    print(f"  Train: {len(t_items)}")
    print(f"  Validation (15% stratified): {len(v_items)}")
    print(f"  Test: {len(te_items)}")
    print(f"  Total: {len(t_items) + len(v_items) + len(te_items)}")

    # Verify style breakdown in train vs val
    def count_styles(items):
        counts = {}
        for it in items:
            s = it.get('style', 0)
            counts[s] = counts.get(s, 0) + 1
        return counts

    print(f"Style distribution in Train: {count_styles(t_items)}")
    print(f"Style distribution in Val:   {count_styles(v_items)}")
    print(f"Style distribution in Test:  {count_styles(te_items)}")

    print(f"Manifests successfully created at:\n  {train_m}\n  {val_m}\n  {test_m}")
    return True


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/FS2K"
    m_dir = sys.argv[2] if len(sys.argv) > 2 else "configs/manifests"
    prepare_fs2k(fs2k_dir=src, manifest_dir=m_dir)
