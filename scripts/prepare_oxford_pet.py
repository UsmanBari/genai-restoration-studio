"""
Download and preparation script for Oxford-IIIT Pet Dataset.
Designed to be run once inside Google Colab (with Drive mounted) or locally.

1. Downloads images.tar.gz and annotations.tar.gz (if not already downloaded).
2. Extracts clean images.
3. Converts all images to RGB and resizes to 128x128.
4. Generates deterministic 80% train / 20% val and fixed-tier test manifests.
"""

import os
import sys
import tarfile
import urllib.request
from PIL import Image
from tqdm import tqdm

# Ensure project root in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.manifest_generator import create_oxford_pet_manifests

OXFORD_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
OXFORD_ANNO_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded {dest_path}.")


def extract_tar(tar_path: str, extract_to: str):
    print(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")


def prepare_oxford_pet(
    output_dir: str = "/content/drive/MyDrive/GenAI-A1/raw/OxfordPet",
    manifest_dir: str = "configs/manifests",
    target_size: tuple = (128, 128),
    split_seed: int = 42
):
    """
    Full pipeline to download, preprocess, and generate manifests for Oxford-IIIT Pet.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    images_tar = os.path.join(output_dir, "images.tar.gz")
    anno_tar = os.path.join(output_dir, "annotations.tar.gz")

    download_file(OXFORD_IMAGES_URL, images_tar)
    download_file(OXFORD_ANNO_URL, anno_tar)

    raw_extract_dir = os.path.join(output_dir, "raw_extracted")
    if not os.path.exists(os.path.join(raw_extract_dir, "images")):
        extract_tar(images_tar, raw_extract_dir)
    if not os.path.exists(os.path.join(raw_extract_dir, "annotations")):
        extract_tar(anno_tar, raw_extract_dir)

    raw_images_dir = os.path.join(raw_extract_dir, "images")
    raw_anno_dir = os.path.join(raw_extract_dir, "annotations")
    processed_images_dir = os.path.join(output_dir, "images_128x128")
    os.makedirs(processed_images_dir, exist_ok=True)

    # Read official trainval.txt and test.txt annotations
    trainval_txt = os.path.join(raw_anno_dir, "trainval.txt")
    test_txt = os.path.join(raw_anno_dir, "test.txt")

    entries = []

    def parse_anno_file(filepath: str, split_source: str):
        if not os.path.exists(filepath):
            return []
        items = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    img_name, class_id, species, breed_id = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                    items.append({
                        'image_id': img_name,
                        'filename': f"{img_name}.jpg",
                        'class_id': class_id,
                        'species': species,
                        'breed_id': breed_id,
                        'split_source': split_source
                    })
        return items

    trainval_items = parse_anno_file(trainval_txt, 'trainval')
    test_items = parse_anno_file(test_txt, 'test')
    entries = trainval_items + test_items

    # If annotations not parsed, list all jpgs directly from raw or processed directory
    if not entries:
        search_dirs = [raw_images_dir, processed_images_dir]
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                print(f"Scanning images from directory: {s_dir}...")
                valid_files = [
                    f for f in sorted(os.listdir(s_dir))
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith(('pet_trainval_', 'pet_test_'))
                ]
                if valid_files:
                    for fname in valid_files:
                        entries.append({
                            'image_id': os.path.splitext(fname)[0],
                            'filename': fname,
                            'split_source': 'trainval'
                        })
                    break

    if not entries:
        raise FileNotFoundError(
            f"No valid OxfordPet images found in {raw_images_dir} or {processed_images_dir}."
        )

    print(f"Resizing {len(entries)} images to {target_size} RGB...")
    for entry in tqdm(entries):
        src_path = os.path.join(raw_images_dir, entry['filename'])
        dst_path = os.path.join(processed_images_dir, entry['filename'])
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            try:
                with Image.open(src_path) as img:
                    rgb_img = img.convert('RGB').resize(target_size, Image.Resampling.BILINEAR)
                    rgb_img.save(dst_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"Warning: could not process {src_path}: {e}")

    # Generate manifests
    print("Generating deterministic train/val/test manifests...")
    train_m, val_m, test_m = create_oxford_pet_manifests(
        image_entries=entries,
        output_dir=manifest_dir,
        train_ratio=0.8,
        seed=split_seed,
        img_h=target_size[0],
        img_w=target_size[1]
    )

    print(f"Manifests successfully created at:\n  {train_m}\n  {val_m}\n  {test_m}")


if __name__ == '__main__':
    dest = sys.argv[1] if len(sys.argv) > 1 else "data/raw/OxfordPet"
    m_dir = sys.argv[2] if len(sys.argv) > 2 else "configs/manifests"
    prepare_oxford_pet(output_dir=dest, manifest_dir=m_dir)
