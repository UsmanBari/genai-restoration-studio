"""
Oxford-IIIT Pet Dataset Module.
Handles loading 128x128 RGB images with on-the-fly programmatic corruptions for training
and deterministic manifests for validation and test.
"""

import os
import json
from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass
    class DataLoader:
        pass

from data.corruptions import (
    apply_random_corruption_runtime,
    apply_deterministic_corruption,
    CORRUPTION_NAMES
)


def resolve_manifest_path(manifest_input: str, default_filename: str) -> str:
    """
    Robustly resolves the full path to a JSON manifest across local repo,
    custom paths, and Google Drive mounts.
    """
    # 1. Exact file match
    if os.path.isfile(manifest_input):
        return manifest_input

    # 2. Directory passed -> append default filename
    if os.path.isdir(manifest_input):
        cand = os.path.join(manifest_input, default_filename)
        if os.path.isfile(cand):
            return cand

    # 3. Check relative repo configs/manifests
    repo_cand = os.path.join('configs', 'manifests', default_filename)
    if os.path.isfile(repo_cand):
        return repo_cand

    # 4. Check relative to this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    module_cand = os.path.join(project_root, 'configs', 'manifests', default_filename)
    if os.path.isfile(module_cand):
        return module_cand

    # 5. Check Colab Google Drive standard paths
    drive_cand = os.path.join('/content/drive/MyDrive/GenAI-A1/manifests', default_filename)
    if os.path.isfile(drive_cand):
        return drive_cand

    colab_repo_cand = os.path.join('/content/genai-restoration-studio/configs/manifests', default_filename)
    if os.path.isfile(colab_repo_cand):
        return colab_repo_cand

    raise FileNotFoundError(
        f"Could not locate manifest '{default_filename}'. Searched candidate paths:\n"
        f"  1. {manifest_input}\n"
        f"  2. {os.path.join(manifest_input, default_filename) if os.path.exists(manifest_input) else 'N/A'}\n"
        f"  3. {repo_cand}\n"
        f"  4. {module_cand}\n"
        f"  5. {drive_cand}\n"
        f"  6. {colab_repo_cand}"
    )


def to_tensor(img_np: np.ndarray):
    """Convert HWC [0, 255] uint8 or float ndarray to CHW float32 tensor in [0.0, 1.0]."""
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    elif img_np.max() > 1.0:
        img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = img_np.astype(np.float32)

    if len(img_np.shape) == 2:
        img_np = np.expand_dims(img_np, axis=2)

    # HWC to CHW
    tensor_np = np.transpose(img_np, (2, 0, 1))
    if HAS_TORCH:
        return torch.from_numpy(tensor_np)
    return tensor_np


class OxfordPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset with runtime & deterministic corruption support.
    
    Args:
        manifest_path: Path to JSON manifest file or manifests directory
        images_dir: Directory where 128x128 RGB images are stored
        split: 'train', 'val', or 'test'
        target_size: tuple (128, 128)
        transform: optional custom transformation function
    """

    def __init__(
        self,
        manifest_path: str,
        images_dir: str,
        split: str = 'train',
        target_size: Tuple[int, int] = (128, 128),
        transform: Optional[Callable] = None
    ):
        self.images_dir = images_dir
        self.split = split
        self.target_size = target_size
        self.transform = transform

        default_filename = f"oxford_{split}_manifest.json"
        resolved_path = resolve_manifest_path(manifest_path, default_filename)
        self.manifest_path = resolved_path

        with open(resolved_path, 'r', encoding='utf-8') as f:
            self.items = json.load(f)

        if len(self.items) == 0:
            raise ValueError(f"Manifest '{resolved_path}' contains 0 items.")

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, item: Dict[str, Any]) -> np.ndarray:
        filename = item.get('filename') or item.get('image_name') or os.path.basename(item.get('path', ''))
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = filename + '.jpg'

        path = os.path.join(self.images_dir, filename)
        if not os.path.exists(path) and 'path' in item:
            path = item['path']

        # Check Colab Drive standard fallback path
        if not os.path.exists(path):
            colab_drive_path = os.path.join('/content/drive/MyDrive/GenAI-A1/raw/OxfordPet/images_128x128', filename)
            if os.path.exists(colab_drive_path):
                path = colab_drive_path

        if not os.path.exists(path):
            # Fallback placeholder if image file not on current machine (e.g. dry run test)
            img = Image.new('RGB', self.target_size, color=(128, 128, 128))
        else:
            img = Image.open(path).convert('RGB')
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.Resampling.BILINEAR)

        return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        clean_np = self._load_image(item)

        if self.split == 'train':
            # Training runtime corruption: 25% clean, 25% s&p, 25% blur, 25% occlusion
            corrupted_np, label, params = apply_random_corruption_runtime(clean_np)
        else:
            # Deterministic corruption from manifest
            corrupted_np = apply_deterministic_corruption(clean_np, item)
            label = item.get('label', 0)
            params = item

        corrupted_t = to_tensor(corrupted_np)
        clean_t = to_tensor(clean_np)

        if self.transform is not None:
            corrupted_t = self.transform(corrupted_t)
            clean_t = self.transform(clean_t)

        return {
            'corrupted': corrupted_t,
            'clean': clean_t,
            'label': label,
            'metadata': item
        }


def get_oxford_dataloaders(
    manifest_dir: str,
    images_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True
):
    """Constructs train, val, and test DataLoaders for Oxford-IIIT Pet."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to build DataLoaders.")

    train_ds = OxfordPetDataset(
        manifest_path=manifest_dir,
        images_dir=images_dir,
        split='train'
    )
    val_ds = OxfordPetDataset(
        manifest_path=manifest_dir,
        images_dir=images_dir,
        split='val'
    )
    test_ds = OxfordPetDataset(
        manifest_path=manifest_dir,
        images_dir=images_dir,
        split='test'
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
