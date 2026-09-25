"""
FS2K Dataset Module for Paired Photo-to-Sketch Synthesis (Task 4).
Handles loading paired photos and sketches with style awareness (styles 0, 1, 2).
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

from data.oxford_pet import resolve_manifest_path


def to_tensor(img_np: np.ndarray, normalize_minus1_1: bool = True):
    """
    Convert HWC [0, 255] uint8 array to CHW float32 tensor.
    If normalize_minus1_1 is True (standard for GANs / pix2pix), range is [-1.0, 1.0].
    Otherwise range is [0.0, 1.0].
    """
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    elif img_np.max() > 1.0:
        img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = img_np.astype(np.float32)

    if normalize_minus1_1:
        img_np = (img_np - 0.5) / 0.5  # maps [0, 1] to [-1, 1]

    if len(img_np.shape) == 2:
        img_np = np.expand_dims(img_np, axis=2)

    tensor_np = np.transpose(img_np, (2, 0, 1))
    if HAS_TORCH:
        return torch.from_numpy(tensor_np)
    return tensor_np


class FS2KDataset(Dataset):
    """
    FS2K Paired Facial Sketch Dataset.
    
    Args:
        manifest_path: Path to JSON manifest file or manifests directory
        fs2k_root: Root directory of extracted FS2K dataset containing 'photo' and 'sketch'
        target_size: tuple (256, 256)
        normalize_gan: normalize images to [-1, 1] for pix2pix GAN
        transform: optional additional transform
    """

    def __init__(
        self,
        manifest_path: str,
        fs2k_root: str,
        split: str = 'train',
        target_size: Tuple[int, int] = (256, 256),
        normalize_gan: bool = True,
        transform: Optional[Callable] = None
    ):
        self.fs2k_root = fs2k_root
        self.split = split
        self.target_size = target_size
        self.normalize_gan = normalize_gan
        self.transform = transform

        default_filename = f"fs2k_{split}_manifest.json"
        resolved_path = resolve_manifest_path(manifest_path, default_filename)
        self.manifest_path = resolved_path

        with open(resolved_path, 'r', encoding='utf-8') as f:
            self.items = json.load(f)

        if len(self.items) == 0:
            raise ValueError(f"FS2K Manifest '{resolved_path}' contains 0 items.")

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_paths(self, item: Dict[str, Any]) -> Tuple[str, str]:
        image_name = item.get('image_name', '')
        
        # Check roots (prioritizing local NVMe cache if available)
        roots = [
            self.fs2k_root,
            '/content/local_data/FS2K',
            '/content/local_data/FS2K/FS2K',
            '/content/drive/MyDrive/GenAI-A1/raw/FS2K/FS2K',
            'data/raw/FS2K'
        ]
        
        photo_path = ''
        sketch_path = ''
        
        for r in roots:
            p_cand = os.path.join(r, 'photo', image_name)
            s_subpath = image_name.replace('photo', 'sketch').replace('image', 'sketch')
            s_cand = os.path.join(r, 'sketch', s_subpath)
            
            p_final = (p_cand + '.jpg') if os.path.exists(p_cand + '.jpg') else ((p_cand + '.png') if os.path.exists(p_cand + '.png') else '')
            s_final = (s_cand + '.jpg') if os.path.exists(s_cand + '.jpg') else ((s_cand + '.png') if os.path.exists(s_cand + '.png') else '')
            
            if p_final and s_final:
                return p_final, s_final
            if p_final:
                photo_path = p_final
            if s_final:
                sketch_path = s_final

        # Fallback default constructed paths
        if not photo_path:
            photo_path = os.path.join(self.fs2k_root, 'photo', image_name + '.jpg')
        if not sketch_path:
            sketch_subpath = image_name.replace('photo', 'sketch').replace('image', 'sketch')
            sketch_path = os.path.join(self.fs2k_root, 'sketch', sketch_subpath + '.jpg')

        return photo_path, sketch_path

    def _load_image(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            img = Image.new('RGB', self.target_size, color=(128, 128, 128))
        else:
            img = Image.open(path).convert('RGB')
            if img.size != self.target_size:
                img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        photo_path, sketch_path = self._resolve_paths(item)

        photo_np = self._load_image(photo_path)
        sketch_np = self._load_image(sketch_path)

        photo_t = to_tensor(photo_np, normalize_minus1_1=self.normalize_gan)
        sketch_t = to_tensor(sketch_np, normalize_minus1_1=self.normalize_gan)

        style = item.get('style', 0)

        return {
            'photo': photo_t,
            'sketch': sketch_t,
            'style': style,
            'metadata': item
        }


def get_fs2k_dataloaders(
    manifest_dir: str,
    fs2k_root: str,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True
):
    """Constructs train, val, and test DataLoaders for FS2K."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to build DataLoaders.")

    train_ds = FS2KDataset(
        manifest_path=manifest_dir,
        fs2k_root=fs2k_root,
        split='train',
        normalize_gan=True
    )
    val_ds = FS2KDataset(
        manifest_path=manifest_dir,
        fs2k_root=fs2k_root,
        split='val',
        normalize_gan=True
    )
    test_ds = FS2KDataset(
        manifest_path=manifest_dir,
        fs2k_root=fs2k_root,
        split='test',
        normalize_gan=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
