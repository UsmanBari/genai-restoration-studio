"""
Data package initializing OxfordPetDataset, FS2KDataset, and corruption routines.
Corruption routines are imported by default with zero PyTorch dependency.
PyTorch Dataset classes and DataLoaders are lazily loaded on demand.
"""

from data.corruptions import (
    apply_salt_and_pepper,
    apply_gaussian_blur,
    apply_rectangular_occlusion,
    generate_occlusion_rectangles,
    apply_random_corruption_runtime,
    apply_targeted_corruption_runtime,
    apply_deterministic_corruption,
    CORRUPTION_NAMES,
    NAME_TO_CLASS
)

__all__ = [
    # Corruption routines (No PyTorch dependency)
    "apply_salt_and_pepper",
    "apply_gaussian_blur",
    "apply_rectangular_occlusion",
    "generate_occlusion_rectangles",
    "apply_random_corruption_runtime",
    "apply_targeted_corruption_runtime",
    "apply_deterministic_corruption",
    "CORRUPTION_NAMES",
    "NAME_TO_CLASS",
    # PyTorch Datasets & Generators (Lazily Loaded)
    "OxfordPetDataset",
    "get_oxford_dataloaders",
    "FS2KDataset",
    "get_fs2k_dataloaders",
    "create_oxford_pet_manifests",
    "create_fs2k_manifests",
]

_LAZY_EXPORTS = {
    "OxfordPetDataset": "data.oxford_pet",
    "get_oxford_dataloaders": "data.oxford_pet",
    "FS2KDataset": "data.fs2k",
    "get_fs2k_dataloaders": "data.fs2k",
    "create_oxford_pet_manifests": "data.manifest_generator",
    "create_fs2k_manifests": "data.manifest_generator",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name = _LAZY_EXPORTS[name]
        import importlib
        try:
            module = importlib.import_module(module_name)
            return getattr(module, name)
        except Exception as e:
            raise ImportError(f"Cannot import '{name}' from '{module_name}' (PyTorch may not be installed): {e}") from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
