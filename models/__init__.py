"""
Models package for Generative AI Restoration Studio.
Lightweight ONNX runners are imported by default for production/backend serving without PyTorch.
Training-time PyTorch model classes and export helpers are lazily loaded on demand.
"""

from models.onnx_runner import (
    UniversalRestorationONNXRunner,
    CorruptionClassifierONNXRunner,
    HardRoutingONNXRunner,
    SoftMoEONNXRunner,
    StyleConditionedCGANONNXRunner
)

__all__ = [
    # Production ONNX Runners (Zero PyTorch dependency)
    "UniversalRestorationONNXRunner",
    "CorruptionClassifierONNXRunner",
    "HardRoutingONNXRunner",
    "SoftMoEONNXRunner",
    "StyleConditionedCGANONNXRunner",
    # PyTorch Training & Architecture Classes (Lazily Loaded)
    "UniversalAutoencoder",
    "CorruptionClassifier",
    "HardRoutingRestorationPipeline",
    "SoftMoERestorationNetwork",
    "StyleConditionedUNetGenerator",
    "ConditionalPatchGANDiscriminator",
    "export_to_onnx",
    "verify_onnx_numerical_equivalence",
    "export_cgan_generator_to_onnx",
    "verify_cgan_onnx_numerical_equivalence",
]

_LAZY_EXPORTS = {
    "UniversalAutoencoder": "models.autoencoders",
    "CorruptionClassifier": "models.classifiers",
    "HardRoutingRestorationPipeline": "models.hard_router",
    "SoftMoERestorationNetwork": "models.moe",
    "StyleConditionedUNetGenerator": "models.cgan",
    "ConditionalPatchGANDiscriminator": "models.cgan",
    "export_to_onnx": "models.onnx_export",
    "verify_onnx_numerical_equivalence": "models.onnx_export",
    "export_cgan_generator_to_onnx": "models.onnx_export_cgan",
    "verify_cgan_onnx_numerical_equivalence": "models.onnx_export_cgan",
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
