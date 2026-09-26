"""
Models package for Generative AI Restoration Studio.
"""

try:
    from models.autoencoders import UniversalAutoencoder
except ImportError:
    UniversalAutoencoder = None

try:
    from models.classifiers import CorruptionClassifier
except ImportError:
    CorruptionClassifier = None

try:
    from models.hard_router import HardRoutingRestorationPipeline
except ImportError:
    HardRoutingRestorationPipeline = None

try:
    from models.onnx_export import export_to_onnx, verify_onnx_numerical_equivalence
except ImportError:
    export_to_onnx = None
    verify_onnx_numerical_equivalence = None

try:
    from models.onnx_runner import UniversalRestorationONNXRunner
except ImportError:
    UniversalRestorationONNXRunner = None

__all__ = [
    'UniversalAutoencoder',
    'CorruptionClassifier',
    'HardRoutingRestorationPipeline',
    'export_to_onnx',
    'verify_onnx_numerical_equivalence',
    'UniversalRestorationONNXRunner'
]
