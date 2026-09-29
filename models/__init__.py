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
    from models.moe import SoftMoERestorationNetwork
except ImportError:
    SoftMoERestorationNetwork = None

try:
    from models.cgan import StyleConditionedUNetGenerator, ConditionalPatchGANDiscriminator
except ImportError:
    StyleConditionedUNetGenerator = None
    ConditionalPatchGANDiscriminator = None

try:
    from models.onnx_export_cgan import export_cgan_generator_to_onnx, verify_cgan_onnx_numerical_equivalence
except ImportError:
    export_cgan_generator_to_onnx = None
    verify_cgan_onnx_numerical_equivalence = None

try:
    from models.onnx_runner import (
        UniversalRestorationONNXRunner,
        CorruptionClassifierONNXRunner,
        HardRoutingONNXRunner,
        SoftMoEONNXRunner,
        StyleConditionedCGANONNXRunner
    )
except ImportError:
    UniversalRestorationONNXRunner = None
    CorruptionClassifierONNXRunner = None
    HardRoutingONNXRunner = None
    SoftMoEONNXRunner = None
    StyleConditionedCGANONNXRunner = None

__all__ = [
    'UniversalAutoencoder',
    'CorruptionClassifier',
    'HardRoutingRestorationPipeline',
    'SoftMoERestorationNetwork',
    'StyleConditionedUNetGenerator',
    'ConditionalPatchGANDiscriminator',
    'export_to_onnx',
    'verify_onnx_numerical_equivalence',
    'export_cgan_generator_to_onnx',
    'verify_cgan_onnx_numerical_equivalence',
    'UniversalRestorationONNXRunner',
    'CorruptionClassifierONNXRunner',
    'HardRoutingONNXRunner',
    'SoftMoEONNXRunner',
    'StyleConditionedCGANONNXRunner'
]

