"""
Generator-Only ONNX Export and Numerical Equivalence Verification for Task 4 (cGAN).
Exports StyleConditionedUNetGenerator with inputs:
  - photo: (B, 3, 128, 128) float32 in [-1.0, 1.0]
  - style_id: (B,) int64 categorical style index {0, 1, 2}
Output:
  - sketch: (B, 3, 128, 128) float32 in [-1.0, 1.0]
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    onnx = None
    HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

from models.cgan import StyleConditionedUNetGenerator


def export_cgan_generator_to_onnx(
    generator: StyleConditionedUNetGenerator,
    onnx_output_path: str,
    opset_version: int = 18
) -> str:
    """
    Exports StyleConditionedUNetGenerator to standalone ONNX format.
    
    Args:
        generator: Trained StyleConditionedUNetGenerator instance.
        onnx_output_path: Target path to save .onnx model.
        opset_version: ONNX opset version (default: 18).
        
    Returns:
        Path to saved ONNX file.
    """
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required to export models to ONNX.")
    if not HAS_ONNX or onnx is None:
        raise RuntimeError("The 'onnx' package is required to verify exported ONNX models.")

    os.makedirs(os.path.dirname(os.path.abspath(onnx_output_path)), exist_ok=True)
    generator.eval()

    device = next(generator.parameters()).device
    dummy_photo = torch.randn(1, 3, 128, 128, device=device)
    dummy_style = torch.tensor([0], dtype=torch.long, device=device)

    input_names = ['photo', 'style_id']
    output_names = ['sketch']

    dynamic_axes = {
        'photo': {0: 'batch_size'},
        'style_id': {0: 'batch_size'},
        'sketch': {0: 'batch_size'}
    }

    export_kwargs = {
        'export_params': True,
        'opset_version': opset_version,
        'do_constant_folding': True,
        'input_names': input_names,
        'output_names': output_names,
        'dynamic_axes': dynamic_axes
    }

    # Use dynamo=False on PyTorch 2.x to ensure weights are fully embedded in protobuf
    try:
        torch.onnx.export(
            generator,
            (dummy_photo, dummy_style),
            onnx_output_path,
            dynamo=False,
            **export_kwargs
        )
    except TypeError:
        torch.onnx.export(
            generator,
            (dummy_photo, dummy_style),
            onnx_output_path,
            **export_kwargs
        )

    # Verify model with onnx checker
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    num_initializers = len(onnx_model.graph.initializer)
    print(f"[SUCCESS] cGAN Generator ONNX model verified and exported ({file_size_mb:.2f} MB, {num_initializers} weight initializers).")
    return onnx_output_path


def verify_cgan_onnx_numerical_equivalence(
    generator: StyleConditionedUNetGenerator,
    onnx_path: str,
    sample_photos: Optional[torch.Tensor] = None,
    sample_styles: Optional[torch.Tensor] = None,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Dict[str, Any]:
    """
    Verifies numerical equivalence between PyTorch generator and ONNX Runtime session.
    """
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for numerical equivalence checks.")
    if not HAS_ORT or ort is None:
        raise RuntimeError("onnxruntime is required for numerical equivalence checks.")

    generator.eval()
    device = next(generator.parameters()).device

    if sample_photos is None:
        sample_photos = torch.rand(3, 3, 128, 128, device=device) * 2.0 - 1.0
    if sample_styles is None:
        sample_styles = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

    with torch.no_grad():
        pt_out = generator(sample_photos, sample_styles).cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {
        'photo': sample_photos.cpu().numpy().astype(np.float32),
        'style_id': sample_styles.cpu().numpy().astype(np.int64)
    }
    onnx_outs = session.run(None, ort_inputs)
    onnx_out = onnx_outs[0]

    max_abs_diff = float(np.max(np.abs(pt_out - onnx_out)))
    mean_sq_diff = float(np.mean((pt_out - onnx_out) ** 2))
    is_close = bool(np.allclose(pt_out, onnx_out, rtol=rtol, atol=atol))

    print("\n=== Task 4 cGAN ONNX Numerical Parity Verification ===")
    print(f"  PyTorch Output Shape: {pt_out.shape} | Range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")
    print(f"  ONNX Output Shape:    {onnx_out.shape} | Range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
    print(f"  Max Absolute Diff:    {max_abs_diff:.6e} (Tolerance atol={atol})")
    print(f"  Mean Squared Diff:    {mean_sq_diff:.6e}")
    print(f"  Parity Status:        {'PASS (Equivalence Confirmed)' if is_close else 'FAIL (Deviation Detected)'}")

    return {
        'max_abs_diff': max_abs_diff,
        'mean_sq_diff': mean_sq_diff,
        'is_close': is_close,
        'onnx_size_mb': os.path.getsize(onnx_path) / (1024 * 1024),
        'equivalent': is_close
    }
