"""
ONNX Export and Numerical Equivalence Verification for Universal Autoencoder (Task 1).
"""

from typing import Tuple, Dict, Any
import os
import numpy as np

try:
    import torch
    import onnx
    import onnxruntime as ort
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    onnx = None
    ort = None


def export_to_onnx(
    model: torch.nn.Module,
    onnx_output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
    opset_version: int = 16
) -> str:
    """
    Exports a PyTorch model to a self-contained ONNX format with all weights embedded.
    """
    os.makedirs(os.path.dirname(os.path.abspath(onnx_output_path)), exist_ok=True)
    model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    print(f"Exporting PyTorch model to self-contained ONNX: {onnx_output_path} (Opset {opset_version})...")
    
    export_kwargs = {
        'export_params': True,
        'opset_version': opset_version,
        'do_constant_folding': True,
        'input_names': ['input_image'],
        'output_names': ['restored_image'],
        'dynamic_axes': {
            'input_image': {0: 'batch_size'},
            'restored_image': {0: 'batch_size'}
        }
    }

    # Use dynamo=False on PyTorch 2.x to guarantee all weights (2.6M - 4.6M params, ~10-18 MB)
    # are embedded directly into the standalone .onnx protobuf rather than stripped or externalized
    try:
        torch.onnx.export(model, dummy_input, onnx_output_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, onnx_output_path, **export_kwargs)

    # Check ONNX model validity and embedded weight initializers
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    num_initializers = len(onnx_model.graph.initializer)
    print(f"[SUCCESS] ONNX model successfully verified and exported ({file_size_mb:.2f} MB, {num_initializers} weight initializers).")
    return onnx_output_path



def verify_onnx_numerical_equivalence(
    model: torch.nn.Module,
    onnx_path: str,
    sample_batch: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Dict[str, Any]:
    """
    Compares PyTorch vs. ONNX Runtime outputs on real test samples.
    Computes max absolute error, mean squared error, and verifies numerical parity.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        pt_out = model(sample_batch.to(device)).cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: sample_batch.cpu().numpy().astype(np.float32)}
    onnx_out = session.run(None, ort_inputs)[0]

    max_abs_diff = float(np.max(np.abs(pt_out - onnx_out)))
    mean_sq_diff = float(np.mean((pt_out - onnx_out) ** 2))
    is_close = bool(np.allclose(pt_out, onnx_out, rtol=rtol, atol=atol))

    print("\n=== ONNX Numerical Parity Verification ===")
    print(f"  PyTorch Output Shape: {pt_out.shape} | Range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")
    print(f"  ONNX Output Shape:    {onnx_out.shape} | Range: [{onnx_out.min():.4f}, {onnx_out.max():.4f}]")
    print(f"  Max Absolute Diff:    {max_abs_diff:.6e} (Tolerance atol={atol})")
    print(f"  Mean Squared Diff:    {mean_sq_diff:.6e}")
    print(f"  Parity Status:        {'PASS (Equivalence Confirmed)' if is_close else 'FAIL (Deviation Detected)'}")

    return {
        'max_abs_diff': max_abs_diff,
        'mean_sq_diff': mean_sq_diff,
        'is_close': is_close
    }
