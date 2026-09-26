"""
ONNX Export and Numerical Equivalence Verification for Universal Autoencoder, Classifier & Specialists.
"""

from typing import Tuple, Dict, Any, Optional, List
import os
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


def export_to_onnx(
    model: Any,
    onnx_output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
    opset_version: int = 18,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None
) -> str:
    """
    Exports a PyTorch model to a self-contained ONNX format with all weights embedded.
    """
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required to export models to ONNX.")
    if not HAS_ONNX or onnx is None:
        raise RuntimeError("The 'onnx' package is required to verify exported ONNX models.")

    os.makedirs(os.path.dirname(os.path.abspath(onnx_output_path)), exist_ok=True)
    model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    in_names = input_names or ['input_image']
    out_names = output_names or ['restored_image']

    dynamic_axes = {in_names[0]: {0: 'batch_size'}}
    for out_name in out_names:
        dynamic_axes[out_name] = {0: 'batch_size'}

    export_kwargs = {
        'export_params': True,
        'opset_version': opset_version,
        'do_constant_folding': True,
        'input_names': in_names,
        'output_names': out_names,
        'dynamic_axes': dynamic_axes
    }

    # Use dynamo=False on PyTorch 2.x to guarantee all weights are embedded directly
    # into the standalone .onnx protobuf rather than stripped or externalized
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
    model: Any,
    onnx_path: str,
    sample_batch: Optional[Any] = None,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Dict[str, Any]:
    """
    Compares PyTorch vs. ONNX Runtime outputs on test samples.
    Supports single-tensor outputs or tuple/multi-tensor outputs (e.g. Soft MoE).
    """
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for numerical equivalence checks.")
    if not HAS_ORT or ort is None:
        raise RuntimeError("onnxruntime is required for numerical equivalence checks.")

    model.eval()
    device = next(model.parameters()).device

    if sample_batch is None:
        sample_batch = torch.rand(2, 3, 128, 128)

    with torch.no_grad():
        pt_res = model(sample_batch.to(device))
        if isinstance(pt_res, tuple):
            pt_outs = [t.cpu().numpy() for t in pt_res if isinstance(t, torch.Tensor)]
        else:
            pt_outs = [pt_res.cpu().numpy()]

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: sample_batch.cpu().numpy().astype(np.float32)}
    onnx_outs = session.run(None, ort_inputs)

    max_abs_diff = float(max(np.max(np.abs(p - o)) for p, o in zip(pt_outs, onnx_outs)))
    mean_sq_diff = float(np.mean([np.mean((p - o) ** 2) for p, o in zip(pt_outs, onnx_outs)]))
    is_close = bool(all(np.allclose(p, o, rtol=rtol, atol=atol) for p, o in zip(pt_outs, onnx_outs)))

    print("\n=== ONNX Numerical Parity Verification ===")
    print(f"  Outputs Verified:     {len(pt_outs)} output tensors")
    print(f"  PyTorch Primary Out:  {pt_outs[0].shape} | Range: [{pt_outs[0].min():.4f}, {pt_outs[0].max():.4f}]")
    print(f"  ONNX Primary Out:     {onnx_outs[0].shape} | Range: [{onnx_outs[0].min():.4f}, {onnx_outs[0].max():.4f}]")
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
