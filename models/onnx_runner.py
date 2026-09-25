"""
Unified ONNX Runtime Inference Runner for serving models locally.
Provides clean APIs for Universal Restoration, Hard-Routing, MoE, and Face-to-Sketch.
"""

from typing import Optional, Union, Tuple
import os
import time
import numpy as np
from PIL import Image
import onnxruntime as ort


class UniversalRestorationONNXRunner:
    """
    Inference Runner for Universal Denoising Autoencoder using ONNX Runtime.
    """

    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_session()

    def _load_session(self):
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_model_path}")

        # Choose best available provider (CPU for local machine)
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Preprocesses PIL image to (1, 3, H, W) float32 array in [0.0, 1.0]."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.BILINEAR)

        arr = np.array(image, dtype=np.float32) / 255.0
        # HWC to CHW
        arr = np.transpose(arr, (2, 0, 1))
        # Add batch dim -> (1, 3, 128, 128)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def postprocess_array(self, tensor_np: np.ndarray) -> Image.Image:
        """Converts (1, 3, H, W) float32 output array back to PIL Image."""
        arr = tensor_np[0]  # (3, H, W)
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def restore(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """
        Runs full end-to-end restoration on input PIL image.
        Returns (restored_pil_image, latency_ms).
        """
        t0 = time.time()
        input_tensor = self.preprocess_image(image)
        output_tensor = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        restored_img = self.postprocess_array(output_tensor)
        latency_ms = (time.time() - t0) * 1000.0
        return restored_img, latency_ms
