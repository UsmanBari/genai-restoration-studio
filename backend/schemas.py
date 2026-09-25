"""
Pydantic schema definitions for FastAPI backend endpoints.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    device: str = "cpu"
    models_loaded: Dict[str, bool] = Field(
        default_factory=lambda: {
            "universal_autoencoder": False,
            "hard_routing_classifier": False,
            "hard_routing_experts": False,
            "soft_moe": False,
            "fs2k_pix2pix": False
        }
    )


class RestorationResponse(BaseModel):
    task: str
    status: str
    output_image_base64: str
    detected_corruption: Optional[str] = None
    confidence: Optional[float] = None
    expert_weights: Optional[Dict[str, float]] = None
    latency_ms: float
    metrics: Optional[Dict[str, float]] = None


class SketchResponse(BaseModel):
    task: str = "face-to-sketch"
    status: str
    sketch_image_base64: str
    style_applied: Optional[int] = None
    latency_ms: float
