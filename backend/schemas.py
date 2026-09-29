"""
Pydantic schema definitions for FastAPI backend endpoints supporting all 4 workspaces.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    device: str = "cpu"
    models_loaded: Dict[str, bool] = Field(
        default_factory=lambda: {
            "task1_universal": False,
            "task2_classifier": False,
            "task2_specialist_sp": False,
            "task2_specialist_blur": False,
            "task2_specialist_occ": False,
            "task2_hard_routing": False,
            "task3_soft_moe": False,
            "task4_cgan_generator": False,
        }
    )


class UniversalRestorationResponse(BaseModel):
    task: str = "universal_restoration"
    status: str = "success"
    output_image_base64: str
    latency_ms: float


class HardRoutingResponse(BaseModel):
    task: str = "hard_routing"
    status: str = "success"
    output_image_base64: str
    predicted_corruption: str
    confidence: float
    probabilities: Dict[str, float]
    selected_expert: str
    oracle_used: bool = False
    latency_ms: float


class SoftMoEResponse(BaseModel):
    task: str = "soft_mixture_of_experts"
    status: str = "success"
    output_image_base64: str
    routing_weights: Dict[str, float]
    dominant_expert: str
    latency_ms: float


class FaceToSketchResponse(BaseModel):
    task: str = "face_to_sketch"
    status: str = "success"
    sketch_image_base64: str
    style_id: int
    style_name: str
    latency_ms: float


class CorruptionResponse(BaseModel):
    status: str = "success"
    corrupted_image_base64: str
    corruption_type: str
    severity_tier: Optional[str] = None
    params: Dict[str, Any]
