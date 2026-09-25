"""
FastAPI Backend for Generative AI Restoration Studio.
Serves ONNX runtime inference endpoints for image restoration and sketch synthesis.
"""

import time
import base64
import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from backend.schemas import HealthResponse, RestorationResponse, SketchResponse

app = FastAPI(
    title="Generative AI Restoration & Synthesis Studio API",
    description="REST API serving Universal Restoration, Hard-Routing, Soft MoE, and FS2K Face-to-Sketch models.",
    version="1.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint providing system status and model readiness."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        device="cpu",
        models_loaded={
            "universal_autoencoder": False,
            "hard_routing_classifier": False,
            "hard_routing_experts": False,
            "soft_moe": False,
            "fs2k_pix2pix": False
        }
    )


@app.post("/universal-restoration", response_model=RestorationResponse)
async def universal_restoration(
    file: UploadFile = File(...)
):
    """
    Task 1: Universal Autoencoder Restoration stub.
    Restores image using single end-to-end autoencoder.
    """
    t0 = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((128, 128))
        # Stub: returns received image resized as base64 until trained ONNX is plugged in
        b64_output = image_to_base64(image)
        latency = (time.time() - t0) * 1000.0
        return RestorationResponse(
            task="universal-restoration",
            status="stub_ready",
            output_image_base64=b64_output,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/hard-routing", response_model=RestorationResponse)
async def hard_routing_restoration(
    file: UploadFile = File(...)
):
    """
    Task 2: Hard-Routed Restoration stub.
    Classifies corruption and routes to specialized autoencoder.
    """
    t0 = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((128, 128))
        b64_output = image_to_base64(image)
        latency = (time.time() - t0) * 1000.0
        return RestorationResponse(
            task="hard-routing",
            status="stub_ready",
            output_image_base64=b64_output,
            detected_corruption="gaussian_blur",
            confidence=0.98,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/soft-mixture", response_model=RestorationResponse)
async def soft_mixture_restoration(
    file: UploadFile = File(...)
):
    """
    Task 3: Soft Mixture-of-Experts Restoration stub.
    Computes soft gating weights and blends expert outputs.
    """
    t0 = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((128, 128))
        b64_output = image_to_base64(image)
        latency = (time.time() - t0) * 1000.0
        return RestorationResponse(
            task="soft-mixture",
            status="stub_ready",
            output_image_base64=b64_output,
            expert_weights={
                "clean_expert": 0.05,
                "sp_expert": 0.10,
                "blur_expert": 0.75,
                "occlusion_expert": 0.10
            },
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/face-to-sketch", response_model=SketchResponse)
async def face_to_sketch(
    file: UploadFile = File(...),
    style: int = Form(0)
):
    """
    Task 4: FS2K Face-to-Sketch Synthesis stub.
    Generates paired sketch from facial photo.
    """
    t0 = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((256, 256))
        b64_output = image_to_base64(image)
        latency = (time.time() - t0) * 1000.0
        return SketchResponse(
            task="face-to-sketch",
            status="stub_ready",
            sketch_image_base64=b64_output,
            style_applied=style,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
