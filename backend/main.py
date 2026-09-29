"""
FastAPI Backend for Generative AI Restoration Studio.
Serves real ONNX runtime inference endpoints for:
1. Universal Restoration (Task 1)
2. Hard-Routed Restoration (Task 2)
3. Soft Mixture-of-Experts Restoration (Task 3)
4. Face-to-Sketch Generator (Task 4)
"""

import os
import time
import base64
import io
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from backend.schemas import (
    HealthResponse,
    UniversalRestorationResponse,
    HardRoutingResponse,
    SoftMoEResponse,
    FaceToSketchResponse,
    CorruptionResponse
)
from models.onnx_runner import (
    UniversalRestorationONNXRunner,
    CorruptionClassifierONNXRunner,
    HardRoutingONNXRunner,
    SoftMoEONNXRunner,
    StyleConditionedCGANONNXRunner
)
from data.corruptions import (
    apply_salt_and_pepper,
    apply_gaussian_blur,
    apply_rectangular_occlusion,
    generate_occlusion_rectangles,
    apply_targeted_corruption_runtime,
    CORRUPTION_NAMES
)

app = FastAPI(
    title="Generative AI Restoration & Synthesis Studio API",
    description="REST API serving Universal Restoration, Hard-Routing, Soft MoE, and FS2K Face-to-Sketch models.",
    version="1.0.0"
)

# Enable CORS for frontend development and local demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model directories to search
MODEL_DIRS = [
    "models_onnx",
    "models",
    os.path.join(os.path.dirname(__file__), "..", "models_onnx"),
    os.path.join(os.path.dirname(__file__), "..", "models"),
    "/app/models_onnx",
    "/app/models"
]


def resolve_model_path(filename: str) -> Optional[str]:
    for d in MODEL_DIRS:
        p = os.path.abspath(os.path.join(d, filename))
        if os.path.exists(p) and os.path.isfile(p):
            return p
    return None


# Global runner instances
runners: Dict[str, Any] = {}


def get_task1_runner() -> Optional[UniversalRestorationONNXRunner]:
    if "task1" not in runners:
        p = resolve_model_path("task1_universal.onnx")
        if p:
            runners["task1"] = UniversalRestorationONNXRunner(p)
    return runners.get("task1")


def get_task2_hard_router() -> Optional[HardRoutingONNXRunner]:
    if "task2" not in runners:
        clf_p = resolve_model_path("task2_classifier.onnx")
        sp_p = resolve_model_path("task2_specialist_salt_and_pepper.onnx")
        blur_p = resolve_model_path("task2_specialist_gaussian_blur.onnx")
        occ_p = resolve_model_path("task2_specialist_rectangular_occlusion.onnx")
        if clf_p and sp_p and blur_p and occ_p:
            runners["task2"] = HardRoutingONNXRunner(clf_p, sp_p, blur_p, occ_p)
    return runners.get("task2")


def get_task3_moe_runner() -> Optional[SoftMoEONNXRunner]:
    if "task3" not in runners:
        p = resolve_model_path("task3_soft_moe.onnx")
        if p:
            runners["task3"] = SoftMoEONNXRunner(p)
    return runners.get("task3")


def get_task4_cgan_runner() -> Optional[StyleConditionedCGANONNXRunner]:
    if "task4" not in runners:
        p = resolve_model_path("cgan_generator.onnx")
        if p:
            runners["task4"] = StyleConditionedCGANONNXRunner(p)
    return runners.get("task4")


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")


async def extract_image_from_request(file: Optional[UploadFile] = None, image_base64: Optional[str] = None) -> Image.Image:
    if file is not None:
        contents = await file.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    elif image_base64:
        return base64_to_image(image_base64)
    else:
        raise HTTPException(status_code=400, detail="Either 'file' upload or 'image_base64' string must be provided.")


@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint providing system status and model readiness for all 7 ONNX models."""
    t1 = get_task1_runner() is not None
    clf = resolve_model_path("task2_classifier.onnx") is not None
    sp = resolve_model_path("task2_specialist_salt_and_pepper.onnx") is not None
    blur = resolve_model_path("task2_specialist_gaussian_blur.onnx") is not None
    occ = resolve_model_path("task2_specialist_rectangular_occlusion.onnx") is not None
    t2 = get_task2_hard_router() is not None
    t3 = get_task3_moe_runner() is not None
    t4 = get_task4_cgan_runner() is not None

    return HealthResponse(
        status="ok",
        version="1.0.0",
        device="cpu",
        models_loaded={
            "task1_universal": t1,
            "task2_classifier": clf,
            "task2_specialist_sp": sp,
            "task2_specialist_blur": blur,
            "task2_specialist_occ": occ,
            "task2_hard_routing": t2,
            "task3_soft_moe": t3,
            "task4_cgan_generator": t4,
        }
    )


@app.post("/universal-restoration", response_model=UniversalRestorationResponse)
@app.post("/api/universal-restoration", response_model=UniversalRestorationResponse)
async def universal_restoration(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Task 1: Universal Multi-Corruption Denoising Autoencoder Workspace Endpoint.
    Restores image using task1_universal.onnx.
    """
    image = await extract_image_from_request(file, image_base64)
    runner = get_task1_runner()
    if runner is None:
        raise HTTPException(status_code=503, detail="Task 1 Universal Autoencoder ONNX model not loaded.")

    restored_img, latency_ms = runner.restore(image)
    return UniversalRestorationResponse(
        task="universal_restoration",
        status="success",
        output_image_base64=image_to_base64(restored_img),
        latency_ms=round(latency_ms, 2)
    )


@app.post("/hard-routing", response_model=HardRoutingResponse)
@app.post("/api/hard-routing", response_model=HardRoutingResponse)
async def hard_routing(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    oracle_class: Optional[int] = Form(None)
):
    """
    Task 2: Hard-Routed Specialist Autoencoders Workspace Endpoint.
    Classifies corruption via task2_classifier.onnx and routes to specialist or identity bypass.
    """
    image = await extract_image_from_request(file, image_base64)
    router = get_task2_hard_router()
    if router is None:
        raise HTTPException(status_code=503, detail="Task 2 Hard Routing ONNX models not loaded.")

    restored_img, pred_name, probs, selected_expert, latency_ms = router.restore(image, oracle_class=oracle_class)
    conf = float(probs.get(pred_name, 0.0))

    return HardRoutingResponse(
        task="hard_routing",
        status="success",
        output_image_base64=image_to_base64(restored_img),
        predicted_corruption=pred_name,
        confidence=round(conf, 4),
        probabilities={k: round(v, 4) for k, v in probs.items()},
        selected_expert=selected_expert,
        oracle_used=oracle_class is not None,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/soft-mixture", response_model=SoftMoEResponse)
@app.post("/api/soft-mixture", response_model=SoftMoEResponse)
async def soft_mixture(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Task 3: Soft Mixture-of-Experts Restoration Workspace Endpoint.
    Computes soft routing weights across 4 experts via task3_soft_moe.onnx.
    """
    image = await extract_image_from_request(file, image_base64)
    runner = get_task3_moe_runner()
    if runner is None:
        raise HTTPException(status_code=503, detail="Task 3 Soft MoE ONNX model not loaded.")

    restored_img, raw_weights, latency_ms = runner.restore(image)
    expert_names = ["clean_identity", "salt_and_pepper_specialist", "gaussian_blur_specialist", "rectangular_occlusion_specialist"]
    routing_dict = {name: round(float(w), 4) for name, w in zip(expert_names, raw_weights)}

    dominant_idx = int(np.argmax(raw_weights))
    dominant_expert = expert_names[dominant_idx]

    return SoftMoEResponse(
        task="soft_mixture_of_experts",
        status="success",
        output_image_base64=image_to_base64(restored_img),
        routing_weights=routing_dict,
        dominant_expert=dominant_expert,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/face-to-sketch", response_model=FaceToSketchResponse)
@app.post("/api/face-to-sketch", response_model=FaceToSketchResponse)
async def face_to_sketch(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    style_id: int = Form(0)
):
    """
    Task 4: Style-Conditioned Face-to-Sketch Generator Workspace Endpoint.
    Synthesizes facial sketch conditioned on Style 1 (0), Style 2 (1), or Style 3 (2).
    """
    image = await extract_image_from_request(file, image_base64)
    runner = get_task4_cgan_runner()
    if runner is None:
        raise HTTPException(status_code=503, detail="Task 4 cGAN Generator ONNX model not loaded.")

    # Bound style_id to [0, 2]
    style_idx = max(0, min(2, int(style_id)))
    style_names = {0: "Style 1 (Pencil / Classic)", 1: "Style 2 (Sketch / Artistic)", 2: "Style 3 (Caricature / Graphic)"}

    sketch_img, latency_ms = runner.generate(image, style_id=style_idx)
    return FaceToSketchResponse(
        task="face_to_sketch",
        status="success",
        sketch_image_base64=image_to_base64(sketch_img),
        style_id=style_idx,
        style_name=style_names[style_idx],
        latency_ms=round(latency_ms, 2)
    )


@app.post("/corrupt", response_model=CorruptionResponse)
@app.post("/api/corrupt", response_model=CorruptionResponse)
async def corrupt_image(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    corruption_type: str = Form("gaussian_blur"),
    severity_tier: Optional[str] = Form("medium"),
    prob: Optional[float] = Form(None),
    kernel_size: Optional[int] = Form(None),
    sigma: Optional[float] = Form(None),
    coverage: Optional[float] = Form(None),
    num_rects: Optional[int] = Form(None)
):
    """
    Interactive programmatic corruption endpoint using data/corruptions.py definitions.
    Supports preset severity tiers (low, medium, high) and custom parameter sliders.
    """
    image = await extract_image_from_request(file, image_base64)
    img_np = np.array(image.resize((128, 128)), dtype=np.uint8)
    corr_type = corruption_type.lower().strip()

    params: Dict[str, Any] = {"type": corr_type}

    if corr_type in ("clean", "none"):
        return CorruptionResponse(
            status="success",
            corrupted_image_base64=image_to_base64(image),
            corruption_type="clean",
            severity_tier=severity_tier,
            params={"type": "clean"}
        )

    elif corr_type in ("salt_and_pepper", "sp"):
        if prob is not None:
            p = float(prob)
        else:
            tier_probs = {"low": 0.03, "medium": 0.08, "high": 0.15}
            p = tier_probs.get(severity_tier.lower() if severity_tier else "medium", 0.08)
        params["prob"] = p
        corrupted_np = apply_salt_and_pepper(img_np, p)

    elif corr_type in ("gaussian_blur", "blur"):
        tier_settings = {
            "low": (3, 0.7),
            "medium": (5, 1.5),
            "high": (7, 2.5)
        }
        default_k, default_sig = tier_settings.get(severity_tier.lower() if severity_tier else "medium", (5, 1.5))
        k = int(kernel_size) if kernel_size is not None else default_k
        sig = float(sigma) if sigma is not None else default_sig
        params["kernel_size"] = k
        params["sigma"] = sig
        corrupted_np = apply_gaussian_blur(img_np, k, sig)

    elif corr_type in ("rectangular_occlusion", "occlusion", "occ"):
        tier_settings = {
            "low": (1, 0.10),
            "medium": (2, 0.20),
            "high": (3, 0.35)
        }
        default_num, default_cov = tier_settings.get(severity_tier.lower() if severity_tier else "medium", (2, 0.20))
        n_rects = int(num_rects) if num_rects is not None else default_num
        cov = float(coverage) if coverage is not None else default_cov
        h, w = img_np.shape[:2]
        rects = generate_occlusion_rectangles(h, w, n_rects, cov)
        params["num_rects"] = n_rects
        params["coverage"] = cov
        params["rectangles"] = rects
        corrupted_np = apply_rectangular_occlusion(img_np, rects)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported corruption type: {corruption_type}")

    corrupted_img = Image.fromarray(corrupted_np)
    return CorruptionResponse(
        status="success",
        corrupted_image_base64=image_to_base64(corrupted_img),
        corruption_type=corr_type,
        severity_tier=severity_tier,
        params=params
    )


@app.get("/api/samples")
@app.get("/samples")
async def list_sample_gallery():
    """
    Returns curated sample presets (clean pets and clean face portraits)
    for evaluator one-click testing across all 4 workspaces.
    """
    samples_dir = os.path.join(os.path.dirname(__file__), "static", "samples")
    if not os.path.exists(samples_dir):
        samples_dir = os.path.join("backend", "static", "samples")

    samples = []
    if os.path.exists(samples_dir):
        for fname in sorted(os.listdir(samples_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fpath = os.path.join(samples_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    b64 = image_to_base64(img)
                    category = "face" if "face" in fname.lower() else "pet"
                    display_name = fname.rsplit(".", 1)[0].replace("_", " ").title()
                    samples.append({
                        "id": fname,
                        "name": display_name,
                        "category": category,
                        "image_base64": b64
                    })
                except Exception as e:
                    print(f"Error reading sample image {fpath}: {e}")

    return {"status": "success", "samples": samples}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
