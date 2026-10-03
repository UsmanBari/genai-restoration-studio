# Generative AI Restoration Studio & Face-to-Sketch Synthesis

A modular generative AI project implementing image restoration (Universal Autoencoder, Hard-Routed Autoencoders, Soft Mixture-of-Experts) and Paired Face-to-Sketch Synthesis (FS2K pix2pix GAN) with a FastAPI backend and React + Tailwind CSS interactive studio.

**Demonstration Video:** [https://youtu.be/AfU6j4_NIp0](https://youtu.be/AfU6j4_NIp0)

---

## 📁 Repository Structure

```
├── .gitignore                   # Ignores data, checkpoints, venvs, node_modules
├── requirements.txt             # Local CPU inference & full workspace dependencies
├── requirements-backend.txt     # Torchless lightweight production backend dependencies
├── requirements-colab.txt       # Colab T4 GPU training dependencies
├── AI_USE_LOG.md                # Tool usage and output verification log
├── EXPERIMENT_LOG.md            # Hyperparameters, runs, loss curves, metrics
├── DECISIONS.md                 # Design & architecture decisions
├── README.md                    # Project documentation & execution guide
├── docker-compose.yml           # Single-command orchestration for backend & frontend
│
├── configs/                     # Central YAML configurations & manifests
│   └── config.yaml              # Global project hyperparameters and dataset paths
│
├── data/                        # Dataset loaders, manifest generators, corruption pipelines
│   ├── __init__.py
│   ├── oxford_pet.py            # Oxford-IIIT Pet PyTorch Dataset (128x128, runtime corruptions)
│   ├── fs2k.py                  # FS2K PyTorch Dataset (photo-sketch pairs, style-aware)
│   ├── corruptions.py           # Runtime & deterministic corruption engine (S&P, Blur, Occlusion)
│   └── manifest_generator.py    # Manifest generator (80/20 train/val for Pet, 15% stratified for FS2K)
│
├── models/                      # Neural network architecture definitions & ONNX wrappers
│   ├── __init__.py              # PEP 562 lazy exports for torchless production import
│   ├── autoencoders.py          # Universal / Hard-routed restoration autoencoders
│   ├── moe.py                   # Soft Mixture-of-Experts (Gating network + Experts)
│   ├── classifier.py            # Corruption classifier for hard-routing
│   ├── gan.py                   # Style-conditioned UNet Generator & PatchGAN Discriminator
│   ├── onnx_runner.py           # Unified ONNX Runtime inference engine
│   ├── onnx_export.py           # Tasks 1-3 ONNX exporter
│   └── onnx_export_cgan.py      # Task 4 ONNX exporter
│
├── training/                    # Modular training routines & loss definitions
│   ├── __init__.py
│   ├── losses.py                # Reconstruction (L1, MSE, SSIM, Perceptual) & MoE losses
│   ├── losses_cgan.py           # Adversarial & L1 style-conditioned losses
│   ├── trainer_universal.py     # Task 1 trainer
│   ├── trainer_classifier.py    # Task 2 classifier trainer
│   ├── trainer_specialist.py    # Task 2 specialist autoencoders trainer
│   ├── trainer_moe.py           # Task 3 gating network & MoE trainer
│   └── trainer_cgan.py          # Task 4 style-conditioned GAN trainer
│
├── evaluation/                  # Evaluation benchmarks, metrics calculation & visualization
│   ├── __init__.py
│   ├── metrics.py               # PSNR, SSIM, LPIPS, Pixel-FD computation
│   ├── benchmark_universal.py   # Task 1 evaluation benchmark
│   ├── benchmark_hard_routing.py# Task 2 evaluation benchmark
│   ├── benchmark_moe.py         # Task 3 evaluation benchmark
│   └── benchmark_cgan.py        # Task 4 evaluation benchmark
│
├── backend/                     # FastAPI REST API serving ONNX models
│   ├── main.py                  # FastAPI application with REST endpoints
│   └── schemas.py               # Request/response Pydantic models
│
├── frontend/                    # Modern React + Vite Studio (Google Stitch Dark Theme)
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx              # Main dashboard with 4 workspace tabs
│       ├── main.jsx
│       └── index.css            # Custom CSS and dark theme styling
│
├── docker/                      # Containerization files
│   ├── backend.Dockerfile       # Python 3.11 slim backend container definition
│   ├── frontend.Dockerfile      # Multi-stage Node 20 / NGINX Alpine container
│   └── nginx.conf               # Frontend reverse proxy config
│
├── scripts/                     # Standalone CLI utilities
│   ├── fetch_models.py          # Remote GitHub Release download & SHA256 verifier
│   ├── prepare_oxford_pet.py    # Download & resize Oxford-IIIT Pet (Colab/Drive)
│   ├── prepare_fs2k.py          # Unpack & verify FS2K raw pairs (Colab/Drive)
│   └── verify_pipeline.py       # Local pipeline & manifest sanity checker
│
├── report/                      # IEEE Research Paper Source & Assets
│   ├── main.tex                 # Complete IEEE paper source with AI-Use Appendix
│   ├── references.bib           # Verified 9-item bibliography
│   └── figures/                 # Google Stitch UI design evidence figures
│
└── notebooks/                   # Google Colab T4 Training Notebooks
    ├── colab_bootstrap.ipynb    # Bootstrap notebook (Drive mount, repo sync, deps)
    ├── 02_task1_universal_autoencoder.ipynb # Task 1 Training Notebook
    ├── 03_task2_hard_routing.ipynb          # Task 2 Training Notebook
    ├── 04_task3_soft_moe.ipynb              # Task 3 Training Notebook
    └── 05_task4_cgan_sketch.ipynb           # Task 4 Training Notebook
```

---

## 🚀 Quick Start (Local)

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install local dependencies
pip install -r requirements.txt
```

### 2. Download & Verify ONNX Model Binaries
The 7 exported ONNX models (194.15 MB total) must sit under `models_onnx/` (mounted read-only into Docker). Run the automated fetch & verification script:
```powershell
python scripts/fetch_models.py
```
This script automatically downloads all 7 models from GitHub Release assets ([`milestone-5-complete`](https://github.com/UsmanBari/genai-restoration-studio/releases/tag/milestone-5-complete)) with real-time transfer progress, speed calculations, and SHA256 checksum verification.

Expected layout and byte sizes:
```
models_onnx/
├── task1_universal.onnx                            (17.36 MB, 18,207,156 bytes)
├── task2_classifier.onnx                           ( 4.48 MB,  4,698,438 bytes)
├── task2_specialist_salt_and_pepper.onnx          (17.36 MB, 18,207,156 bytes)
├── task2_specialist_gaussian_blur.onnx            (17.36 MB, 18,207,156 bytes)
├── task2_specialist_rectangular_occlusion.onnx    (17.36 MB, 18,207,156 bytes)
├── task3_soft_moe.onnx                            (56.58 MB, 59,330,355 bytes)
└── cgan_generator.onnx                            (63.63 MB, 66,719,645 bytes)
```
Total footprint: 194.15 MB (203,577,344 bytes). Verified 100% byte-accurate against the Colab training exports.

### 3. Face-to-Sketch Style ID Mapping
The FS2K Face-to-Sketch generator supports 3 distinct styles:
- **Style 1**: ID `0` (Classic / Fine Pencil Sketch)
- **Style 2**: ID `1` (Artistic / Shaded Sketch)
- **Style 3**: ID `2` (Caricature / Graphic Sketch)

### 4. Run Backend (Lightweight CPU ONNX Runtime)
```powershell
# In production / torchless environment:
pip install -r requirements-backend.txt
uvicorn backend.main:app --reload --port 8000
```
API Documentation is available at `http://localhost:8000/docs`.

### 5. Run Frontend (Vite Dev Server)
```powershell
cd frontend
npm install
npm run dev
```
Frontend will be live at `http://localhost:5173`.

### 6. Containerized Deployment (Docker Compose / WSL2 Ubuntu)
Run both backend (lightweight Python 3.11 + onnxruntime) and frontend (Nginx reverse proxy) with:
```bash
docker compose up --build
```
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000` (docs at `http://localhost:8000/docs`)
- **Models Volume**: `models_onnx/` is mounted read-only (`:ro`) at runtime without baking binary weights into the container image.


---

## ☁️ Google Colab Training Workflow

All heavy training runs on Google Colab (Free T4 GPU) using Google Drive for persistent storage:
1. Open `notebooks/00_bootstrap.ipynb` in Colab.
2. Mount Google Drive (`/content/drive/MyDrive/GenAI-A1/`).
3. Prepare datasets using `scripts/prepare_oxford_pet.py` and `scripts/prepare_fs2k.py`.
4. Train models and log metrics automatically to MLflow at `/content/drive/MyDrive/GenAI-A1/mlruns/`.
5. Checkpoints (`.pth`) and ONNX models (`.onnx`) are automatically saved to Google Drive.
6. Copy the `.onnx` models locally into `models_onnx/` for local serving.

