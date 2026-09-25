# Generative AI Restoration Studio & Face-to-Sketch Synthesis

A modular generative AI project implementing image restoration (Universal Autoencoder, Hard-Routed Autoencoders, Soft Mixture-of-Experts) and Paired Face-to-Sketch Synthesis (FS2K pix2pix GAN) with a FastAPI backend and React + Tailwind CSS interactive studio.

---

## 📁 Repository Structure

```
├── .gitignore                   # Ignores data, checkpoints, venvs, node_modules
├── requirements.txt             # Local CPU inference & tooling dependencies
├── requirements-colab.txt       # Colab T4 GPU training dependencies
├── AI_USE_LOG.md                # Tool usage and output verification log
├── EXPERIMENT_LOG.md            # Hyperparameters, runs, loss curves, metrics
├── DECISIONS.md                 # Design & architecture decisions
├── README.md                    # Project documentation & execution guide
├── split_train_test.py          # Reference official FS2K split script
│
├── configs/                     # Central YAML configurations & manifests
│   ├── config.yaml              # Global project hyperparameters and dataset paths
│   └── manifests/               # Precomputed deterministic split & corruption manifests
│
├── data/                        # Dataset loaders, manifest generators, corruption pipelines
│   ├── __init__.py
│   ├── oxford_pet.py            # Oxford-IIIT Pet PyTorch Dataset (128x128, runtime corruptions)
│   ├── fs2k.py                  # FS2K PyTorch Dataset (photo-sketch pairs, style-aware)
│   ├── corruptions.py           # Runtime & deterministic corruption engine (S&P, Blur, Occlusion)
│   └── manifest_generator.py    # Manifest generator (80/20 train/val for Pet, 15% stratified for FS2K)
│
├── models/                      # Neural network architecture definitions & ONNX wrappers
│   ├── __init__.py
│   ├── autoencoders.py          # Universal / Hard-routed restoration autoencoders
│   ├── moe.py                   # Soft Mixture-of-Experts (Gating network + Experts)
│   ├── classifier.py            # Corruption classifier for hard-routing
│   ├── gan.py                   # pix2pix UNet Generator & PatchGAN Discriminator
│   └── onnx_runner.py           # Unified ONNX Runtime inference engine
│
├── training/                    # Modular training routines & loss definitions
│   ├── __init__.py
│   ├── losses.py                # Reconstruction (L1, MSE, SSIM, Perceptual) & GAN losses
│   ├── trainer_universal.py     # Task 1 trainer
│   ├── trainer_routing.py       # Task 2 classifier & specialized autoencoders trainer
│   ├── trainer_moe.py           # Task 3 gating network & MoE trainer
│   └── trainer_gan.py           # Task 4 pix2pix GAN trainer
│
├── evaluation/                  # Evaluation benchmarks, metrics calculation & visualization
│   ├── __init__.py
│   ├── metrics.py               # PSNR, SSIM, LPIPS, FID computation
│   └── benchmark.py             # Deterministic test evaluation across corruption tiers
│
├── backend/                     # FastAPI REST API serving ONNX models
│   ├── main.py                  # FastAPI application with REST endpoints
│   ├── schemas.py               # Request/response Pydantic models
│   └── routes/                  # API routers (/health, /universal, /routing, /moe, /sketch)
│
├── frontend/                    # Modern React + Vite + Tailwind CSS Studio
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── App.jsx              # Main dashboard with 4 workspace tabs
│       ├── main.jsx
│       ├── index.css            # Tailwind & sleek dark glassmorphic styling
│       └── components/          # Interactive image upload & comparison components
│
├── docker/                      # Containerization files
│   ├── backend.Dockerfile       # Backend container definition
│   ├── frontend.Dockerfile      # Frontend container definition
│   └── nginx.conf               # Frontend reverse proxy config
├── docker-compose.yml           # Single-command orchestration for backend & frontend
│
├── scripts/                     # Standalone CLI utilities
│   ├── prepare_oxford_pet.py    # Download & resize Oxford-IIIT Pet (Colab/Drive)
│   ├── prepare_fs2k.py          # Unpack & verify FS2K raw pairs (Colab/Drive)
│   └── verify_pipeline.py       # Local pipeline & manifest sanity checker
│
└── notebooks/                   # Google Colab T4 Training Notebooks
    ├── colab_bootstrap.ipynb    # Bootstrap notebook (Drive mount, repo sync, deps)
    ├── task1_universal.ipynb    # Task 1: Universal Autoencoder Training
    ├── task2_hard_routing.ipynb # Task 2: Classifier & Hard-Routed Restoration
    ├── task3_soft_moe.ipynb     # Task 3: Soft Mixture-of-Experts Training
    └── task4_fs2k_gan.ipynb     # Task 4: FS2K pix2pix GAN Training
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

### 2. Run Backend
```powershell
uvicorn backend.main:app --reload --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.

### 3. Run Frontend
```powershell
cd frontend
npm install
npm run dev
```
Frontend will be live at `http://localhost:5173`.

### 4. Docker Compose
Run both backend and frontend simultaneously with a single command:
```powershell
docker compose up --build
```

---

## ☁️ Google Colab Training Workflow

All heavy training runs on Google Colab (Free T4 GPU) using Google Drive for persistent storage:
1. Open `notebooks/colab_bootstrap.ipynb` in Colab.
2. Mount Google Drive (`/content/drive/MyDrive/GenAI-A1/`).
3. Prepare datasets using `scripts/prepare_oxford_pet.py` and `scripts/prepare_fs2k.py`.
4. Train models and log metrics automatically to MLflow at `/content/drive/MyDrive/GenAI-A1/mlruns/`.
5. Checkpoints (`.pth`) and ONNX models (`.onnx`) are automatically saved to Google Drive.
6. Copy the `.onnx` models locally into `models/` for local FastAPI serving.
