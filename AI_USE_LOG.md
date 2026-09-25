# AI Use Log

This log tracks all AI assistance, tools used, and verification/correction steps performed across each milestone.

| Date | Milestone / Task | Tool / Model Used | Action Description | Verification & Correction Method |
|---|---|---|---|---|
| 2026-09-25 | M1: Environment & Project Setup | Antigravity IDE (Gemini 3.7 Flash) | Checked local machine tools (Python 3.14, Node v24, Git), configured `.venv`, wrote `requirements.txt` and `requirements-colab.txt`. Initialized git repository with remote `https://github.com/UsmanBari/genai-restoration-studio.git`. | Verified Python/Node/Git versions via terminal. Confirmed GPU training pipeline constraints (CPU local, T4 GPU on Colab). Verified `.gitignore` covers datasets, checkpoints, venvs. |
| 2026-09-25 | M1: Directory Structure & Configs | Antigravity IDE (Gemini 3.7 Flash) | Created project folders (`data/`, `models/`, `training/`, `evaluation/`, `backend/`, `frontend/`, `docker/`, `configs/`, `scripts/`, `notebooks/`). Created central YAML configs. | Verified directory hierarchy and configuration schema for train/val/test splits and corruption parameters. |
| 2026-09-25 | M1: Oxford-IIIT Pet & FS2K Pipelines | Antigravity IDE (Gemini 3.7 Flash) | Created preparation scripts and PyTorch Dataset/DataLoader modules for Oxford-IIIT Pet and FS2K with runtime corruptions and deterministic test manifests. | Tested dataset classes, sample batch corruptions, style stratification on FS2K (15% val split), and fixed severity tier verification. |
| 2026-09-25 | M1: Backend & Frontend Skeletons | Antigravity IDE (Gemini 3.7 Flash) | Built FastAPI backend stubs (`/health`, `/universal-restoration`, `/hard-routing`, `/soft-mixture`, `/face-to-sketch`) and React+Tailwind frontend with 4 workspace pages. | Tested API endpoints with FastAPI test client and verified frontend build/pages and Dockerfile configs. |
| 2026-09-25 | M1: Colab Bootstrap | Antigravity IDE (Gemini 3.7 Flash) | Created `notebooks/colab_bootstrap.ipynb` to mount Google Drive, clone repo, install Colab CUDA deps, and test imports. | Verified notebook cell logic for Google Drive mounting and persistent paths (`GenAI-A1/`). |
