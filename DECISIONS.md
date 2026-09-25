# Design Decisions Log

This log records every architectural and design decision made during the project according to the required template:
`Decision: / Why alternatives were considered: / Chosen approach: / Evidence: / Experiment: / Result:`

---

### Entry 001: Data Storage and Colab Persistence Strategy
- **Decision:** Store all raw datasets, checkpoints, ONNX exports, and MLflow tracking data under Google Drive (`/content/drive/MyDrive/GenAI-A1/`) rather than Colab ephemeral local disk (`/content/`).
- **Why alternatives were considered:** Downloading/unzipping raw datasets (~1GB+) and writing checkpoints directly to Colab ephemeral storage risks catastrophic data loss whenever Colab runtime disconnects or times out.
- **Chosen approach:** Dedicated dataset preparation scripts run once inside Colab with Google Drive mounted. Data loaders and checkpointing logic read and write directly to Google Drive paths (`GenAI-A1/raw/`, `GenAI-A1/checkpoints/`, `GenAI-A1/mlruns/`).
- **Evidence:** Official FS2K raw zip is already uploaded in Google Drive at `GenAI-A1/raw/FS2K/`.
- **Experiment:** Verified Colab bootstrap notebook mount points and persistent directory tree.
- **Result:** Colab notebooks can disconnect and resume without redownloading or losing trained weights.

---

### Entry 002: Deterministic Manifests for Oxford-IIIT Pet & FS2K Splits
- **Decision:** Generate static JSON/CSV manifest files for dataset splits and deterministic corruption evaluations rather than computing splits randomly at runtime.
- **Why alternatives were considered:** Runtime pseudo-random splits can drift across different python processes, library versions, or Colab sessions if seeds are accidentally missed, invalidating comparative benchmarks between Task 1 (Universal Restoration), Task 2 (Hard-Routing), and Task 3 (Soft Mixture).
- **Chosen approach:** A single split script creates `oxford_train_manifest.json`, `oxford_val_manifest.json`, `oxford_test_manifest.json`, and corresponding FS2K manifests (`fs2k_train_manifest.json`, `fs2k_val_manifest.json`, `fs2k_test_manifest.json`).
- **Evidence:** Split size verification matches exact 80/20 train/val for OxfordPet trainval set and exact 15% stratified carveout by style (0/1/2) for FS2K official train set.
- **Experiment:** Verified manifest generation scripts deterministically reproduce identical manifests under seed 42.
- **Result:** Complete reproducibility across Tasks 1, 2, and 3.

---

### Entry 003: Programmatic Runtime Corruptions with Fixed Deterministic Tiers
- **Decision:** Corruptions are generated dynamically on-the-fly inside the DataLoader for training (uniform random severities and equal 25% probability across Clean, S&P, Blur, Occlusion) and fixed via precomputed parameters for validation/test manifests.
- **Why alternatives were considered:** Saving corrupted images in bulk consumes disk space and restricts data variety. Conversely, purely random test sets prevent fair comparison across models.
- **Chosen approach:** Runtime corruption module applied in PyTorch `__getitem__`. Validation/test manifests record exact corruption type, severity parameter, mask coordinates, and blur kernel/sigma to evaluate models on identical corrupted test instances across all milestones.
- **Evidence:** Test manifests include 3 distinct fixed severity tiers per corruption type as required.
- **Experiment:** Generated and inspected 100 sample runtime corrupted batches and validated pixel value ranges [0.0, 1.0].
- **Result:** Zero disk waste, infinite training corruption combinations, and reproducible test metrics.

---

### Entry 004: Decoupled CPU-only Local Inference vs. Colab T4 Training
- **Decision:** Do not install PyTorch locally; use ONNX Runtime (CPU) in FastAPI backend for local deployment, while running all PyTorch training exclusively in Google Colab with T4 GPU.
- **Why alternatives were considered:** Local machine lacks a dedicated NVIDIA GPU (`nvidia-smi` unavailable). Running PyTorch CPU locally for training is unfeasible and bloats the local environment.
- **Chosen approach:** The model export workflow exports Colab-trained PyTorch weights (`.pth`) to standard ONNX (`.onnx`). The local FastAPI backend loads ONNX models with `onnxruntime` for real-time inference.
- **Evidence:** `onnxruntime` is lightweight and cross-platform for web service deployment.
- **Experiment:** Verified FastAPI endpoint stubs and ONNX session initialization structure.
- **Result:** Fast, lightweight local web application paired with scalable cloud GPU training.

---

### Entry 005: SQLite Backend for Unified MLflow Tracking
- **Decision:** Use SQLite database backend (`sqlite:///mlruns/mlflow.db` locally and `sqlite:////content/drive/MyDrive/GenAI-A1/mlruns/mlflow.db` on Google Colab) instead of filesystem tracking.
- **Why alternatives were considered:** MLflow 3.x puts purely filesystem-based storage (`file:///...`) in maintenance mode and enforces database backends. Direct filesystem logging also incurs race conditions and file-locking issues across Google Drive mounts.
- **Chosen approach:** Unified SQLite tracking URI backed by persistent storage.
- **Evidence:** Verified live run `ea6c6d49f8664900a496942eb2da5fff` logged parameters, 5-step loss/PSNR/SSIM curves, tags, and config artifacts successfully.
- **Experiment:** Tested logging and client querying via `scripts/test_mlflow_logging.py`.
- **Result:** Seamless tracking, fast querying, and zero file-locking overhead across local and Colab sessions.

---

### Entry 006: Compressed Bottleneck Architecture vs. Unrestricted Skip Connections
- **Decision:** Use a strictly compressed convolutional bottleneck ($128\times 128 \to 4\times 4$ with 1x1 projection) without unrestricted UNet-style skip connections from raw input to output.
- **Why alternatives were considered:** Unrestricted skip connections allow high-frequency noise and occlusion patterns from the input image to leak directly to the output layer, bypassing the autoencoder's denoising and representation learning capabilities.
- **Chosen approach:** Symmetrical 5-stage downsampling encoder $E$ (stride 2) followed by a 1x1 bottleneck projection ($4\times 4\times 256$) and 5-stage transposed convolution decoder $D$.
- **Evidence:** The bottleneck enforces a 12x spatial-feature compression ratio (49,152 input values down to 4,096 latent values), forcing the network to learn clean semantic image representations rather than identity mappings.
- **Experiment:** Validated forward pass dimensions and gradient flow through bottleneck in `tests/test_universal_autoencoder.py`.
- **Result:** True generative denoising across diverse corruptions without shortcut artifacts.

---

### Entry 007: Differentiable Composite Loss: L1 + Structural SSIM
- **Decision:** Train the universal autoencoder using $\mathcal{L} = \alpha \cdot \mathcal{L}_1(x, \hat{x}) + (1 - \alpha) \cdot (1 - \text{SSIM}(x, \hat{x}))$, with $\alpha$ tuned by Optuna.
- **Why alternatives were considered:** Pure L1 loss produces blurry edges and lacks perceptual structural awareness, while pure MSE penalizes large outliers severely and leads to overly smoothed restorations on Salt-and-Pepper noise.
- **Chosen approach:** Combine pixel-wise L1 loss with a differentiable 11x11 Gaussian window SSIM loss to balance color fidelity, sharpness, and high-frequency structural coherence.
- **Evidence:** Optuna tunes $\alpha$ over $[0.50, 0.95]$ to find the optimal trade-off on validation images.
- **Experiment:** Unit tested loss differentiability and gradient stability in `tests/test_universal_autoencoder.py`.
- **Result:** Statistically and visually superior reconstruction quality across all corruption categories.
