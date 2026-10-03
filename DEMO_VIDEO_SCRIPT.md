# Generative AI Assignment 1: Demonstration Video Script
**Student:** Muhammad Usman Bari (23i-0680, Section A)  
**Target Duration:** 5:30 – 6:30 minutes (Strictly within the 5–7 minute specification)  
**Deliverable URL:** [https://youtu.be/AfU6j4_NIp0](https://youtu.be/AfU6j4_NIp0)

---

## Video Timeline Overview

| Section | Timestamp | Description | Key Focus Area |
|---|---|---|---|
| **1. Intro & Container Startup** | 0:00 – 0:45 | Setup, Docker Compose / Local service startup, `/health` endpoint check | Full containerization, torchless ONNX backend |
| **2. Runtime Corruption Engine** | 0:45 – 1:30 | Interactive corruption toolbar, parameter sliders, preset tiers | Deterministic & runtime math ($p, k, \sigma, c$) |
| **3. Task 1: Universal Autoencoder** | 1:30 – 2:15 | Single-model restoration across S&P, Blur, Occlusion | Gated skip connection dynamics, bottleneck trade-off |
| **4. Task 2: Hard-Routed Specialists** | 2:15 – 3:15 | 4-class classifier, probability radar, expert routing, Oracle toggle | $99.95\%$ accuracy, lossless clean bypass ($100\text{ dB}$) |
| **5. Task 3: Soft Mixture-of-Experts** | 3:15 – 4:15 | Dynamic routing weight meters ($w_0, w_1, w_2, w_3$), temperature scaling | Continuous blending, per-corruption gains (+1.13 to +1.27 dB) |
| **6. Task 4: Face-to-Sketch Studio** | 4:15 – 5:15 | FS2K synthesis across Styles 1, 2, 3, side-by-side view, download | Style conditioning, PatchGAN discriminator analysis |
| **7. MLflow Tracking & Wrap-Up** | 5:15 – 6:00 | Local MLflow dashboard (17 runs), hyperparameter sweeps, conclusion | Reproducibility, metrics integrity, system summary |

---

## Detailed Timestamped Shot List & Spoken Script

---

### Segment 1: Introduction & Architecture Startup (0:00 – 0:45)
- **Visual:** Terminal window in split-screen next to web browser.
- **Action:**
  1. Open terminal in workspace root.
  2. Run `docker-compose up -d` (or show active services: backend on port 8000, frontend on port 3000).
  3. Navigate browser to `http://localhost:8000/health` showing JSON response with status `"ok"` and all 7 ONNX models loaded.
  4. Navigate browser to `http://localhost:3000/` showing the Google Stitch-inspired dark mode UI (`#0B0F19`).
- **Spoken Narrative:**
  > *"Hello, my name is Muhammad Usman Bari (Roll Number 23i-0680, Section A). This video demonstrates my submission for Generative AI Assignment 1: an end-to-end Generative Restoration and Synthesis Studio.*  
  > *Our architecture is fully containerized using Docker Compose. The backend is an asynchronous FastAPI service utilizing a torchless, CPU-optimized ONNX Runtime for all seven neural networks, reducing container size to 220 MB. The frontend is built in React and styled with a curated Google Stitch dark-mode design system. As seen on our `/health` endpoint, all seven models are serialized, verified, and ready for instant inference."*

---

### Segment 2: Interactive Runtime Corruption Simulator (0:45 – 1:30)
- **Visual:** Frontend top toolbar displaying the image upload card, corruption controls, and sample selector.
- **Action:**
  1. Select a clean sample image of a pet from the Oxford-IIIT Pet test set gallery.
  2. Click the **"Salt-and-Pepper"** button; adjust the noise slider from $p=0.03$ (Low) to $p=0.15$ (High); click **"Corrupt"** to see live noise injection.
  3. Click **"Gaussian Blur"**; adjust kernel slider to $k=7$ and $\sigma=2.5$; observe blur effect.
  4. Click **"Rectangular Occlusion"**; select $n=2$ masks with $20\%$ coverage; show black occlusion masks placed over the pet image.
- **Spoken Narrative:**
  > *"To ensure rigorous evaluation, our studio includes an interactive Runtime Corruption Toolbar implementing exact mathematical degradations from `data/corruptions.py`.*  
  > *Users can choose between low, medium, and high preset tiers or fine-tune parameters directly: adjusting salt-and-pepper probability up to 50%, setting Gaussian blur kernels from 3 to 9 with continuous sigma scaling, and dynamically placing up to 3 random rectangular occlusions covering up to 35% of the spatial canvas."*

---

### Segment 3: Workspace 1 — Universal Denoising Autoencoder (1:30 – 2:15)
- **Visual:** Switch to the **Universal Restoration Workspace** tab.
- **Action:**
  1. Load a corrupted pet image (e.g., Gaussian Blur tier 2).
  2. Click **"Restore Image"**.
  3. Show the restored result canvas side-by-side with the corrupted input; point out the latency badge (showing ~35–55 ms end-to-end API latency / 60.56 ms raw ONNX model benchmark).
  4. Load a Salt-and-Pepper corrupted image; click restore to show impulse noise suppression.
  5. Load a heavy Occlusion image; click restore to show the model inpainting the masked region.
- **Spoken Narrative:**
  > *"In Task 1, we evaluate our Universal Multi-Corruption Autoencoder. This model operates under a strict 6.0-times bottleneck compression ratio (down to 8x8x96 channels) to prevent identity memorization.*  
  > *To restore fine facial contours and fur textures without leaking corruption artifacts, we engineered a learned high-resolution Gated Skip Connection. Over clean or blurred regions, the gate dynamically opens to route high frequencies; over corrupted noise impulses or occlusions, the gate closes, forcing pure generative hallucination from the bottleneck. The model achieves 26.24 dB on blur and 23.96 dB on noise with a raw ONNX inference latency of 60.56 milliseconds on CPU (35 to 55 milliseconds in the live API)."*

---

### Segment 4: Workspace 2 — Hard-Routed Specialist Autoencoders (2:15 – 3:15)
- **Visual:** Switch to the **Hard-Routing Workspace** tab.
- **Action:**
  1. Load a Salt-and-Pepper corrupted image; click **"Classify & Route"**.
  2. Highlight the **Corruption Classifier Panel**: show the predicted badge (`Salt-and-Pepper Noise`), confidence score (`1.0000`), and probability distribution radar.
  3. Point out the active specialist indicator (`Executing S&P Specialist Autoencoder`); show the restored output ($27.56\text{ dB}$).
  4. Load a pristine clean image; click **"Classify & Route"**; demonstrate the **Lossless Clean Identity Bypass** ($100.00\text{ dB}$, near-zero latency $\sim 17\text{ ms}$).
  5. Toggle the **Oracle Override** checkbox to demonstrate manual specialist forcing.
- **Spoken Narrative:**
  > *"Task 2 addresses the capacity sharing trade-off of universal autoencoders through a two-stage Hard-Routing pipeline.*  
  > *An upfront 4-class convolutional classifier—achieving 99.95% test accuracy across 3,669 test images—predicts the corruption class. If a corruption is detected, the image is dispatched to an isolated, dedicated specialist autoencoder. If the input is clean, it executes an exact, mathematically lossless identity bypass.*  
  > *This lifts overall restoration performance to 32.85 dB PSNR—a dramatic +10.34 dB improvement over the universal baseline. Users can also use the Oracle override control to manually evaluate individual specialists on arbitrary inputs."*

---

### Segment 5: Workspace 3 — Soft Mixture-of-Experts (MoE) (3:15 – 4:15)
- **Visual:** Switch to the **Soft Mixture-of-Experts Workspace** tab.
- **Action:**
  1. Load a Gaussian blur image; click **"Run Soft MoE"**.
  2. Point to the **Routing Weights Panel**: show horizontal bars ($w_0 \approx 0.00, w_1 \approx 0.00, w_2 = 1.00, w_3 \approx 0.00$).
  3. Load an image with compound/mild degradation or boundary noise; observe non-zero collaborative weights blending across experts.
  4. Highlight the dominant expert badge and CPU inference latency badge ($\sim 180\text{ ms}$).
- **Spoken Narrative:**
  > *"While Hard Routing is highly effective, discrete routing is vulnerable to classification edge cases. In Task 3, we implement a Differentiable Soft Mixture-of-Experts architecture with temperature scaling at tau = 0.50.*  
  > *A convolutional gating network computes continuous routing weights across all four expert branches: clean identity, salt-and-pepper specialist, blur specialist, and occlusion specialist. All four branches contribute to the final convex combination.*  
  > *Because all specialists were fine-tuned jointly with the gating network using a composite cross-entropy and reconstruction objective, Soft MoE outperforms Hard Routing across every single degradation type: +1.13 dB on noise, +1.27 dB on blur, and +0.71 dB on occlusion."*

---

### Segment 6: Workspace 4 — Style-Conditioned Face-to-Sketch Studio (4:15 – 5:15)
- **Visual:** Switch to the **Face-to-Sketch Generator** tab.
- **Action:**
  1. Select a sample photograph from the FS2K test set gallery (or upload a custom portrait).
  2. Select **Style 1: Classic / Pencil Sketch**; click **"Generate Sketch"**; show the crisp line-art synthesis.
  3. Switch to **Style 2: Artistic / Shaded Sketch**; click **"Generate Sketch"**; observe the tonal cross-hatching and shaded textures.
  4. Switch to **Style 3: Caricature / Graphic Outline**; click **"Generate Sketch"**; show the emphasized geometric contours.
  5. **Explicit Download Action:** Click the **"Download High-Res PNG"** button on screen; show the downloaded sketch image file (`sketch_style1.png`) appearing in the browser's download shelf / desktop folder to satisfy spec line 35.
  6. Show the side-by-side zoom comparison.
- **Spoken Narrative:**
  > *"Task 4 implements paired face-to-sketch synthesis on the FS2K dataset using a style-conditioned Conditional GAN.*  
  > *The generator employs a 5-stage U-Net encoder-decoder conditioned via learned style embeddings, paired with a 70x70 PatchGAN discriminator. Users can seamlessly toggle between Style 1 (Classic Pencil), Style 2 (Artistic Shaded), and Style 3 (Graphic Caricature).*  
  > *In our report, we document an honest empirical finding regarding PatchGAN dynamics: discriminator accuracy saturated at 91 to 94% by Epoch 3. Despite two extensive rebalancing interventions, adversarial updates were throttled to 0-5% of batches, causing the generator to rely primarily on L1 reconstruction. This produces smooth, shaded pencil textures rather than hyper-sharp vectorized ink lines—achieving 18.06 dB on Style 1 and 16.24 dB overall."*

---

### Segment 7: MLflow Experiment Tracking & Conclusion (5:15 – 6:00)
- **Visual:** Switch browser tab to the local **MLflow Tracking UI** (`http://localhost:5000` or local sqlite database view).
- **Action:**
  1. Show the experiment `Task4_cGAN_Face_To_Sketch` with all 17 logged training runs.
  2. Click into the winning run (`task4_cgan_final_40epochs`); show logged metrics (`val_PSNR`, `val_SSIM`, `val_L1`, `val_Pixel_FD`, `D_loss`, `G_loss`).
  3. Show logged hyperparameter tags (`lambda_L1 = 275.0`, `lr_g = 0.000353`, `lr_d = 0.0000706`, `batch_size = 8`).
  4. Conclude with summary slide.
- **Spoken Narrative:**
  > *"To ensure complete transparency and scientific reproducibility, all training runs, Bayesian Optuna sweeps, and evaluation benchmarks were tracked with MLflow.*  
  > *Here in our local tracking server, all 17 runs of our style-conditioned GAN are archived with exact metric curves, loss histories, and artifact checkpoints.*  
  > *In summary, this assignment demonstrated the trade-offs of bottleneck compression, the massive domain gains of expert routing, the collaborative power of Soft MoE, and the production efficiency of torchless ONNX deployment within Docker. Thank you for watching!"*

---

## Recording Checklist for Presenter

- [ ] Ensure backend is running locally on port 8000 (`uvicorn backend.main:app --port 8000`).
- [ ] Ensure frontend is running on port 3000 (`npm run dev` or via `docker-compose up`).
- [ ] Have sample images ready from `data/oxford_pet/` and `data/fs2k/`.
- [ ] Keep speech paced evenly to finish between 5:30 and 6:00 minutes.
- [ ] Upload final recorded video as Unlisted or Public on YouTube and update the URL in `report/main.tex` and `README.md`.
