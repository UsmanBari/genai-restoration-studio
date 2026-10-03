# Generative AI Assignment 1: Video Demonstration Script (Roman Urdu)

**Student:** Muhammad Usman Bari (Roll No: 23i-0680, Section A)  
**Target Duration:** 5:30 – 6:30 minutes (Strictly within the 5–7 min requirement)  
**Deliverable:** Demonstration Video: [https://youtu.be/AfU6j4_NIp0](https://youtu.be/AfU6j4_NIp0)  
**Language:** Roman Urdu (Technical terms in English)

---

## Video Timeline & Segment Breakdown

| # | Segment Name | Timestamp | Segment Duration | Running Total | Key Screen Action |
|---|---|---|---|---|---|
| **1** | **Startup & Architecture Health** | 0:00 – 0:45 | 0:45 | **0:45** | WSL terminal, Docker compose, `/api/health`, Frontend UI |
| **2** | **Image Upload & Runtime Corruption** | 0:45 – 1:25 | 0:40 | **1:25** | Clean Pet Cat, S&P noise + Medium severity, Choose Image button |
| **3** | **Task 1: Universal Autoencoder** | 1:25 – 2:05 | 0:40 | **2:05** | Run Deep Restoration, Restored view, Latency badge |
| **4** | **Task 2: Hard-Routed Restoration** | 2:05 – 2:50 | 0:45 | **2:50** | Clean Pet Dog, Occlusion High, 4 Probability bars, Selected expert |
| **5** | **Task 3: Soft Mixture-of-Experts** | 2:50 – 3:35 | 0:45 | **3:35** | Gaussian Blur, 4 Gating weights ($w_0..w_3$), Dominant expert |
| **6** | **Task 4: Face-to-Sketch Studio** | 3:35 – 4:35 | 1:00 | **4:35** | Webcam / Portrait, Style 1/2/3 compare, Download Result PNG |
| **7** | **MLflow Experiment Tracking** | 4:35 – 5:30 | 0:55 | **5:30** | PowerShell 4 ports (5000, 5001, 5002, 5003), Model training toggle |
| **8** | **Limitations & Honest Closing** | 5:30 – 6:05 | 0:35 | **6:05** | PatchGAN D-dominance discussion, Summary & conclusion |

---

## Detailed Step-by-Step Script

---

### Segment 1: Startup & Architecture Health Check
- **TIMESTAMP:** 0:00 – 0:45 (Duration: 0:45)
- **RUNNING TOTAL:** 0:45

#### SCREEN ACTION:
1. WSL terminal window screen par kholo.
2. Directory mein jao aur Docker containers start karo:
   ```bash
   cd /mnt/c/Users/usmanbari/Desktop/Gen-Ai-A1
   sudo docker compose up -d
   ```
   *(Wait 2 seconds jab tak containers start ho jayein).*
3. Web browser kholo aur pehli tab par jao:
   `http://localhost:8000/api/health`
   *(Screen par JSON response dikhao jahan status `"ok"` hai aur 8 ONNX models `true` hain).*
4. Browser ki doosri tab par jao:
   `http://localhost:3000`
   *(Top-right header mein green pill dikhao: `Backend: Online (CPU)`).*

#### BOLNA HAI (Roman Urdu):
> *"Assalam-o-Alaikum, mera naam Muhammad Usman Bari hai, roll number 23i-0680, Section A. Yeh video meri Generative AI Assignment 1 ki demonstration hai: ek end-to-end Image Restoration aur Style-Conditioned Face-to-Sketch Studio.*
>
> *Humara architecture fully containerized hai using Docker Compose. Backend ek asynchronous FastAPI service hai jo CPU-optimized ONNX Runtime use karti hai for all 7 neural networks—bina kisi heavy PyTorch ya CUDA dependency ke.*
>
> *Jaisa ke aap browser mein `/api/health` endpoint par dekh sakte hain, backend status bilkul OK hai aur saare ONNX models successfully loaded hain. Frontend React 18 mein built hai with a Google Stitch dark-mode aesthetic, aur top-right par `Backend: Online (CPU)` indicator green active hai."*

---

### Segment 2: Image Upload & Runtime Corruption Engine
- **TIMESTAMP:** 0:45 – 1:25 (Duration: 0:40)
- **RUNNING TOTAL:** 1:25

#### SCREEN ACTION:
1. Browser mein `http://localhost:3000` par **"Universal Restoration"** tab active rakho.
2. Controls column mein **"Upload Custom Image"** ke neeche **"Choose Image"** dropzone box cursor se point karo.
3. **"Preset Sample Gallery"** mein se **"Clean Pet Cat"** thumbnail par click karo.
4. Input image preview mein clean cat load ho jayegi.
5. **"Simulate Programmatic Corruption"** card par jao:
   - First dropdown mein **"Salt & Pepper Noise"** select karo.
   - Second dropdown mein **"Medium Severity"** select karo.
6. Button par click karo: **"Apply Runtime Corruption"**.
7. Input preview panel mein noise inject ho jayegi.

#### BOLNA HAI (Roman Urdu):
> *"Humare studio mein real-time dynamic corruption simulator integrated hai jo mathematically exact degradations apply karta hai.*
>
> *Yahan user apna custom image bhi upload kar sakta hai using the 'Choose Image' button, ya predefined Oxford-IIIT Pet samples select kar sakta hai. Yahan main 'Clean Pet Cat' sample select kar raha hoon.*
>
> *Ab hum runtime corruption toolbox se 'Salt & Pepper Noise' aur 'Medium Severity' select karke 'Apply Runtime Corruption' click karte hain. Image mein live impulse noise inject ho chuki hai. Tasks 1, 2, aur 3 ke liye hum in-distribution Oxford Pet images hi use kar rahe hain kyunke yeh autoencoders specifically pet domain par train hue hain."*

---

### Segment 3: Task 1 – Universal Denoising Autoencoder
- **TIMESTAMP:** 1:25 – 2:05 (Duration: 0:40)
- **RUNNING TOTAL:** 2:05

#### SCREEN ACTION:
1. **"Universal Restoration"** tab par hi rehte hue main action button par click karo:
   **"Run Deep Restoration"**
2. 1 second wait karo jab tak ONNX inference execute ho.
3. Screen par **"Input Image"** aur **"Restored Image"** ka side-by-side comparison dikhao.
4. **"Execution Diagnostics"** card mein top-right par green latency badge dikhao:
   *(Live API latency: 35 – 55 ms | Raw ONNX CPU benchmark: 60.56 ms).*

#### BOLNA HAI (Roman Urdu):
> *"Task 1 mein humara Universal Denoising Autoencoder ek single shared model hai jo saari degradations ko handle karta hai.*
>
> *Yeh model strict 6-times bottleneck compression use karta hai to prevent identity memorization, aur high-frequency details recover karne ke liye humne ek learned Gated Skip Connection introduce kiya hai.*
>
> *Jab hum 'Run Deep Restoration' click karte hain, toh model noise suppress karke clear cat image reconstruct kar deta hai. Live application mein end-to-end CPU latency 35 se 55 milliseconds aati hai, jabke isolated raw ONNX CPU benchmark 60.56 milliseconds hai. Overall test benchmark par is model ne 22.51 dB overall test average achieve kiya, aur Salt & Pepper noise par specific score 23.96 dB hai."*

---

### Segment 4: Task 2 – Hard-Routed Specialist Restoration
- **TIMESTAMP:** 2:05 – 2:50 (Duration: 0:45)
- **RUNNING TOTAL:** 2:50

#### SCREEN ACTION:
1. Top navigation se 2nd tab par click karo:
   **"Hard-Routed Restoration"**
2. Sample gallery mein se **"Clean Pet Dog"** par click karo.
3. Corruption dropdowns mein:
   - Corruption: **"Rectangular Occlusion"** select karo.
   - Severity: **"High Severity"** select karo.
4. Click button: **"Apply Runtime Corruption"** (Input image par black occlusions aa jayengi).
5. Click main button: **"Run Deep Restoration"**.
6. Execution Diagnostics panel par focus karo:
   - **Predicted Corruption:** `RECTANGULAR_OCCLUSION`
   - **Selected Expert:** `specialist_rectangular_occlusion`
   - 4 probability bars dikhao (Occlusion bar close to 100%).
   - Restored output panel mein inpainting result dikhao.

#### BOLNA HAI (Roman Urdu):
> *"Task 2 capacity sharing trade-off ko solve karta hai through a two-stage Hard-Routing pipeline.*
>
> *Pehle ek lightweight 4-class convolutional corruption classifier input ko analyze karta hai, jiski test accuracy 99.95% hai across 3,669 test images. Predict hone ke baad input image dedicated specialist autoencoder ko dispatch ho jaati hai.*
>
> *Yahan humne 'Clean Pet Dog' par 'Rectangular Occlusion' apply kiya. Classifier ne correctly detect kiya aur dedicated Occlusion Specialist ko route kiya. Agar input bilkul clean ho, toh system mathematically lossless Clean Identity Bypass trigger karta hai. Hard routing ne overall test performance ko 32.85 dB PSNR tak pohchaya—jo ke universal baseline se +10.34 dB ka huge improvement hai."*

---

### Segment 5: Task 3 – Soft Mixture-of-Experts (MoE)
- **TIMESTAMP:** 2:50 – 3:35 (Duration: 0:45)
- **RUNNING TOTAL:** 3:35

#### SCREEN ACTION:
1. Top navigation se 3rd tab par click karo:
   **"Soft Mixture-of-Experts Restoration"**
2. Sample gallery mein se **"Clean Pet Cat"** ya **"Clean Pet Dog"** select karo.
3. Corruption dropdowns mein:
   - Corruption: **"Gaussian Blur"**
   - Severity: **"Medium Severity"**
4. Click button: **"Apply Runtime Corruption"**.
5. Click main button: **"Run Deep Restoration"**.
6. Execution Diagnostics panel par focus karo:
   - **Dominant Expert:** `GAUSSIAN_BLUR`
   - 4 continuous gating weight bars dikhao ($w_0$ Clean, $w_1$ S&P, $w_2$ Blur, $w_3$ Occlusion).
   - Show how $w_2$ is dominant while others contribute continuous soft blending.

#### BOLNA HAI (Roman Urdu):
> *"Task 3 mein hum discrete hard decisions ke bajaye ek Differentiable Soft Mixture-of-Experts architecture use karte hain with temperature-scaled gating at tau = 0.50.*
>
> *Gating network chaaron expert branches—Clean Identity, Salt & Pepper, Blur, aur Occlusion—par continuous weights assign karta hai, aur final output unka weighted convex combination hota hai.*
>
> *Jab hum Gaussian blur image par Soft MoE run karte hain, toh diagnostic panel par aap chaaron continuous gating weights dekh sakte hain jahan blur expert dominant hai. End-to-end multi-expert fine-tuning ki wajah se Soft MoE ne Hard Routing ke muqable mein har real degradation par consistent gains achieve kiye: Salt & Pepper par +1.13 dB, Blur par +1.27 dB, aur Occlusion par +0.71 dB improvement."*

---

### Segment 6: Task 4 – Style-Conditioned Face-to-Sketch Studio & Download
- **TIMESTAMP:** 3:35 – 4:35 (Duration: 1:00)
- **RUNNING TOTAL:** 4:35

#### SCREEN ACTION:
1. Top navigation se 4th tab par click karo:
   **"Face-to-Sketch Generator"**
2. **"Webcam"** button par click karo, modal khulega, apna face capture karo (ya custom portrait upload karo / "Clean Face Portrait" sample lo).
3. **Style Selector** par jao:
   - Click **"Style 1"** (Pencil / Classic) -> Click **"Synthesize Facial Sketch"** -> Output sketch dikhao.
   - Click **"Style 2"** (Sketch / Artistic) -> Click **"Synthesize Facial Sketch"** -> Output sketch dikhao (darker cross-hatching tone).
   - Click **"Style 3"** (Caricature / Graphic) -> Click **"Synthesize Facial Sketch"** -> Output sketch dikhao (graphic contours).
4. **Mandatory Spec Action:**
   - Output display ke top-right par **"Download Result"** button click karo.
   - OS file browser ya Downloads folder / browser download shelf open karke downloaded PNG file (`sketch_style1.png` ya `sketch_style*.png`) verify karao.

#### BOLNA HAI (Roman Urdu):
> *"Task 4 FS2K dataset par paired Face-to-Sketch synthesis perform karta hai using a Style-Conditioned Conditional GAN.*
>
> *Generator ek 5-stage U-Net architecture hai jo categorical style embeddings receive karta hai. Hum webcam se live face portrait capture kar sakte hain ya portrait upload kar sakte hain.*
>
> *Ab hum teeno styles ko sequentially evaluate karte hain:*
> *Pehle Style 1 (Classic Pencil)—yeh crisp facial outlines produce karta hai with 18.06 dB PSNR.*
> *Phir Style 2 (Artistic Shaded)—yeh darker tones aur heavy cross-hatching deliver karta hai.*
> *Aur Style 3 (Graphic Caricature)—yeh emphasized structural contours generate karta hai.*
> *Teeno styles distinctly alag visual features render karte hain.*
>
> *Ab hum spec requirement ke mutabiq 'Download Result' button click karte hain, aur aap dekh sakte hain ke synthesized high-resolution PNG file successfully humare Downloads folder mein save ho chuki hai."*

---

### Segment 7: MLflow Experiment Tracking (Tasks 1, 2, 3, 4)
- **TIMESTAMP:** 4:35 – 5:30 (Duration: 0:55)
- **RUNNING TOTAL:** 5:30

#### SCREEN ACTION:
1. Terminal / PowerShell par jao project root directory se. Chaaron separate MLflow tracking databases active dikhao:
   - **Task 4 (Port 5000):** `.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`
   - **Task 2 (Port 5001):** `.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlruns_task2.db --port 5001`
   - **Task 3 (Port 5002):** `.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlruns_task3.db --port 5002`
   - **Task 1 (Port 5003):** `.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5003`

2. Browser mein chaaron tabs sequentially dikhao (10–12 seconds each):
   - **Tab 1 (`localhost:5000` - Task 4):**
     - Top-left par **"Model training"** toggle switch pehle se selected ho.
     - Experiment **`Task4_cGAN_Face_To_Sketch`** par click karo -> Table mein **17 logged runs** dikhao -> Best run khol kar **"Model metrics"** tab par loss/PSNR curves dikhao.
   - **Tab 2 (`localhost:5001` - Task 2):**
     - Experiment **`task2_classifier_optuna_study`** (50 trials), **`Task2_Classifier_Training`** (3 runs), aur **`Task2_Specialists`** (9 specialist runs) dikhao.
   - **Tab 3 (`localhost:5002` - Task 3):**
     - Experiment **`Task3_Soft_Mixture_of_Experts`** kholo -> Final 2-phase training run **`Soft_MoE_2Phase_Final`** aur metric history dikhao. (Optuna trials report aur EXPERIMENT_LOG mein hain).
   - **Tab 4 (`localhost:5003` - Task 1):**
     - Experiment **`Task1-Universal-Restoration`** khol kar local verification run **`milestone_1_verification_run`** dikhao.

#### BOLNA HAI (Roman Urdu):
> *"Scientific reproducibility aur experiment tracking ke liye humne MLflow SQLite databases maintain kiye hain.*
>
> *Port 5000 par Task 4 cGAN tracking hai, jahan `Task4_cGAN_Face_To_Sketch` experiment ke 17 logged runs archived hain with complete generator aur discriminator loss curves.*
>
> *Port 5001 par Task 2 tracking server hai, jisme 50 Optuna classifier search trials, 3 classifier final training runs, aur 9 specialist training runs logged hain.*
>
> *Port 5002 par Task 3 Soft MoE ka final 2-Phase training run `Soft_MoE_2Phase_Final` archived hai. Note karein ke Optuna Bayesian hyperparameter trials report aur EXPERIMENT_LOG mein fully documented hain.*
>
> *Aur Port 5003 par Task 1 ka Milestone 1 local verification run `milestone_1_verification_run` logged hai, jabke full 50-epoch GPU training curves humari report aur experiment log mein documented hain."*

---

### Segment 8: Honest Limitations, Adversarial Analysis & Conclusion
- **TIMESTAMP:** 5:30 – 6:05 (Duration: 0:35)
- **RUNNING TOTAL:** 6:05 (Well within 5:30 – 6:30 target)

#### SCREEN ACTION:
1. Browser mein waapis frontend studio UI par aao (`http://localhost:3000`).
2. Task 4 Face-to-Sketch tab par synthesized sketch ko zoom karke texture dikhao.
3. Final conclusion slide ya studio overview par zoom out karo.

#### BOLNA HAI (Roman Urdu):
> *"Aakhir mein, hum Task 4 ki ek honest empirical limitation highlight karte hain jo humne detailed report mein bhi analyze ki hai:*
>
> *Training ke dauran PatchGAN discriminator bohot early saturate ho gaya aur Epoch 3 tak iski accuracy 91 se 94% tak pohch gayi. Is discriminator dominance ki wajah se generator adversarial updates sirf 0 se 5% batches par trigger ho saki, aur model primarily L1 pixel reconstruction par rely karne laga. Yahi wajah hai ke synthesized sketches soft aur shaded pencil tones mein aate hain bajaye razor-sharp vectorized line-art ke—achieving 16.24 dB overall PSNR.*
>
> *Summary yeh hai ke is project ne autoencoding bottlenecks, hard specialist routing, differentiable soft mixture-of-experts, conditional GAN synthesis, aur lightweight torchless ONNX CPU deployment ko successfully demonstrate kiya.*
>
> *Thank you very much!"*

---

## Pre-Record Checklist (Recording se Pehle Yeh Check Karein)

- [ ] **1. Ports & Services:**
  - WSL Backend running on port `8000` (`http://localhost:8000/api/health` returns status "ok" and 8 models true).
  - Frontend running on port `3000` (`http://localhost:3000` shows green "Backend: Online (CPU)" pill).
  - MLflow Task 4 running on port `5000` (`sqlite:///mlflow.db`).
  - MLflow Task 2 running on port `5001` (`sqlite:///mlruns_task2.db`).
  - MLflow Task 3 running on port `5002` (`sqlite:///mlruns_task3.db`).
  - MLflow Task 1 running on port `5003` (`sqlite:///mlruns/mlflow.db`).
- [ ] **2. Browser Tabs Order (Pre-Opened):**
  - Tab 1: `http://localhost:8000/api/health`
  - Tab 2: `http://localhost:3000` (Main React Studio UI)
  - Tab 3: `http://localhost:5000` (MLflow Task 4)
  - Tab 4: `http://localhost:5001` (MLflow Task 2)
  - Tab 5: `http://localhost:5002` (MLflow Task 3)
  - Tab 6: `http://localhost:5003` (MLflow Task 1)
- [ ] **3. MLflow UI UI Setting:**
  - Har MLflow tab mein top-left par switch karke **"Model training"** toggle click karke pehle se set rakhein (taake page load hote hi runs dikhein).
- [ ] **4. Browser Zoom & Display:**
  - Browser zoom 100% ya 90% par set karein taake saare cards aur side-by-side images cleanly visible hon.
- [ ] **5. Clean OS Environment:**
  - Windows notifications band karein (Focus Assist on).
  - Clear Downloads folder taake "Download Result" click karne par nayi download file top par clearly dikhe.
- [ ] **6. Mic & Audio Test:**
  - Audio input level test karein (no echo/background hiss).
  - Speech pace normal rakhein taake 5:30 se 6:15 min ke darmiyan video naturally complete ho.

---

## Agar Galti Ho Jaye / Troubleshooting Guide

| Issue | Kya Karna Hai (Quick Fix) |
|---|---|
| **App load nahi ho rahi ya Backend Offline dikha raha hai** | WSL terminal mein check karein `sudo docker compose restart` ya local backend check karein `curl http://localhost:8000/api/health`. Frontend par refresh icon click karein. |
| **MLflow UI khali dikh rahi hai (No runs)** | MLflow ke top-left corner par **"Model training"** radio button / toggle par click karein. GenAI default view mein standard metric runs hide ho jaate hain. |
| **Webcam popup kaam na kare** | Browser permissions mein Camera allow karein, ya dropzone mein **"Choose Image"** se portrait image upload kar dein ya gallery se portrait select karein. |
| **Download button click hone par file na dikhe** | Browser ki download bar check karein ya Windows File Explorer mein `Downloads` folder khol kar dikhayein. |
| **Bolte waqt fumbles ya time zyada hone lage** | Har segment ke timestamps guide ke mutabiq hain. Agar kisi segment mein 5 second zyada lag jayein toh agle segment mein thoda fast transition karein taake total time 6:30 se exceed na ho. |

---

## Summary of Verified Metrics (Reference Only - Do Not Invent New Numbers)

- **Classification Accuracy (Task 2):** $99.95\%$ (3,667 / 3,669 test images).
- **Universal Autoencoder (Task 1):** $22.51\text{ dB}$ PSNR overall test average ($29.20\text{ dB}$ clean, $26.20\text{ dB}$ blur, $23.96\text{ dB}$ S&P, $14.98\text{ dB}$ occlusion).
- **Hard Routing Gain (Task 2):** $32.85\text{ dB}$ PSNR ($+10.34\text{ dB}$ overall leap over Universal baseline; $100.00\text{ dB}$ clean bypass).
- **Soft MoE Gains (Task 3):** $+1.13\text{ dB}$ on S&P ($28.69\text{ dB}$ vs $27.56\text{ dB}$), $+1.27\text{ dB}$ on Blur ($28.18\text{ dB}$ vs $26.91\text{ dB}$), $+0.71\text{ dB}$ on Occlusion ($22.40\text{ dB}$ vs $21.69\text{ dB}$); Overall Test Mean $40.53\text{ dB}$ vs $32.85\text{ dB}$.
- **Face-to-Sketch cGAN (Task 4):** $16.24\text{ dB}$ PSNR overall ($18.06\text{ dB}$ Style 1, $12.89\text{ dB}$ Style 2, $19.60\text{ dB}$ Style 3 [$N=46$ low-sample]).
- **PatchGAN Discriminator Saturation (Task 4):** $91$--$94\%$ accuracy by Epoch 3; generator updates throttled to $0$--$5\%$ of batches.
- **Inference Latencies (Raw ONNX CPU Benchmarks):** Universal $60.56\text{ ms}$ (live API $35$--$55\text{ ms}$), Hard Routing $81.60\text{ ms}$, Soft MoE $211.40\text{ ms}$, Face-to-Sketch $36.79\text{ ms}$.
