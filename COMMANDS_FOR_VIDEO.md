# Demonstration Video Execution & Command Reference

**File:** `COMMANDS_FOR_VIDEO.md`  
**Purpose:** Quick-reference, copy-paste command sheet for recording the assignment demonstration video.

---

## 1. WSL Terminal (Start & Manage Docker Services)

### Step 1: Navigate to repository root in WSL
```bash
cd /mnt/c/Users/usmanbari/Desktop/Gen-Ai-A1
```

### Step 2: Start Docker Compose in detached mode
```bash
sudo docker compose up -d
```

### Step 3: Check running containers status
```bash
sudo docker compose ps
```

### Step 4: Test backend health endpoint in terminal
```bash
curl http://localhost:8000/api/health
```

### Clean up after recording (Stop Docker)
```bash
sudo docker compose down
```

---

## 2. Browser Tabs to Pre-Open (In Order)

- **Tab 1 (Backend Health Check):**
```text
http://localhost:8000/api/health
```

- **Tab 2 (Main React Studio Application):**
```text
http://localhost:3000
```

- **Tab 3 (MLflow Task 4 - Face to Sketch):**
```text
http://localhost:5000
```

- **Tab 4 (MLflow Task 2 - Hard Routing & Specialists):**
```text
http://localhost:5001
```

- **Tab 5 (MLflow Task 3 - Soft Mixture of Experts):**
```text
http://localhost:5002
```

- **Tab 6 (MLflow Task 1 - Universal Restoration):**
```text
http://localhost:5003
```

---

## 3. PowerShell Windows for MLflow Tracking Servers

### Window 1 — Task 4 cGAN Tracking (Port 5000)
```powershell
cd C:\Users\usmanbari\Desktop\Gen-Ai-A1
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Window 2 — Task 2 Classifier & Specialists Tracking (Port 5001)
```powershell
cd C:\Users\usmanbari\Desktop\Gen-Ai-A1
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlruns_task2.db --port 5001
```

### Window 3 — Task 3 Soft MoE Tracking (Port 5002)
```powershell
cd C:\Users\usmanbari\Desktop\Gen-Ai-A1
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlruns_task3.db --port 5002
```

### Window 4 — Task 1 Universal Verification Tracking (Port 5003)
```powershell
cd C:\Users\usmanbari\Desktop\Gen-Ai-A1
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5003
```

---

## 4. Pre-Recording Startup Sequence (Order of Operations)

### Step-by-step startup order before hitting Record:
1. Open **WSL Terminal** -> Run `sudo docker compose up -d`.
2. Open **4 PowerShell Windows** -> Run MLflow commands for Ports 5000, 5001, 5002, and 5003.
3. Open **Web Browser** with 6 tabs in exact order:
   - `http://localhost:8000/api/health`
   - `http://localhost:3000`
   - `http://localhost:5000` (Click top-left **"Model training"** toggle)
   - `http://localhost:5001` (Click top-left **"Model training"** toggle)
   - `http://localhost:5002` (Click top-left **"Model training"** toggle)
   - `http://localhost:5003` (Click top-left **"Model training"** toggle)
4. Set browser zoom to **100%** (or **90%** so dual panels and diagnostics fit cleanly without vertical scrolling).
5. Clean out `Downloads/` folder so new exported sketches appear at the top.

---

## 5. UI Click Path & Exact Button Names per Task (`frontend/src/App.jsx`)

### Universal Restoration Workspace (Task 1)
1. Tab: **"Universal Restoration"**
2. Point cursor to: **"Choose Image"** dropzone area.
3. Sample click: **"Clean Pet Cat"** (from Preset Sample Gallery).
4. Corruption selector: **"Salt & Pepper Noise"**
5. Severity selector: **"Medium Severity"**
6. Button click: **"Apply Runtime Corruption"**
7. Button click: **"Run Deep Restoration"**

### Hard-Routed Specialist Workspace (Task 2)
1. Tab: **"Hard-Routed Restoration"**
2. Sample click: **"Clean Pet Dog"**
3. Corruption selector: **"Rectangular Occlusion"**
4. Severity selector: **"High Severity"**
5. Button click: **"Apply Runtime Corruption"**
6. Button click: **"Run Deep Restoration"**
7. Diagnostic verification: Look for `RECTANGULAR_OCCLUSION` and `specialist_rectangular_occlusion`.

### Soft Mixture-of-Experts Workspace (Task 3)
1. Tab: **"Soft Mixture-of-Experts Restoration"**
2. Sample click: **"Clean Pet Cat"**
3. Corruption selector: **"Gaussian Blur"**
4. Severity selector: **"Medium Severity"**
5. Button click: **"Apply Runtime Corruption"**
6. Button click: **"Run Deep Restoration"**
7. Diagnostic verification: Look for dominant `GAUSSIAN_BLUR` and 4 continuous horizontal weight bars ($w_0, w_1, w_2, w_3$).

### Face-to-Sketch Generator Workspace (Task 4)
1. Tab: **"Face-to-Sketch Generator"**
2. Button click: **"Webcam"** -> Modal opens -> Button click: **"Capture Photo"** *(or upload a portrait / select "Clean Face Portrait")*.
3. Style 1 click: **"Style 1"** (`Pencil / Classic`) -> Button click: **"Synthesize Facial Sketch"**
4. Style 2 click: **"Style 2"** (`Sketch / Artistic`) -> Button click: **"Synthesize Facial Sketch"**
5. Style 3 click: **"Style 3"** (`Caricature / Graphic`) -> Button click: **"Synthesize Facial Sketch"**
6. Mandatory Spec Download: Button click: **"Download Result"** -> Show saved PNG file in Downloads.

---

## 6. Recovery & Troubleshooting Commands

### If Docker backend or frontend is unresponsive in WSL
```bash
sudo docker compose restart
```

### If Docker needs a full rebuild in WSL
```bash
sudo docker compose down && sudo docker compose up --build -d
```

### If an MLflow port is already in use on Windows (PowerShell)
#### Find the process holding port (e.g. 5000, 5001, 5002, 5003):
```powershell
Get-NetTCPConnection -LocalPort 5000,5001,5002,5003 -ErrorAction SilentlyContinue | Select-Object LocalPort, OwningProcess
```

#### Kill the process holding a specific port by Process ID (replace <PID>):
```powershell
Stop-Process -Id <PID> -Force
```

#### One-liner to kill any process on a specific port (e.g. port 5000):
```powershell
Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```
