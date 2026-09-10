# VoiceShield - Real-Time AI Audio Deepfake & Forensic Analysis Platform

VoiceShield is an advanced forensic audio inspection system designed to detect AI voice deepfakes, synthetic audio manipulations, and spoofing attacks using multi-model ensemble score fusion.

---

## 🌟 Key Features

* **Multi-Model Ensemble Fusion:** Combines RawNet2 (1D raw waveform analysis) and ResNet18 (2D Mel-spectrogram spectral analysis) for high-confidence spoof detection.
* **Temporal Forensic Timeline:** Pinpoints exact time intervals where synthetic or manipulated audio occurs rather than returning a static binary verdict.
* **Interactive Seeking:** Allows forensic analysts to click on high-threat time frames to inspect and listen to specific audio segments instantly.
* **Forensic PDF Audit Export:** Generates client-side PDF inspection reports containing evaluation metadata, individual model threat scores, and frame-level analytics.
* **Cybersecurity Operations Dashboard:** Dark-mode responsive UI designed for real-time monitoring and offline file scanning.

---

## 🏗️ System Architecture & Tech Stack

* **Frontend:** React, TypeScript, Tailwind CSS, Lucide Icons, jsPDF
* **Backend:** FastAPI (Python), PyTorch, Uvicorn
* **Audio DSP:** Librosa, Torchaudio, SciPy
* **ML Architectures:** RawNet2, ResNet18 (Mel-Spectrogram Classifier)

---

## 🚀 Quickstart Guide

### 1. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python -m uvicorn backend.app.main:app --reload --port 8000



2. Frontend Setup
Bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev

📄 License
Developed for Smart India Hackathon (SIH) 2026.


---

### Step-by-Step Commands to Push `README.md` to GitHub

After saving the file, run these commands in your terminal:

1. **Stage the new file:**
   ```bash
   git add README.md
