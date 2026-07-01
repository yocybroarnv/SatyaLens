# SatyaLens: Advanced AI Security Hub

SatyaLens is an AI security-focused computer vision and signal processing suite designed to scan face media and audio signals for synthetic manipulation (deepfakes), estimate passive liveness/spoof risk, and provide explainable model diagnostics. It functions as an audit workspace for security reviewers, KYC research teams, and cybersecurity students.

---

## 🚀 Advanced Capabilities

1. **Dual Dashboard Modes**:
   - **Recruiter Mode (Simplified)**: Renders high-level classification verdicts, clean risk badges, and recommended reviewer procedures.
   - **Investigator Mode (Expert)**: Exposes neural network certainty indices, confidence calibration warnings, robustness Stress-Testing charts, demographic bias matrices, and model export formats.
2. **Modular Face Detector**: Uses **MediaPipe Face Detection** for high-precision boundary boxes, falling back automatically to **OpenCV Haar Cascades** on system environments without MediaPipe.
3. **Acoustic Deepfake Scanner**: Supports uploading audio tracks (`.wav`, `.mp3`, `.m4a`, `.ogg`) and extracts spectral flatness, pitch jitter stability, high-frequency codec artifacts, and noise profiles.
4. **FastAPI Backend Gateway**: Integrates a complete FastAPI server alongside the Streamlit app to enable programmatic REST API calls for automated scans.
5. **Downloadable PDF Scan Reports**: Generates professional, in-memory PDF risk reports containing metadata, primary assessments, component tables, and ethical disclaimers.
6. **Distortion Robustness stress test**: Distorts input images (blur, noise, exposure, crop, and JPEG compression) to graph model output stability under varying signal conditions.
7. **ONNX & TFLite Model Exporter**: Converts the core Keras model directly into TFLite or ONNX formats inside the expert workspace.
8. **Multi-Service Docker Containerization**: Configures the entire environment with a Dockerfile and docker-compose script for quick, multi-port deployments.

---

## 📂 Modular File Architecture
- [app.py](app.py): Streamlit frontend dashboard (handles recruiter and expert modes, uploads, and charts).
- [api.py](api.py): FastAPI backend routes for image, video, and audio predictions.
- `utils/`:
  - [face_detection.py](utils/face_detection.py): MediaPipe face cropping with Haar Cascade fallback.
  - [audio_analysis.py](utils/audio_analysis.py): Acoustic spectral and FFT analysis.
  - [robustness.py](utils/robustness.py): Stress-testing distortion injectors.
  - [pdf_report.py](utils/pdf_report.py): Report PDF builder using FPDF2.
  - [model_export.py](utils/model_export.py): ONNX/TFLite converter functions.

---

## ⚙️ Running Locally

### 1. Requirements Setup
```bash
pip install -r requirements.txt
```

### 2. Streamlit Dashboard
Launch the frontend dashboard:
```bash
streamlit run app.py
```

### 3. FastAPI Gateway
Launch the backend server:
```bash
uvicorn api:app --reload
```
Access the Swagger interactive documentation at `http://127.0.0.1:8000/docs` to test endpoints.

---

## 🐳 Docker Deployment
To build and run both Streamlit (frontend) and FastAPI (backend) in isolated containers:
```bash
# Build and start services
docker-compose up --build

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```
- Streamlit will be exposed on: `http://localhost:8501`
- FastAPI will be exposed on: `http://localhost:8000`

---

## ⚖️ Limitations & Ethical Disclaimer
1. **Decision Support Only**: SatyaLens is a research prototype. It estimates threat classes and provides risk markers to aid manual verification. It should not be used as an automated gating check or for background rejections.
2. **Heuristic Liveness**: The passive liveness and acoustic checkers use visual/signal heuristics. High-quality print, screen-replay, or voice clones can evade these checks. Escalation to human reviewers is mandatory for suspicious results.