# SatyaLens: Deepfake Fraud Detection & Identity-Risk Estimator
### *AI Security Dashboard for Synthetic Face Media, Video Risk Aggregation & Passive Liveness Intelligence*

---

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)

**SatyaLens** is an AI security-focused computer vision prototype designed to detect possible deepfake or synthetic face media, estimate identity-risk signals, analyze video-frame behavior, and generate Grad-CAM explainability maps for model transparency.

Built around the rising threat of **deepfake fraud, synthetic identity abuse, fake remote interviews, replay attacks, and digital impersonation**, SatyaLens acts as a decision-support dashboard for AI security reviewers, recruiters, KYC researchers, cybersecurity learners, and fraud-risk teams.

> [!NOTE]  
> **Research Prototype:** This application is a functional AI security demonstration. It is not a production KYC system, certified biometric liveness solution, legal validation tool, surveillance platform, or automated hiring decision engine. SatyaLens provides risk signals for manual review support only.

---

## 🚀 Advanced Capabilities

1. **Dual Dashboard Modes**:
   - **Recruiter Mode (Simplified)**: Renders high-level classification verdicts, clean risk badges, and recommended reviewer procedures.
   - **Investigator Mode (Expert)**: Exposes neural network certainty indices, confidence calibration warnings, robustness Stress-Testing charts, demographic bias matrices, and model export formats.
2. **Modular Face Detector**: Uses **MediaPipe Face Detection** for high-precision boundary boxes, falling back automatically to **OpenCV Haar Cascades** on system environments without MediaPipe.
3. **Acoustic Deepfake Scanner**: Supports uploading audio tracks (`.wav`, `.mp3`, `.m4a`, `.ogg`) and extracts spectral flatness, pitch jitter stability, high-frequency codec artifacts, and noise profiles.
4. **FastAPI Backend Gateway**: Integrates a complete FastAPI server alongside the Streamlit app to enable programmatic REST API calls for automated scans.
5. **Downloadable PDF Scan Reports**: Generates professional, in-memory PDF risk reports containing metadata, primary assessments, component tables, and ethical disclaimers.
6. **Distortion Robustness Stress-Test**: Distorts input images (blur, noise, exposure, crop, and JPEG compression) to graph model output stability under varying signal conditions.
7. **ONNX & TFLite Model Exporter**: Converts the core Keras model directly into TFLite or ONNX formats inside the expert workspace.
8. **Multi-Service Docker Containerization**: Configures the entire environment with a Dockerfile and docker-compose script for quick, multi-port deployments.

---

## 📂 Modular File Architecture

```text
SatyaLens/
├── app.py                              # Streamlit frontend dashboard (Recruiter & Investigator modes)
├── api.py                              # FastAPI backend gateway service
├── utils/
│   ├── face_detection.py               # MediaPipe Face cropping with Haar Cascade fallback
│   ├── audio_analysis.py               # Acoustic spectral and FFT analysis
│   ├── robustness.py                   # Stress-testing distortion injectors
│   ├── pdf_report.py                   # Report PDF builder using FPDF2
│   └── model_export.py                 # ONNX/TFLite converter functions
├── satyalens_v6_efficientnetb0.keras   # Trained TensorFlow/Keras deepfake detection model
├── satyalens_v6_metrics.json           # Stored evaluation metrics
├── label_map.json                      # Class mapping: Real = 0, Fake = 1
├── model_card.md                       # Responsible AI notes and model limitations
├── MODEL_DOWNLOAD.md                   # Model file setup instructions if excluded from GitHub
├── DEPLOYMENT_GUIDE.md                 # Hugging Face Spaces deployment instructions
├── requirements.txt                    # Python dependencies config
├── Dockerfile                          # Containerization instructions
├── docker-compose.yml                  # Streamlit + FastAPI compose script
└── LICENSE                             # MIT License
```

---

## Quick Start

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/yocybroarnv/SatyaLens.git
cd SatyaLens

# Install dependencies
pip install -r requirements.txt
```

### 2. Add the Trained Model
The app expects the trained model file `satyalens_v6_efficientnetb0.keras` in the root directory. If the model file is not available in the checkout, follow the download instructions in [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md).

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The application will open locally at `http://localhost:8501`.

### 4. Launch the FastAPI Gateway
```bash
uvicorn api:app --reload
```
The backend API service will run at `http://127.0.0.1:8000`. You can access the interactive Swagger documentation at `http://127.0.0.1:8000/docs` to test endpoints.

---

## 🐳 Docker Deployment
To build and run both the Streamlit frontend and FastAPI backend in isolated containers:
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

## Model Performance

The current trained model is `satyalens_v6_efficientnetb0.keras`. Evaluation metrics are stored in `satyalens_v6_metrics.json`:

| Metric | Score |
|---|---:|
| Accuracy | 75.67% |
| Precision | 79.39% |
| Recall | 69.33% |
| F1 Score | 74.02% |
| ROC-AUC | 84.51% |

> These results are based on the available evaluation dataset. Real-world performance may vary due to lighting, compression, face angle, camera quality, dataset bias, and unseen deepfake generation methods.

---

## Deep-Dive: Machine Learning Pipeline

### EfficientNetB0 Transfer Learning
- **Objective**: Classify face media as real or potentially fake/synthetic.
- **Input Size**: 224x224 RGB image.
- **Output**: Single sigmoid probability representing fake likelihood.
- **Classes**: Real = 0, Fake = 1.
- **Use Case**: Deepfake fraud research, synthetic identity detection, and AI security portfolio demonstration.

### Face Isolation & Preprocessing
- **Detection Method**: MediaPipe Face Detection with OpenCV Haar Cascade fallback.
- **Processing Logic**: Detect largest face, apply padding margin, crop, and resize.
- **Fallback Handling**: If no face is detected, the full frame is used instead of stopping execution.
- **Purpose**: Keeps inference focused on the face area while maintaining robustness.

### Grad-CAM Explainability
- **Objective**: Show which image regions influenced the model’s prediction.
- **Method**: Dynamically identifies the last convolutional layer and generates a heatmap.
- **Output**: Overlay visualization on the processed face image.
- **Limitation**: Grad-CAM is interpretability support, not forensic evidence.

### Video Risk Aggregation
- **Objective**: Improve video analysis stability by avoiding single-frame dependency.
- **Frame Sampling**: Up to 16 frames are sampled uniformly from uploaded videos.
- **Metrics Used**: Mean fake probability, median fake probability, top-3 fake probability, and high-risk frame ratio.

Video-level deepfake risk is calculated as:
$$\text{Deepfake Video Risk} = 0.50 \times \text{Mean} + 0.25 \times \text{Median} + 0.15 \times \text{Top-3} + 0.10 \times \text{High-Risk Ratio}$$

### Passive Liveness & Spoof-Risk Heuristics
- **Objective**: Estimate whether uploaded video media appears live-like or spoof-like.
- **Signals Used**: Motion, sharpness, texture, exposure, and color richness.
- **Spoof-Risk Logic**: Lower liveness score means higher spoof risk.
- **Limitation**: This is a heuristic method, not certified biometric liveness validation.

$$\text{Liveness Score} = 0.25 \times \text{Motion} + 0.25 \times \text{Sharpness} + 0.20 \times \text{Texture} + 0.15 \times \text{Exposure} + 0.15 \times \text{Color Richness}$$
$$\text{Spoof Risk} = 100 - \text{Liveness Score}$$

---

## Risk Classification Logic

SatyaLens does not approve, reject, verify, or blacklist anyone. It converts model confidence into a simple manual-review risk category.

| Score Range | Risk Level | Recommended Action |
|---:|---|---|
| Below 30% | Low Risk | Continue normal review, but do not treat as final proof |
| 30% to 66% | Suspicious | Manual review and additional verification recommended |
| Above 66% | High Risk | Strong manual verification recommended |

---

## Deployment Notes

SatyaLens is designed to deploy on **Hugging Face Spaces** using the Streamlit SDK. Upload these files to your Space:
`app.py`, `requirements.txt`, `README.md`, `model_card.md`, `label_map.json`, `satyalens_v6_metrics.json`, and `satyalens_v6_efficientnetb0.keras`. Refer to [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for details.

---

## Intended Use & Limitations

### Intended Use
SatyaLens is intended for:
- AI security learning
- Deepfake detection research
- Synthetic identity fraud awareness
- KYC fraud research
- Remote hiring fraud research
- Explainable AI experimentation
- Computer vision portfolio demonstration
- Manual-review workflow demonstration

### Limitations
SatyaLens has important limitations:
- It is a research prototype, not a production security product.
- The model may fail on unseen deepfake generation methods.
- Public datasets may not represent real-world scam media.
- Lighting, blur, compression, camera quality, pose, and face visibility can affect predictions.
- Dataset bias can affect model behavior across different demographics.
- Passive liveness is heuristic, not certified biometric validation.
- False positives and false negatives are possible.
- Outputs should support human review, not replace human judgment.

### Ethical Use
SatyaLens should only be used for defensive, educational, research, and fraud-prevention purposes. It should not be used for:
- Surveillance
- Harassment
- Clark/demographic tracking
- Automated hiring decisions
- Automated KYC rejection
- Legal evidence validation
- Any decision affecting a person’s rights or opportunities without human review

---

## Developer Credits

Developed with passion by **Arnav Raj (Cybroarnv)**:
*   **GitHub**: [@yocybroarnv](https://github.com/yocybroarnv)
*   **LinkedIn**: [Arnav Raj](https://www.linkedin.com/in/arnav-raj-professional)
*   **Organization**: Independent AI Security Research Project

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
