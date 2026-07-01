# SatyaLens v7: Deepfake Fraud Detection & Identity-Risk Estimator

SatyaLens v7 is an AI security-focused computer vision prototype designed to scan face media for synthetic manipulation, calculate passive liveness/spoof-risk scores, and provide Grad-CAM explainability maps. It acts as a decision-support tool for recruiters, AI security reviewers, and KYC researchers.

---

## 🚀 Key Upgrades & Features

- **Polished AI Security Dashboard**: A complete UI overhaul with soft cards, dynamic risk badges, and interactive use-case indicators.
- **Aggregated Video Risk Scoring**: Samples up to 16 frames uniformly and computes video-level threat indicators using a multi-factor formula.
- **Heuristic Passive Liveness**: Estimates spoof risks (printed photos, screen replays) across five components: motion, sharpness, texture (moiré grids), exposure, and color richness.
- **Grad-CAM Attention Mapping**: Visually displays heatmaps showing the facial regions the model focused on during prediction.
- **Status Indicator & Metrics Panel**: The sidebar checks model state and dynamically parses benchmark evaluations from `satyalens_v6_metrics.json`.
- **Face isolation cropping**: OpenCV Haar Cascades crop the primary face area, falling back automatically to full frames on failure.

---

## 📊 Benchmark Model Metrics
The model `satyalens_v6_efficientnetb0.keras` performs with the following validation metrics:
*   **Accuracy**: 75.67%
*   **Precision**: 79.39%
*   **Recall**: 69.33%
*   **ROC-AUC**: 84.51%

---

## ⚙️ Setup & Local Execution

### 1. Requirements Installation
Ensure Python 3.8+ is installed, then install requirements:
```bash
pip install -r requirements.txt
```

### 2. Model Download
The app requires the model weights `satyalens_v6_efficientnetb0.keras` in the root folder. If it is missing, refer to the setup steps in [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) to download or pull via Git LFS.

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## ⚖️ Limitations & Ethical Disclaimer
1. **Research Prototype**: SatyaLens v7 is a research-grade demo and should not be used as a production KYC gates, automatic hiring rejector, or legal validation tool.
2. **Heuristic Liveness**: The passive liveness analyzer utilizes heuristics (not certified biometric validation). High-quality digital displays or print bypasses may evade detection.
3. **No Automatic Decisions**: The application provides risk signals to support manual review workflows, not automated approvals or rejections.