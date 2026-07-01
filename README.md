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

## Features

*   **Polished AI Security Dashboard**: A recruiter-friendly Streamlit interface with modern cards, dynamic risk badges, model status indicators, upload previews, result summaries, and responsible AI disclaimers.

*   **Real vs. Fake Face Detection**: Uses an EfficientNetB0-based TensorFlow/Keras model to classify uploaded face media as real or potentially synthetic/manipulated.

*   **Face Isolation Pipeline**: Applies OpenCV Haar Cascade face detection to crop the primary face region with padding before inference. If face detection fails, the app safely falls back to the full image or frame.

*   **Video Frame Aggregation**: Samples up to 16 frames from uploaded videos and combines frame-level predictions into a more stable video-level deepfake risk score.

*   **Passive Liveness & Spoof-Risk Estimation**: Uses lightweight visual heuristics such as motion, sharpness, texture, exposure, and color richness to estimate possible replay, static photo, or low-quality spoof attempts.

*   **Grad-CAM Explainability**: Generates model attention heatmaps to show which regions influenced the prediction, improving transparency without claiming forensic proof.

*   **Manual Review Recommendation Engine**: Converts prediction confidence into Low Risk, Suspicious, or High Risk categories with clear human-review guidance.

*   **Deployment Ready**: Structured for GitHub documentation, Hugging Face Spaces deployment, and local Streamlit execution.

---

## 4-Layer Architecture

To keep the project clean, explainable, and recruiter-friendly, SatyaLens follows a practical **4-layer AI security workflow**:

```text
SatyaLens/
├── app.py                              # Layer 1-4: Streamlit UI, preprocessing, inference, risk dashboard
├── satyalens_v6_efficientnetb0.keras   # Trained TensorFlow/Keras deepfake detection model
├── satyalens_v6_metrics.json           # Stored evaluation metrics
├── label_map.json                      # Class mapping: Real = 0, Fake = 1
├── model_card.md                       # Responsible AI notes and model limitations
├── MODEL_DOWNLOAD.md                   # Model file setup instructions if excluded from GitHub
├── DEPLOYMENT_GUIDE.md                 # Hugging Face Spaces deployment instructions
├── requirements.txt                    # Python dependencies config
├── README.md                           # Project documentation
├── gradcam_1.jpg                       # Sample Grad-CAM explainability output
├── gradcam_2.jpg
├── gradcam_3.jpg
├── gradcam_4.jpg
└── LICENSE                             # MIT License
```

### Layer 1: Media Intake Dashboard

Handles uploaded images and videos through Streamlit.

Supported formats:

```text
Images: JPG, JPEG, PNG, WEBP
Videos: MP4, AVI, MOV, MKV, WEBM
```

### Layer 2: Face Preprocessing Engine

Processes uploaded media before inference:

* Reads image or video frames.
* Detects the largest face using OpenCV Haar Cascade.
* Crops the face with margin padding.
* Resizes input to 224x224 pixels.
* Falls back to the full frame if face detection fails.

### Layer 3: Deepfake Inference Engine

Runs the EfficientNetB0-based model and returns a fake probability score.

```text
0 = Real
1 = Fake
```

### Layer 4: Identity-Risk Intelligence Layer

Generates human-readable outputs:

* Fake probability
* Risk category
* Manual-review recommendation
* Grad-CAM heatmap
* Video aggregation metrics
* Passive liveness score
* Spoof-risk estimate

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

The app expects the trained model file in the root project directory:

```text
satyalens_v6_efficientnetb0.keras
```

If the model file is not available in the GitHub repository due to file-size limits, follow the instructions in:

```text
MODEL_DOWNLOAD.md
```

### 3. Launch the AI Security Dashboard

```bash
streamlit run app.py
```

The application will open locally at:

```text
http://localhost:8501
```

## Streamlit Run Application

```text
Add live demo link here:
https://huggingface.co/spaces/YOUR_USERNAME/SatyaLens
```

---

## Model Performance

The current trained model is:

```text
satyalens_v6_efficientnetb0.keras
```

Evaluation metrics are stored in:

```text
satyalens_v6_metrics.json
```

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

*   **Objective**: Classify face media as real or potentially fake/synthetic.
*   **Input Size**: 224x224 RGB image.
*   **Output**: Single sigmoid probability representing fake likelihood.
*   **Classes**: Real = 0, Fake = 1.
*   **Use Case**: Deepfake fraud research, synthetic identity detection, and AI security portfolio demonstration.

### Face Isolation & Preprocessing

*   **Detection Method**: OpenCV Haar Cascade face detector.
*   **Processing Logic**: Detect largest face, apply padding margin, crop, and resize.
*   **Fallback Handling**: If no face is detected, the full frame is used instead of stopping execution.
*   **Purpose**: Keeps inference focused on the face area while maintaining robustness.

### Grad-CAM Explainability

*   **Objective**: Show which image regions influenced the model’s prediction.
*   **Method**: Dynamically identifies the last convolutional layer and generates a heatmap.
*   **Output**: Overlay visualization on the processed face image.
*   **Limitation**: Grad-CAM is interpretability support, not forensic evidence.

### Video Risk Aggregation

*   **Objective**: Improve video analysis stability by avoiding single-frame dependency.
*   **Frame Sampling**: Up to 16 frames are sampled uniformly from uploaded videos.
*   **Metrics Used**: Mean fake probability, median fake probability, top-3 fake probability, and high-risk frame ratio.

Video-level deepfake risk is calculated as:

```text
Deepfake Video Risk =
0.50 × Mean Fake Probability
+ 0.25 × Median Fake Probability
+ 0.15 × Top-3 Fake Probability
+ 0.10 × High-Risk Frame Ratio
```

### Passive Liveness & Spoof-Risk Heuristics

*   **Objective**: Estimate whether uploaded video media appears live-like or spoof-like.
*   **Signals Used**: Motion, sharpness, texture, exposure, and color richness.
*   **Spoof-Risk Logic**: Lower liveness score means higher spoof risk.
*   **Limitation**: This is a heuristic method, not certified biometric liveness validation.

Liveness score:

```text
Liveness Score =
0.25 × Motion
+ 0.25 × Sharpness
+ 0.20 × Texture
+ 0.15 × Exposure
+ 0.15 × Color Richness
```

Spoof risk:

```text
Spoof Risk = 100 - Liveness Score
```

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

SatyaLens is designed to deploy on **Hugging Face Spaces** using the Streamlit SDK.

Upload these files to your Space:

```text
app.py
requirements.txt
README.md
model_card.md
label_map.json
satyalens_v6_metrics.json
satyalens_v6_efficientnetb0.keras
```

The model file must be placed in the same directory as `app.py`.

Recommended hosting structure:

```text
GitHub        = Source code, documentation, metrics, screenshots
Hugging Face  = Live demo and model file
Kaggle        = Training notebook and evaluation proof
```

---

## Intended Use

SatyaLens is intended for:

*   AI security learning
*   Deepfake detection research
*   Synthetic identity fraud awareness
*   KYC fraud research
*   Remote hiring fraud research
*   Explainable AI experimentation
*   Computer vision portfolio demonstration
*   Manual-review workflow demonstration

---

## Limitations

SatyaLens has important limitations:

*   It is a research prototype, not a production security product.
*   The model may fail on unseen deepfake generation methods.
*   Public datasets may not represent real-world scam media.
*   Lighting, blur, compression, camera quality, pose, and face visibility can affect predictions.
*   Dataset bias can affect model behavior across different demographics.
*   Passive liveness is heuristic, not certified biometric validation.
*   False positives and false negatives are possible.
*   Outputs should support human review, not replace human judgment.

---

## Ethical Use

SatyaLens should only be used for defensive, educational, research, and fraud-prevention purposes.

It should not be used for:

*   Surveillance
*   Harassment
*   Unauthorized identity analysis
*   Automated hiring decisions
*   Automated KYC rejection
*   Legal evidence validation
*   Any decision affecting a person’s rights or opportunities without human review

---

## Future Improvements

*   Replace Haar Cascade with MTCNN, RetinaFace, or MediaPipe Face Detection.
*   Add audio deepfake detection.
*   Add FastAPI backend for API-based inference.
*   Add downloadable PDF risk report.
*   Add robustness testing for blur, compression, low light, crop, and noise.
*   Add confidence calibration.
*   Add dataset bias evaluation.
*   Add Docker support.
*   Export model to ONNX or TFLite.
*   Add investigator/reviewer dashboard mode.

---

## Developer Credits

Developed with passion by **Arnav Raj (Cybroarnv)**:

*   **GitHub**: [@yocybroarnv](https://github.com/yocybroarnv)
*   **LinkedIn**: [Arnav Raj](https://www.linkedin.com/in/arnav-raj-professional)
*   **Organization**: Independent AI Security Research Project

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
