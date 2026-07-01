# SatyaLens: Deepfake Detection & Identity-Risk Estimator  
### AI Security Project for Synthetic Face Media Review

---

[![Live Demo](https://img.shields.io/badge/Live%20Demo-SatyaLensAI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://satyalensai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-27338e?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## Live Demo

The deployed app is available here:

**SatyaLens AI:** https://satyalensai.streamlit.app/

---

## Recruiter Brief

**SatyaLens** is an AI security-focused computer vision project built to analyze face images and videos for possible deepfake or synthetic media signals.

I built this project to explore a real-world security problem: fake remote interviews, synthetic identity abuse, KYC fraud attempts, impersonation scams, and replay-based media attacks.

The project is not just a model notebook. It is a working review dashboard that combines model inference, video-frame analysis, passive liveness signals, metadata extraction, explainability, performance optimization, and responsible AI messaging into one usable application.

> SatyaLens is a research-grade manual-review support tool. It is not a production KYC, hiring, legal, surveillance, or biometric verification system.

---

## What This Project Demonstrates

| Area | What SatyaLens Demonstrates |
|---|---|
| AI / ML | Deepfake classification using TensorFlow and EfficientNetB0 |
| Computer Vision | Face preprocessing, image handling, and video-frame sampling |
| AI Security | Synthetic identity and deepfake fraud risk framing |
| Explainable AI | Grad-CAM heatmaps for model transparency |
| Product Thinking | Recruiter Mode and Investigator Mode for different reviewers |
| Performance | Model caching, file-hash reuse, batch video inference, and on-demand Grad-CAM |
| Metadata Analysis | File audit snapshot, image details, EXIF checks, and video properties |
| Deployment | Streamlit live deployment with GitHub-ready structure |
| Responsible AI | Manual-review framing, limitations, and ethical-use boundaries |

---

## Why I Built It

Deepfakes are no longer only an entertainment or social media issue. They are increasingly connected to fraud, fake interviews, identity misuse, digital impersonation, and trust failures in online systems.

A simple real/fake classifier is not enough for this kind of problem. A reviewer needs more context:

- How confident is the model?
- Is the uploaded file an image or video?
- What does the video-level risk look like across frames?
- Does the media show weak liveness signals?
- What part of the face influenced the prediction?
- What metadata is available from the uploaded file?
- What should a human reviewer do next?

SatyaLens was built around these questions.

---

## Core Features

### Deepfake Detection

SatyaLens uses an EfficientNetB0-based TensorFlow/Keras model to estimate whether uploaded face media appears real or potentially synthetic.

The model output is converted into an easy-to-understand risk level.

| Score Range | Risk Level | Review Guidance |
|---:|---|---|
| 0–29% | Low Risk | Continue normal review, but do not treat as final proof |
| 30–66% | Suspicious | Manual review and additional verification recommended |
| 67–100% | High Risk | Strong manual verification recommended |

---

### Video-Level Risk Aggregation

For video uploads, SatyaLens does not rely on a single frame.

It samples frames from the video, runs batch inference, and calculates:

- Mean fake probability
- Median fake probability
- Top-risk frame average
- High-risk frame ratio
- Overall identity-risk estimate

This makes the output more useful than a single-frame prediction.

---

### Passive Liveness Signals

SatyaLens estimates basic passive liveness indicators from visual patterns such as:

- Motion
- Sharpness
- Texture
- Exposure
- Color richness

These signals help identify possible static image attacks, replayed videos, or low-quality spoof media.

This is intentionally described as a heuristic signal, not certified biometric liveness.

---

### Grad-CAM Explainability

The app includes on-demand Grad-CAM explainability.

Instead of forcing heavy explainability computation on every upload, SatyaLens generates Grad-CAM only when the reviewer requests it.

This keeps the app faster while still giving transparency when needed.

---

### Media Metadata & Audit Snapshot

SatyaLens extracts useful media-level information such as:

- File name
- File size
- File type
- SHA-256 hash preview
- Image resolution
- Image format
- EXIF presence
- Video duration
- FPS
- Frame count
- Codec details where available

This helps reviewers understand the uploaded media beyond the model prediction.

---

## User Experience

SatyaLens is designed as a Cyberpunk Sentinel-style AI security dashboard.

It includes two review modes:

### Recruiter Mode

A simplified mode for quick project review.

It focuses on:

- Upload preview
- Main prediction result
- Risk level
- Recommended action
- Short explanation
- Responsible AI note

### Investigator Mode

A deeper mode for technical inspection.

It includes:

- Metadata details
- Video-frame analysis
- Passive liveness breakdown
- Grad-CAM explainability
- Model notes
- Future roadmap

---

## Model Performance

The current model file is:

```text
satyalens_v6_efficientnetb0.keras
```

Evaluation metrics are loaded from:

```text
satyalens_v6_metrics.json
```

| Metric | Score |
|---|---:|
| Accuracy | 75.7% |
| Precision | 79.4% |
| Recall | 69.3% |
| ROC-AUC | 84.5% |

These scores are based on the available evaluation dataset. Real-world performance may vary because of compression, lighting, pose, camera quality, dataset bias, and unseen deepfake generation methods.

---

## Runtime Optimizations

SatyaLens was optimized to make the review experience faster and smoother.

### Cached Model Loading

The model is loaded once using Streamlit caching so it does not reload on every app rerun.

### Model Warm-Up

A dummy prediction is executed after model loading to reduce first-upload cold-start delay.

### File Hash Caching

Uploaded files are hashed using SHA-256. If the same file is analyzed again, SatyaLens can reuse cached results instead of recomputing everything.

### On-Demand Grad-CAM

Grad-CAM is generated only when requested, instead of during every initial prediction.

### Batched Video Inference

Sampled video frames are processed together in one batch instead of one-by-one prediction loops.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend App | Streamlit |
| ML Framework | TensorFlow / Keras |
| Model Backbone | EfficientNetB0 |
| Computer Vision | OpenCV, Pillow |
| Data Handling | NumPy, Pandas |
| Explainability | Grad-CAM |
| Metadata Handling | Pillow, OpenCV, file hashing |
| Deployment Target | Streamlit Cloud |
| Language | Python |

---

## System Flow

```text
User Upload
   |
   |-- Image or Video
   |
   |-- Metadata Extraction
   |
   |-- Face Detection and Preprocessing
   |
   |-- EfficientNetB0 Model Inference
   |
   |-- Risk Classification
   |
   |-- Video Aggregation / Liveness Signals
   |
   |-- Grad-CAM Explanation on Demand
   |
   |-- Manual Review Recommendation
```

---

## Project Structure

```text
SatyaLens/
├── app.py                              # Main Streamlit dashboard
├── satyalens_v6_efficientnetb0.keras   # Trained model file
├── satyalens_v6_metrics.json           # Evaluation metrics
├── label_map.json                      # Class mapping
├── model_card.md                       # Responsible AI notes
├── MODEL_DOWNLOAD.md                   # Model file instructions
├── DEPLOYMENT_GUIDE.md                 # Deployment guide
├── requirements.txt                    # Dependencies
├── README.md                           # Project documentation
└── LICENSE                             # MIT License
```

## Responsible Use

SatyaLens is designed for research, learning, and manual review support.

It should not be used for:

- Automated hiring decisions
- Automated KYC rejection
- Legal verification
- Biometric gatekeeping
- Surveillance
- Harassment
- Any decision affecting a person without human review

The project intentionally avoids overclaiming because deepfake detection can fail under real-world conditions.

---

## What I Learned

While building SatyaLens, I worked on:

- Turning an ML model into a usable product demo
- Improving inference speed with caching and batch prediction
- Handling image and video uploads in Streamlit
- Designing AI outputs for human reviewers
- Adding metadata context to model predictions
- Communicating model limitations responsibly
- Building a project that connects AI, cybersecurity, and real-world fraud risk

---

## Developer Credits

Developed by **Arnav Raj (Cybroarnv)**

- **GitHub:** [@yocybroarnv](https://github.com/yocybroarnv)
- **LinkedIn:** [Arnav Raj](https://www.linkedin.com/in/arnav-raj-professional)
- **Project Type:** Independent AI Security Research Project

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
