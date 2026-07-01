# Deployment Guide

This guide details how to deploy **SatyaLens** to both **Hugging Face Spaces** and **Streamlit Community Cloud**.

---

## 🌐 Streamlit Community Cloud Deployment (Recommended for Quick Demos)

Streamlit Community Cloud is a free hosting platform for Streamlit applications.

> [!IMPORTANT]  
> **Python Version Constraint**: You must configure the application environment to use **Python 3.11** or **Python 3.12**. Do not use Python 3.13 or Python 3.14, as TensorFlow and MediaPipe do not currently have pre-compiled wheels for those versions on Linux, which will cause dependency installation failures.

### Steps to Deploy and Configure:
1. Log in to [share.streamlit.io](https://share.streamlit.io) and click **New App**.
2. Select your GitHub repository (`yocybroarnv/SatyaLens`), branch (`main`), and set the main file path to `app.py`.
3. Before deploying, click **Advanced settings** (or access it from your App Settings in the dashboard).
4. Locate the **Python version** dropdown and select **3.11** or **3.12**.
5. Click **Save** and deploy the app.
6. If your app is already deployed and failing with dependency errors:
   - Go to your Streamlit Cloud Dashboard.
   - Click the three dots next to your app, and select **Settings**.
   - Under **Advanced settings**, change the Python version to **3.11** or **3.12**.
   - Save and reboot the application.

---

## 🤗 Hugging Face Spaces Deployment (Streamlit SDK)

Deploying to Hugging Face Spaces enables live hosting and integration.

### Required Files for Deployment
Push the following files to your Hugging Face Space repository:
- `app.py` (The updated web application code)
- `requirements.txt` (Python dependencies)
- `satyalens_v6_efficientnetb0.keras` (Pretrained model weights)
- `satyalens_v6_metrics.json` (Model performance statistics)
- `label_map.json` (Real vs. Fake mapping)
- `model_card.md` (Ethical considerations & benchmarks)
- `README.md` (Project overview)

### Step-by-Step Deployment

1. **Create Space on Hugging Face**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Choose a unique name (e.g., `satyalens-deepfake-scanner`).
   - Select **Streamlit** as the SDK.
   - Choose the **CPU Basic** hardware tier (Free).

2. **Handle Large Model Files (Git LFS)**:
   The model file `satyalens_v6_efficientnetb0.keras` is around 29 MB. To push it properly, track it using Git LFS:
   ```bash
   # Initialize Git LFS in your local checkout
   git lfs install
   git lfs track "*.keras"
   
   # Add your files
   git add .gitattributes
   git add satyalens_v6_efficientnetb0.keras app.py requirements.txt satyalens_v6_metrics.json label_map.json
   git commit -m "feat: deploy SatyaLens Streamlit app with weights"
   ```

3. **Configure Git Remote & Push**:
   - Link your local repository to the Hugging Face Space git remote (copy the URL from Hugging Face space dashboard):
     ```bash
     git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
     git push -u hf main --force
     ```
   - Hugging Face will automatically read `requirements.txt`, install dependencies, cache the model weights, and spin up the Streamlit interface.

---

## Resume & Portfolio Description
Use this short summary when adding the project to your resume, portfolio website, or LinkedIn:

> **SatyaLens: Deepfake Fraud Detection & Identity-Risk Estimator**
> Developed a research-focused AI security computer vision application in Streamlit using TensorFlow (EfficientNetB0) and OpenCV. The platform isolates faces, detects synthetic manipulations (deepfakes) with a 75.7% accuracy model, computes video risk scores through multi-frame timeline aggregation, and estimates heuristic passive liveness components (motion, sharpness, texture moiré) to detect screen replay and print spoofing attacks. Employs Grad-CAM gradient overlays to provide explainable model focus maps for human reviewer support.