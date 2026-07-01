import os
import cv2
import json
import tempfile
import sys
import numpy as np
import streamlit as st

# 1. CHECK CORE DEPENDENCIES (For Streamlit Cloud Compatibility)
required_libs_installed = True
missing_libs = []

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    required_libs_installed = False
    missing_libs.append("TensorFlow")

try:
    import scipy
except ImportError:
    required_libs_installed = False
    missing_libs.append("SciPy")

if not required_libs_installed:
    st.set_page_config(
        page_title="SatyaLens: Compatibility Error",
        page_icon="!",
        layout="centered"
    )
    
    st.markdown("""
    <style>
        .stApp { background-color: #070B16; color: #F8FAFC; }
        .error-card {
            background-color: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(239, 68, 68, 0.25);
            padding: 2rem;
            border-radius: 12px;
            margin-top: 2rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .error-title {
            color: #EF4444;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="error-card">', unsafe_allow_html=True)
    st.markdown('<div class="error-title">Streamlit Cloud Compatibility Error</div>', unsafe_allow_html=True)
    st.markdown(
        f"The application successfully compiled its dependencies, but the required machine learning packages "
        f"({', '.join(missing_libs)}) could not be loaded in this Python environment."
    )
    st.markdown(f"Current Python Runtime version: Python {sys.version.split()[0]}")
    st.markdown(
        """
        Why is this happening?
        By default, Streamlit Community Cloud deployed your application on Python 3.14. 
        TensorFlow does not support Python 3.14 on Linux yet, meaning it cannot be installed.
        
        How to Resolve This (2-Step Fix):
        1. Go to your Streamlit Cloud Dashboard at share.streamlit.io.
        2. Click the three dots next to your app, choose Settings, and under Advanced settings, set the Python version to 3.12 or 3.11.
        
        Once you click Save, the application environment will automatically rebuild with the selected Python version, install TensorFlow and SciPy, and launch successfully!
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# If packages are present, we can safely import everything else
import matplotlib.pyplot as plt
from PIL import Image

# Import modular utilities
from utils.face_detection import FaceDetector

# ---------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------
IMG_SIZE = 224
MODEL_PATH = "satyalens_v6_efficientnetb0.keras"
METRICS_PATH = "satyalens_v6_metrics.json"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Initialize modular face detector
face_detector = FaceDetector()

# ---------------------------------------------------------
# CYBERPUNK SENTINEL THEME INJECTION
# ---------------------------------------------------------
def inject_cyberpunk_css():
    css = """
    <style>
        /* Base page styling */
        .stApp {
            background-color: #070B16 !important;
            color: #F8FAFC !important;
            background-image: radial-gradient(circle at 50% 50%, #0F172A 0%, #070B16 100%) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #090E1A !important;
            border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
        }
        
        [data-testid="stSidebar"] * {
            color: #CBD5E1 !important;
        }
        
        /* Container max-width and padding */
        .block-container {
            max-width: 1100px !important;
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Glassmorphism Card */
        .glass-card {
            background: rgba(15, 23, 42, 0.78) !important;
            border: 1px solid rgba(56, 189, 248, 0.22) !important;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            margin-bottom: 1.5rem;
            transition: border-color 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(56, 189, 248, 0.4) !important;
            box-shadow: 0 8px 32px 0 rgba(56, 189, 248, 0.08);
        }
        
        .glass-card-secondary {
            background: rgba(30, 41, 59, 0.72) !important;
            border: 1px solid rgba(56, 189, 248, 0.12) !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        /* Headers & Typography */
        .cyber-title {
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(90deg, #38BDF8 0%, #8B5CF6 50%, #EC4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
            letter-spacing: -0.03em;
        }
        
        .cyber-subtitle {
            font-size: 1.15rem;
            color: #38BDF8 !important;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 1.5rem;
        }
        
        /* Use-Case Chips */
        .chip-container {
            margin-bottom: 1.5rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .chip {
            display: inline-block;
            background-color: rgba(30, 41, 59, 0.6);
            color: #38BDF8 !important;
            border: 1px solid rgba(56, 189, 248, 0.25);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        
        /* Risk Badges */
        .risk-badge {
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.8rem;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 1px solid transparent;
        }
        
        .risk-badge-low {
            background-color: rgba(34, 197, 94, 0.1);
            color: #22C55E !important;
            border-color: rgba(34, 197, 94, 0.25);
        }
        
        .risk-badge-suspicious {
            background-color: rgba(245, 158, 11, 0.1);
            color: #F59E0B !important;
            border-color: rgba(245, 158, 11, 0.25);
        }
        
        .risk-badge-high {
            background-color: rgba(239, 68, 68, 0.1);
            color: #EF4444 !important;
            border-color: rgba(239, 68, 68, 0.25);
        }
        
        /* Custom Progress Bar */
        .progress-container {
            margin: 0.85rem 0;
        }
        .progress-bar-bg {
            background-color: rgba(30, 41, 59, 0.5);
            border-radius: 9999px;
            height: 10px;
            width: 100%;
            overflow: hidden;
            border: 1px solid rgba(56, 189, 248, 0.15);
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.4s ease;
        }
        
        /* Custom Tables */
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.88rem;
        }
        .metrics-table th {
            background-color: rgba(30, 41, 59, 0.5);
            color: #F8FAFC !important;
            font-weight: 600;
            text-align: left;
            padding: 0.75rem;
            border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        }
        .metrics-table td {
            padding: 0.75rem;
            border-bottom: 1px solid rgba(56, 189, 248, 0.1);
            color: #CBD5E1 !important;
        }
        
        /* Recommendation Alert Boxes */
        .action-box {
            padding: 1.25rem;
            border-radius: 8px;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border: 1px solid;
        }
        .action-low {
            background-color: rgba(34, 197, 94, 0.05);
            color: #22C55E !important;
            border-color: rgba(34, 197, 94, 0.25);
        }
        .action-suspicious {
            background-color: rgba(245, 158, 11, 0.05);
            color: #F59E0B !important;
            border-color: rgba(245, 158, 11, 0.25);
        }
        .action-high {
            background-color: rgba(239, 68, 68, 0.05);
            color: #EF4444 !important;
            border-color: rgba(239, 68, 68, 0.25);
        }
        
        /* Sidebar Metrics Grid */
        .sidebar-metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .sidebar-metric-card {
            background-color: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(56, 189, 248, 0.12);
            border-radius: 6px;
            padding: 0.5rem;
            text-align: center;
        }
        .sidebar-metric-val {
            font-size: 1rem;
            font-weight: 700;
            color: #38BDF8 !important;
        }
        .sidebar-metric-lbl {
            font-size: 0.7rem;
            color: #94A3B8 !important;
        }
        
        /* File Uploader styling override */
        [data-testid="stFileUploader"] {
            background-color: rgba(30, 41, 59, 0.4) !important;
            border: 1px dashed rgba(56, 189, 248, 0.3) !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }
        [data-testid="stFileUploader"] * {
            color: #CBD5E1 !important;
        }
        
        /* Widget Overrides */
        .stButton>button {
            background-color: rgba(37, 99, 235, 0.2) !important;
            color: #38BDF8 !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            background-color: rgba(56, 189, 248, 0.3) !important;
            border-color: #38BDF8 !important;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
        }
        
        /* Tabs override */
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(15, 23, 42, 0.4) !important;
            border-radius: 8px !important;
            border-bottom: 1px solid rgba(56, 189, 248, 0.15) !important;
            padding: 0.2rem !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94A3B8 !important;
            font-weight: 600 !important;
            background-color: transparent !important;
            border: none !important;
            transition: all 0.2s ease !important;
        }
        .stTabs [aria-selected="true"] {
            color: #38BDF8 !important;
            border-bottom: 2px solid #38BDF8 !important;
        }
        
        /* Code blocks */
        code {
            background-color: rgba(30, 41, 59, 0.5) !important;
            color: #38BDF8 !important;
            padding: 0.15rem 0.35rem !important;
            border-radius: 4px;
            font-size: 0.85em;
        }
        
        /* Global Text overrides */
        .stMarkdown div, .stMarkdown p, .stMarkdown span {
            color: #CBD5E1;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------------------------------------------------
# CORE MODEL LOADING & METRICS
# ---------------------------------------------------------
@st.cache_resource
def load_deepfake_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights file {MODEL_PATH} is missing.")
    return keras.models.load_model(MODEL_PATH)

model_exists = os.path.exists(MODEL_PATH)

metrics_data = None
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
    except Exception:
        pass

# ---------------------------------------------------------
# SIDEBAR RENDER
# ---------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown('<h2 style="margin-top: 0; margin-bottom: 0.1rem;">SatyaLens</h2>', unsafe_allow_html=True)
    st.sidebar.markdown('<span style="font-size:0.8rem; opacity:0.7; color:#38BDF8 !important; text-transform:uppercase; letter-spacing:0.05em;">AI Security Command Dashboard</span>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="margin-top:0.35rem;"><span class="risk-badge risk-badge-suspicious" style="padding: 0.1rem 0.4rem; font-size:0.65rem;">Research Prototype</span></div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Workspace Mode Switcher
    st.sidebar.markdown("### Workspace")
    dashboard_mode = st.sidebar.selectbox(
        "Select Interface Style:",
        ["Recruiter Mode", "Investigator Mode"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # System Status Tracker
    st.sidebar.markdown("### System Status")
    if model_exists:
        st.sidebar.markdown('<span class="risk-badge risk-badge-low" style="padding: 0.15rem 0.5rem; font-size:0.75rem;">Model Loaded</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"Filename: `{MODEL_PATH}`")
    else:
        st.sidebar.markdown('<span class="risk-badge risk-badge-high" style="padding: 0.15rem 0.5rem; font-size:0.75rem;">Model Missing</span>', unsafe_allow_html=True)
        st.sidebar.markdown(
            "Model file not found. Expected: satyalens_v6_efficientnetb0.keras. "
            "Place it in the same folder as app.py.", 
            unsafe_allow_html=True
        )
    
    # Model Performance metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Benchmark Metrics")
    if metrics_data:
        html_metrics = f"""
        <div class="sidebar-metrics-grid">
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-val">{metrics_data.get('accuracy', 0.0):.1%}</div>
                <div class="sidebar-metric-lbl">Accuracy</div>
            </div>
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-val">{metrics_data.get('roc_auc', 0.0):.1%}</div>
                <div class="sidebar-metric-lbl">ROC-AUC</div>
            </div>
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-val">{metrics_data.get('precision', 0.0):.1%}</div>
                <div class="sidebar-metric-lbl">Precision</div>
            </div>
            <div class="sidebar-metric-card">
                <div class="sidebar-metric-val">{metrics_data.get('recall', 0.0):.1%}</div>
                <div class="sidebar-metric-lbl">Recall</div>
            </div>
        </div>
        """
        st.sidebar.markdown(html_metrics, unsafe_allow_html=True)
    else:
        st.sidebar.warning("Metrics file not found")
        
    # Risk Guidelines
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Risk Guidelines")
    st.sidebar.markdown(
        """
        - **0 - 29%**: <span style="color:#22C55E; font-weight:bold;">Low Risk</span>
        - **30 - 66%**: <span style="color:#F59E0B; font-weight:bold;">Suspicious</span>
        - **67 - 100%**: <span style="color:#EF4444; font-weight:bold;">High Risk</span>
        """,
        unsafe_allow_html=True
    )
    
    # Supported formats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Supported Formats")
    st.sidebar.markdown(
        """
        - Images: JPG, JPEG, PNG, WEBP
        - Videos: MP4, AVI, MOV, MKV, WEBM
        """
    )
    
    # Responsible AI Use
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Responsible AI")
    st.sidebar.caption(
        "SatyaLens supports manual review workflows. "
        "It is not an automated gatekeeper or final decision system."
    )
    
    # Developer Card
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Developer")
    st.sidebar.markdown(
        """
        <div class="glass-card-secondary" style="margin-bottom:0; padding:0.75rem; border-color:rgba(139, 92, 246, 0.25);">
            <strong style="font-size:0.85rem; color:#F8FAFC;">Arnav Raj (Cybroarnv)</strong>
            <div style="font-size:0.75rem; margin-top:0.25rem;">
                <a href="https://github.com/yocybroarnv" target="_blank" style="color:#38BDF8; text-decoration:none; margin-right:0.5rem;">GitHub</a>
                <a href="https://www.linkedin.com/in/arnav-raj-professional" target="_blank" style="color:#38BDF8; text-decoration:none;">LinkedIn</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    return dashboard_mode

# Apply selected styling and render sidebar
inject_cyberpunk_css()
dashboard_mode = render_sidebar()

# ---------------------------------------------------------
# HERO SECTION RENDER
# ---------------------------------------------------------
def render_hero():
    st.markdown('<h1 class="cyber-title">SatyaLens</h1>', unsafe_allow_html=True)
    st.markdown('<div class="cyber-subtitle">Deepfake Detection & Identity-Risk Estimator</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="chip-container">
            <span class="chip">Deepfake Fraud</span>
            <span class="chip">Synthetic Identity Risk</span>
            <span class="chip">Remote Hiring Review</span>
            <span class="chip">KYC Fraud Research</span>
            <span class="chip">Explainable AI</span>
            <span class="chip">Video Risk Aggregation</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Description
    st.markdown(
        "An AI security dashboard for analyzing synthetic face media, video-level deepfake risk, "
        "passive liveness signals, and Grad-CAM explainability."
    )
    
    # Disclaimer
    st.markdown(
        """
        <div class="glass-card" style="border-left: 4px solid var(--warning-color); background-color: rgba(245, 158, 11, 0.05); padding: 1rem 1.25rem;">
            <strong style="color: var(--warning-color); font-size: 0.9rem;">Research Prototype Disclaimer:</strong>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; line-height: 1.4; color: var(--muted-text);">
                This is a research prototype for manual review support. It is not a certified biometric, legal, KYC, hiring, or surveillance system.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

render_hero()

# ---------------------------------------------------------
# CAPABILITY CARDS RENDER
# ---------------------------------------------------------
def render_capability_cards():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="glass-card-secondary" style="height: 100%; min-height: 140px; border-color: rgba(56, 189, 248, 0.25);">
                <div style="font-size:0.7rem; color:#38BDF8; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Active</div>
                <strong style="font-size:0.9rem; color:#F8FAFC; display:block;">Deepfake Detection</strong>
                <p style="font-size:0.75rem; color:#94A3B8; margin-top:0.25rem; line-height:1.3;">
                    Classifies uploaded face media as real or potentially synthetic.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card-secondary" style="height: 100%; min-height: 140px; border-color: rgba(139, 92, 246, 0.25);">
                <div style="font-size:0.7rem; color:#8B5CF6; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Active</div>
                <strong style="font-size:0.9rem; color:#F8FAFC; display:block;">Video Aggregation</strong>
                <p style="font-size:0.75rem; color:#94A3B8; margin-top:0.25rem; line-height:1.3;">
                    Samples multiple frames and combines frame-level predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="glass-card-secondary" style="height: 100%; min-height: 140px; border-color: rgba(236, 72, 153, 0.25);">
                <div style="font-size:0.7rem; color:#EC4899; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Active</div>
                <strong style="font-size:0.9rem; color:#F8FAFC; display:block;">Passive Liveness</strong>
                <p style="font-size:0.75rem; color:#94A3B8; margin-top:0.25rem; line-height:1.3;">
                    Estimates spoof-risk using motion, sharpness, texture, and color richness.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="glass-card-secondary" style="height: 100%; min-height: 140px; border-color: rgba(56, 189, 248, 0.25);">
                <div style="font-size:0.7rem; color:#38BDF8; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;">Active</div>
                <strong style="font-size:0.9rem; color:#F8FAFC; display:block;">Grad-CAM Explain</strong>
                <p style="font-size:0.75rem; color:#94A3B8; margin-top:0.25rem; line-height:1.3;">
                    Shows visual regions that influenced model prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

render_capability_cards()

# Load Keras Model
model = None
if model_exists:
    try:
        model = load_deepfake_model()
    except Exception as e:
        st.error(f"Failed to load core neural network: {e}")
        st.stop()
else:
    st.markdown(
        """
        <div class="glass-card" style="border-left: 4px solid var(--danger-color); background-color: rgba(220, 38, 38, 0.05);">
            <strong style="color: var(--danger-color);">Core Model Weights Missing</strong>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">
                The core deep learning model weights file <code>satyalens_v6_efficientnetb0.keras</code> was not found.
            </p>
            <p style="margin: 0; font-size: 0.9rem;">
                Please follow the download setup steps in MODEL_DOWNLOAD.md to continue.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# ---------------------------------------------------------
# IMAGE INFERENCE HELPER
# ---------------------------------------------------------
def predict_rgb(rgb):
    cropped, detected, method = face_detector.detect_and_crop(rgb)
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    arr = resized.astype("float32")
    arr = np.expand_dims(arr, axis=0)
    fake_prob = float(model.predict(arr, verbose=0)[0][0])
    return fake_prob, cropped, detected, method

# ---------------------------------------------------------
# LIVENESS & TIMELINE SAMPLING
# ---------------------------------------------------------
def calculate_passive_liveness_heuristics(frames):
    if len(frames) == 0:
        return {"liveness_score": 0.0, "spoof_risk": 100.0, "liveness_label": "Unknown"}

    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    sharpness = float(np.clip(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]) / 180.0, 0, 1))

    texture = float(np.clip(
        np.mean([np.mean(cv2.Canny(g, 80, 160) > 0) for g in grays]) / 0.12,
        0,
        1
    ))

    brightness = float(np.mean([np.mean(g) for g in grays]))
    exposure = float(1 - np.clip(abs(brightness - 127.5) / 127.5, 0, 1))

    if len(grays) > 1:
        small = [cv2.resize(g, (96, 96)) for g in grays]
        diffs = [np.mean(cv2.absdiff(a, b)) for a, b in zip(small[:-1], small[1:])]
        motion = float(np.clip(np.mean(diffs) / 18.0, 0, 1))
    else:
        motion = 0.0

    color = float(np.clip(np.mean([np.mean(np.std(f, axis=(0, 1))) for f in frames]) / 55.0, 0, 1))

    liveness_score = (
        0.25 * motion +
        0.25 * sharpness +
        0.20 * texture +
        0.15 * exposure +
        0.15 * color
    ) * 100

    liveness_score = float(np.clip(liveness_score, 0, 100))
    spoof_risk = 100.0 - liveness_score

    if liveness_score >= 65:
        label = "Live-like"
    elif liveness_score >= 40:
        label = "Uncertain"
    else:
        label = "Possible spoof or replay media"

    return {
        "liveness_score": round(liveness_score, 2),
        "spoof_risk": round(spoof_risk, 2),
        "liveness_label": label,
        "motion_score": round(motion * 100, 2),
        "sharpness_score": round(sharpness * 100, 2),
        "texture_score": round(texture * 100, 2),
        "exposure_score": round(exposure * 100, 2),
        "color_score": round(color * 100, 2)
    }

def read_video_frames(video_path, max_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        indices = np.linspace(0, max(total_frames - 1, 0), max_frames).astype(int)
        for idx in sorted(set(indices.tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames

def predict_video_aggregate(path):
    frames = read_video_frames(path, 16)
    if len(frames) == 0:
        return None, []

    probs = []
    detected_faces_count = 0
    for f in frames:
        prob, _, detected, _ = predict_rgb(f)
        probs.append(prob)
        if detected:
            detected_faces_count += 1

    probs = np.array(probs)
    mean_prob = float(np.mean(probs))
    median_prob = float(np.median(probs))
    top3_prob = float(np.mean(np.sort(probs)[-min(3, len(probs)):]))
    high_risk_ratio = float(np.mean(probs >= 0.66))

    deepfake_risk = (
        0.50 * mean_prob +
        0.25 * median_prob +
        0.15 * top3_prob +
        0.10 * high_risk_ratio
    )

    liveness = calculate_passive_liveness_heuristics(frames)
    overall_risk = 0.70 * deepfake_risk + 0.30 * (liveness["spoof_risk"] / 100.0)

    risk_lbl, action = estimate_risk_category(overall_risk)

    return {
        "frames_used": len(frames),
        "faces_detected": detected_faces_count,
        "mean_fake_probability": round(mean_prob, 4),
        "median_fake_probability": round(median_prob, 4),
        "top3_fake_probability": round(top3_prob, 4),
        "high_risk_frame_ratio": round(high_risk_ratio, 4),
        "deepfake_video_risk": round(deepfake_risk, 4),
        "overall_identity_risk": round(overall_risk, 4),
        "risk": risk_lbl,
        "action": action,
        "liveness": liveness
    }, frames

# ---------------------------------------------------------
# GRAD-CAM EXPLAINABILITY heatmaps
# ---------------------------------------------------------
def get_nested_base_model(full_model):
    for layer in full_model.layers:
        if isinstance(layer, keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    return None

def get_last_conv_layer_name(model_obj):
    for layer in reversed(model_obj.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None

def make_gradcam_overlay(rgb, alpha=0.45):
    try:
        base = get_nested_base_model(model)
        if base is None:
            return None

        last_layer_name = get_last_conv_layer_name(base)
        if last_layer_name is None:
            return None

        rgb_crop, _, _ = face_detector.detect_and_crop(rgb)
        resized = cv2.resize(rgb_crop, (IMG_SIZE, IMG_SIZE))
        arr = resized.astype("float32")
        arr = np.expand_dims(arr, axis=0)

        last_conv = base.get_layer(last_layer_name)

        grad_model = keras.Model(
            inputs=base.input,
            outputs=[last_conv.output, base.output]
        )

        gap_layer = model.get_layer("gap")
        dropout_layer = model.get_layer("dropout")
        output_layer = model.get_layer("deepfake_probability")

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_model(arr, training=False)
            x = gap_layer(base_outputs)
            x = dropout_layer(x, training=False)
            preds = output_layer(x)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_resized = cv2.resize(heatmap_uint8, (IMG_SIZE, IMG_SIZE))

        cmap = plt.get_cmap("jet")
        colored = cmap(heatmap_resized)[:, :, :3]
        colored = np.uint8(255 * colored)

        original = cv2.resize(rgb_crop, (IMG_SIZE, IMG_SIZE)).astype(np.uint8)
        overlay = np.uint8(original * (1 - alpha) + colored * alpha)

        return overlay
    except Exception:
        return None

# ---------------------------------------------------------
# UPLOAD SECTION
# ---------------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<h3 style='margin-top: 0;'>Media Analysis Intake</h3>", unsafe_allow_html=True)
st.markdown("<span class='muted-text'>Upload a face image or short video for AI-assisted risk analysis.</span>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Media File Upload",
    type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv", "webm"],
    label_visibility="collapsed"
)
st.markdown("<span style='font-size:0.8rem; color:#94A3B8;'>Supported formats: JPG, JPEG, PNG, WEBP, MP4, AVI, MOV, MKV, WEBM</span>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# EXECUTE SCAN / PREVIEW
# ---------------------------------------------------------
if not uploaded_file:
    # Awaiting Media Upload Empty State
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 3rem 1.5rem;">
            <h3 style="margin-top: 0; color: #F8FAFC;">Awaiting Media Upload</h3>
            <p class="muted-text" style="max-width: 500px; margin: 0.5rem auto 0 auto; line-height: 1.45;">
                Upload a face image or short video in the intake panel above to generate risk signals, 
                explainability maps, and manual-review recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    file_suffix = uploaded_file.name.lower().split(".")[-1]
    is_image = file_suffix in ["jpg", "jpeg", "png", "webp"]

    if is_image:
        # ---------------------------------------------------------
        # IMAGE MODE
        # ---------------------------------------------------------
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            raw_rgb = np.array(pil_image)
        except Exception as e:
            st.error(f"Could not load image: {e}")
            st.stop()

        with st.spinner("Analyzing media and generating risk signals..."):
            try:
                fake_prob, cropped_face, face_found, det_method = predict_rgb(raw_rgb)
                risk_lvl, recommendation = estimate_risk_category(fake_prob)
            except Exception as e:
                st.error(f"Inference execution failed: {e}")
                st.stop()

        pred_lbl = "Fake" if fake_prob >= 0.5 else "Real"

        if dashboard_mode == "Recruiter Mode":
            st.markdown("---")
            st.markdown("### Assessment Results")
            
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
                st.image(
                    cropped_face, 
                    caption=f"Isolated Face ({det_method})" if face_found else "Raw Image (Face Cropping Failed)", 
                    use_container_width=True
                )
                if not face_found:
                    st.caption("No face isolated. Inference ran on full frame.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                # Custom HTML Metric Cards
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Assessment Summary")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(
                        f"""
                        <div class="glass-card-secondary" style="text-align: center;">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Classification</div>
                            <div style="font-size:1.4rem; font-weight:800; margin:0.35rem 0; color:#F8FAFC;">{pred_lbl}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_m2:
                    st.markdown(
                        f"""
                        <div class="glass-card-secondary" style="text-align: center; padding: 1rem 0.5rem;">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600; margin-bottom:0.4rem;">Risk Level</div>
                            {get_risk_badge_html(risk_lvl)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_m3:
                    st.markdown(
                        f"""
                        <div class="glass-card-secondary" style="text-align: center;">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Probability</div>
                            <div style="font-size:1.4rem; font-weight:800; margin:0.35rem 0; color:#38BDF8;">{fake_prob:.2%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                pb_color = "#22C55E" if fake_prob < 0.30 else ("#F59E0B" if fake_prob < 0.66 else "#EF4444")
                st.markdown(get_progress_bar_html(fake_prob * 100, pb_color), unsafe_allow_html=True)
                st.markdown(get_recommendation_box_html(risk_lvl), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.warning("Image-only liveness is limited. Video provides stronger liveness signals.")

            # Explainability
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Explainability Map")
            st.markdown("Highlighted areas show regions that influenced the model prediction. Grad-CAM is interpretability support, not forensic proof.")
            with st.spinner("Analyzing explainability focus..."):
                overlay_img = make_gradcam_overlay(raw_rgb)
            if overlay_img is not None:
                st.image(overlay_img, caption="Grad-CAM Overlay", use_container_width=True)
            else:
                st.caption("Grad-CAM could not be generated, but prediction results are still available.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # What this means card
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="margin-top:0;">What This Means</h4>
                    <p style="font-size:0.9rem; color:#CBD5E1; line-height:1.5; margin:0;">
                        The EfficientNetB0 classification engine evaluated the isolated face and calculated a 
                        <strong>{fake_prob:.2%}</strong> likelihood of synthetic generation. 
                        Based on this score, the submission is categorized as <strong>{risk_lvl}</strong>. 
                        Reviewers should follow the recommended protocol above.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            # Investigator Mode Tabs
            st.markdown("---")
            tab_ov, tab_gc, tab_mn = st.tabs([
                "Overview", 
                "Explainability", 
                "Model Notes"
            ])

            with tab_ov:
                col1, col2 = st.columns([1, 1.2])
                with col1:
                    st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
                    st.image(cropped_face, caption=f"Cropped Face ({det_method})", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### Diagnostic Summary")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(
                            f"""
                            <div class="glass-card-secondary" style="text-align: center;">
                                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Verdict</div>
                                <div style="font-size:1.4rem; font-weight:800; margin:0.35rem 0; color:#F8FAFC;">{pred_lbl}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_m2:
                        st.markdown(
                            f"""
                            <div class="glass-card-secondary" style="text-align: center; padding: 1rem 0.5rem;">
                                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600; margin-bottom:0.4rem;">Risk Level</div>
                                {get_risk_badge_html(risk_lvl)}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_m3:
                        st.markdown(
                            f"""
                            <div class="glass-card-secondary" style="text-align: center;">
                                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Neural Score</div>
                                <div style="font-size:1.4rem; font-weight:800; margin:0.35rem 0; color:#38BDF8;">{fake_prob:.4%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    pb_color = "#22C55E" if fake_prob < 0.30 else ("#F59E0B" if fake_prob < 0.66 else "#EF4444")
                    st.markdown(get_progress_bar_html(fake_prob * 100, pb_color), unsafe_allow_html=True)
                    st.markdown(get_recommendation_box_html(risk_lvl), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.warning("Image-only liveness is limited. Video provides stronger liveness signals.")

            with tab_gc:
                st.markdown("### Neural Activation Focus")
                st.markdown(
                    "Grad-CAM overlay highlights the final convolutional block layers to map visual "
                    "attention indicators during predictions. Red highlights indicate high model dependency."
                )
                with st.spinner("Computing activation gradients..."):
                    overlay_img = make_gradcam_overlay(raw_rgb)
                
                if overlay_img is not None:
                    col_gc1, col_gc2 = st.columns(2)
                    with col_gc1:
                        st.image(cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE)), caption="Base Face Crop", use_container_width=True)
                    with col_gc2:
                        st.image(overlay_img, caption="Grad-CAM Focus Overlay", use_container_width=True)
                    st.info("Highlights around jaw borders, eye alignments, and frame lines are common in synthetically manipulated inputs.")
                else:
                    st.warning("Grad-CAM could not be generated, but prediction results are still available.")

            with tab_mn:
                st.markdown("### Model Specifications")
                st.markdown(
                    "The neural network classifies face areas using transfer learning parameters on EfficientNetB0."
                )
                st.markdown(
                    f"""
                    - **Model File**: `{MODEL_PATH}`
                    - **Input size**: `224x224 RGB`
                    - **Target Classes**: `Real = 0, Fake = 1`
                    """
                )
                if metrics_data:
                    st.markdown("#### Evaluation Benchmark Dataset")
                    st.markdown(
                        f"""
                        - Accuracy: `{metrics_data.get('accuracy', 0.0):.2%}`
                        - ROC-AUC: `{metrics_data.get('roc_auc', 0.0):.2%}`
                        - Precision: `{metrics_data.get('precision', 0.0):.2%}`
                        - Recall: `{metrics_data.get('recall', 0.0):.2%}`
                        """
                    )
                st.markdown(
                    "Limitations: Validation benchmarks are calculated on standard test sets. "
                    "Accuracy may degrade under low contrast, severe compression, or extreme camera angles."
                )

    else:
        # ---------------------------------------------------------
        # VIDEO MODE
        # ---------------------------------------------------------
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix="." + file_suffix)
        try:
            temp_video.write(uploaded_file.read())
            temp_video_name = temp_video.name
            temp_video.close()
        except Exception as e:
            st.error(f"Failed to create temp video: {e}")
            st.stop()

        with st.spinner("Analyzing media and generating risk signals..."):
            result, frames = predict_video_aggregate(temp_video_name)

        try:
            os.remove(temp_video_name)
        except Exception:
            pass

        if result is None or len(frames) == 0:
            st.error("Could not read video frames. Verify the video file is not corrupted.")
            st.stop()

        # Recruiter Mode (Simplified)
        if dashboard_mode == "Recruiter Mode":
            st.markdown("---")
            st.markdown("### Video Threat Assessment")
            
            col_vid_1, col_vid_2 = st.columns([1, 1.2])
            with col_vid_1:
                st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
                st.video(uploaded_file)
                st.caption("Video Playback")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_vid_2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"**Threat Assessment:** {get_risk_badge_html(result['risk'])}", unsafe_allow_html=True)
                
                st.markdown("##### Threat Scoreboard")
                c_m1, c_m2, c_m3 = st.columns(3)
                with c_m1:
                    st.markdown(
                        f"""
                        <div class="glass-card-secondary" style="text-align: center;">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Overall Risk</div>
                            <div style="font-size:1.3rem; font-weight:800; margin:0.35rem 0; color:#8B5CF6;">{result['overall_identity_risk']:.1%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with c_m2:
                    st.markdown(
                        f"""
                        <div class="glass-card-secondary" style="text-align: center;">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Deepfake Risk</div>
                            <div style="font-size:1.3rem; font-weight:800; margin:0.35rem 0; color:#38BDF8;">{result['deepfake_video_risk']:.1%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with c_m3:
                    st.markdown(
                        f"""
                        <div class="glass-card-secondary" style="text-align: center;">
                            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Liveness</div>
                            <div style="font-size:1.3rem; font-weight:800; margin:0.35rem 0; color:#22C55E;">{result['liveness']['liveness_score']}/100</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.markdown(get_recommendation_box_html(result['risk']), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Sampled timeline gallery
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Sampled Video Timeline Frames")
            st.markdown("extrapolated frames selected for aggregate pipeline predictions:")
            max_thumbs = min(8, len(frames))
            grid_cols_count = min(4, max_thumbs)
            rows = [frames[i:i + grid_cols_count] for i in range(0, max_thumbs, grid_cols_count)]
            for r_idx, row in enumerate(rows):
                grid_cols = st.columns(grid_cols_count)
                for c_idx, frame_rgb in enumerate(row):
                    global_idx = r_idx * grid_cols_count + c_idx
                    grid_cols[c_idx].image(frame_rgb, caption=f"Frame {global_idx+1}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Investigator Mode (Expert Video Workspace)
        else:
            st.markdown("---")
            tab_v_ov, tab_v_ag, tab_v_lv, tab_v_fm, tab_v_mn = st.tabs([
                "Overview", 
                "Video Analysis", 
                "Liveness Signals", 
                "Sampled Timeline",
                "Model Notes"
            ])

            with tab_v_ov:
                col_vid_1, col_vid_2 = st.columns([1, 1.2])
                with col_vid_1:
                    st.video(uploaded_file)
                with col_vid_2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f"**Threat Assessment:** {get_risk_badge_html(result['risk'])}", unsafe_allow_html=True)
                    c_m1, c_m2, c_m3 = st.columns(3)
                    with c_m1:
                        st.markdown(
                            f"""
                            <div class="glass-card-secondary" style="text-align: center;">
                                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Overall Risk</div>
                                <div style="font-size:1.3rem; font-weight:800; margin:0.35rem 0; color:#8B5CF6;">{result['overall_identity_risk']:.1%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with c_m2:
                        st.markdown(
                            f"""
                            <div class="glass-card-secondary" style="text-align: center;">
                                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Deepfake Risk</div>
                                <div style="font-size:1.3rem; font-weight:800; margin:0.35rem 0; color:#38BDF8;">{result['deepfake_video_risk']:.1%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with c_m3:
                        st.markdown(
                            f"""
                            <div class="glass-card-secondary" style="text-align: center;">
                                <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Liveness</div>
                                <div style="font-size:1.3rem; font-weight:800; margin:0.35rem 0; color:#22C55E;">{result['liveness']['liveness_score']}/100</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    st.markdown(get_recommendation_box_html(result['risk']), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            with tab_v_ag:
                st.markdown("### Frame-level Aggregation Insights")
                st.markdown(
                    "Video classification combines frame-level predictions into a more stable video-level deepfake risk score."
                )
                
                table_html = f"""
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Analysis Description</th>
                    </tr>
                    <tr>
                        <td><strong>Frames Sampled</strong></td>
                        <td>{result['frames_used']} / 16</td>
                        <td>Number of uniform video frames extracted for model analysis.</td>
                    </tr>
                    <tr>
                        <td><strong>Faces Extracted</strong></td>
                        <td>{result['faces_detected']} / {result['frames_used']}</td>
                        <td>Frames where the system successfully isolated and cropped a face.</td>
                    </tr>
                    <tr>
                        <td><strong>Mean Fake Probability</strong></td>
                        <td>{result['mean_fake_probability']:.2%}</td>
                        <td>Average synthetic likelihood across the entire video.</td>
                    </tr>
                    <tr>
                        <td><strong>Median Fake Probability</strong></td>
                        <td>{result['median_fake_probability']:.2%}</td>
                        <td>Midpoint value, highly robust against single-frame outlier noise.</td>
                    </tr>
                    <tr>
                        <td><strong>Top-3 Fake Probability</strong></td>
                        <td>{result['top3_fake_probability']:.2%}</td>
                        <td>Average of top 3 high-risk frames. Catches short deepfake splice insertions.</td>
                    </tr>
                    <tr>
                        <td><strong>High-Risk Frame Ratio</strong></td>
                        <td>{result['high_risk_frame_ratio']:.2%}</td>
                        <td>Ratio of frames exceeding the 66% suspicious threat threshold.</td>
                    </tr>
                    <tr style="background-color: rgba(30, 41, 59, 0.4);">
                        <td><strong>Deepfake Video Risk</strong></td>
                        <td><strong>{result['deepfake_video_risk']:.2%}</strong></td>
                        <td>Weighted combination: 50% Mean + 25% Median + 15% Top-3 + 10% Ratio.</td>
                    </tr>
                </table>
                """
                st.markdown(table_html, unsafe_allow_html=True)

            with tab_v_lv:
                st.markdown("### Heuristic Passive Liveness Analysis")
                st.markdown(
                    "Passive liveness uses raw frame statistics to check for screen reflections, "
                    "moiré artifacts, and paper borders without requiring active actions."
                )
                
                liveness = result["liveness"]
                col_l1, col_l2 = st.columns([1, 1.4])
                
                with col_l1:
                    st.markdown(f"**Liveness Label**: `{liveness['liveness_label']}`")
                    st.markdown(f"**Liveness Score**: `{liveness['liveness_score']}/100`")
                    st.markdown(f"**Spoof/Replay Risk**: `{liveness['spoof_risk']:.1f}%`")
                    
                    lv_color = "#22C55E" if liveness['liveness_score'] >= 65 else ("#F59E0B" if liveness['liveness_score'] >= 40 else "#EF4444")
                    st.markdown("Liveness Confidence:")
                    st.markdown(get_progress_bar_html(liveness['liveness_score'], lv_color), unsafe_allow_html=True)
                    
                with col_l2:
                    liveness_table = f"""
                    <table class="metrics-table">
                        <tr>
                            <th>Liveness Component</th>
                            <th>Score</th>
                            <th>Analysis Context</th>
                        </tr>
                        <tr>
                            <td>Motion (25%)</td>
                            <td>{liveness['motion_score']}%</td>
                            <td>Detects pixel variance to identify static image presentations.</td>
                        </tr>
                        <tr>
                            <td>Sharpness (25%)</td>
                            <td>{liveness['sharpness_score']}%</td>
                            <td>Detects blurriness typical of print/low-res screen reviews.</td>
                        </tr>
                        <tr>
                            <td>Texture (20%)</td>
                            <td>{liveness['texture_score']}%</td>
                            <td>Identifies high-frequency patterns, such as digital screen moiré.</td>
                        </tr>
                        <tr>
                            <td>Exposure (15%)</td>
                            <td>{liveness['exposure_score']}%</td>
                            <td>Measures face lighting balance and contrast values.</td>
                        </tr>
                        <tr>
                            <td>Color Richness (15%)</td>
                            <td>{liveness['color_score']}%</td>
                            <td>Analyzes standard color deviations to verify live skin profiles.</td>
                        </tr>
                    </table>
                    """
                    st.markdown(liveness_table, unsafe_allow_html=True)

                st.warning("Passive liveness is heuristic and should only support manual review.")

            with tab_v_fm:
                st.markdown("### Sampled Video Timeline Frames")
                max_thumbs = min(16, len(frames))
                grid_cols_count = min(4, max_thumbs)
                rows = [frames[i:i + grid_cols_count] for i in range(0, max_thumbs, grid_cols_count)]
                for r_idx, row in enumerate(rows):
                    grid_cols = st.columns(grid_cols_count)
                    for c_idx, frame_rgb in enumerate(row):
                        global_idx = r_idx * grid_cols_count + c_idx
                        grid_cols[c_idx].image(frame_rgb, caption=f"Frame {global_idx+1}", use_container_width=True)

            with tab_v_mn:
                st.markdown("### Model Specifications")
                st.markdown(
                    f"""
                    - **Model File**: `{MODEL_PATH}`
                    - **Input size**: `224x224 RGB`
                    - **Target Classes**: `Real = 0, Fake = 1`
                    """
                )
                if metrics_data:
                    st.markdown("#### Evaluation Benchmark Dataset")
                    st.markdown(
                        f"""
                        - Accuracy: `{metrics_data.get('accuracy', 0.0):.2%}`
                        - ROC-AUC: `{metrics_data.get('roc_auc', 0.0):.2%}`
                        - Precision: `{metrics_data.get('precision', 0.0):.2%}`
                        - Recall: `{metrics_data.get('recall', 0.0):.2%}`
                        """
                    )
                st.markdown(
                    "Limitations: Validation benchmarks are calculated on standard test sets. "
                    "Accuracy may degrade under low contrast, severe compression, or extreme camera angles."
                )

# ---------------------------------------------------------
# FUTURE ROADMAP
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0;'>Future Roadmap</h3>", unsafe_allow_html=True)
st.markdown("<p class='muted-text'>Planned upgrades to expand SatyaLens as an AI security research platform.</p>", unsafe_allow_html=True)

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown(
        """
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Acoustic Deepfake Detection</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Add voice and audio manipulation analysis.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">MediaPipe / RetinaFace Upgrade</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Improve face detection accuracy beyond Haar Cascades.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Robustness Stress Testing</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Test prediction stability under blur, crop, noise, exposure, and compression.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">PDF Risk Reports</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Generate downloadable review summaries.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">FastAPI Backend</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Add API-based inference for external integrations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_r2:
    st.markdown(
        """
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Docker Deployment</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Containerize the app for reproducible deployment.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">ONNX / TFLite Export</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Support lightweight model deployment formats.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Bias and Fairness Testing</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Evaluate performance differences across demographics and data conditions.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Confidence Calibration</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Improve probability reliability.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding: 0.15rem 0.45rem; color: #8B5CF6 !important; border-color: rgba(139, 92, 246, 0.25); background-color: rgba(139, 92, 246, 0.1); float:right;">Later</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Live Camera Review Mode</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Add optional camera-based review workflow.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# THEME-AWARE DEVELOPER FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="glass-card" style="text-align: center; margin-top: 3rem; padding: 1.5rem; border-color: rgba(139, 92, 246, 0.25);">
        <strong style="font-size: 1.15rem; color: #F8FAFC;">SatyaLens</strong>
        <p style="margin: 0.5rem 0 0.25rem 0; font-size: 0.85rem; color: #CBD5E1;">
            Built with passion by <strong>Arnav Raj (Cybroarnv)</strong>
        </p>
        <p style="margin: 0.25rem auto; font-size: 0.825rem; color: #94A3B8; max-width: 650px;">
            AI Security Research Prototype for Deepfake Risk Analysis, Identity-Risk Estimation, Passive Liveness Signals, and Manual Review Support.
        </p>
        <div style="margin: 0.75rem 0; font-size: 0.85rem;">
            <a href="https://github.com/yocybroarnv" target="_blank" style="color: #38BDF8; text-decoration: none; margin-right: 1.5rem;">GitHub</a>
            <a href="https://www.linkedin.com/in/arnav-raj-professional" target="_blank" style="color: #38BDF8; text-decoration: none; margin-right: 1.5rem;">LinkedIn</a>
            <span style="color: #94A3B8;">MIT Licensed</span>
        </div>
        <p style="margin: 0.5rem auto 0 auto; font-size: 0.75rem; color: #94A3B8; max-width: 600px; line-height: 1.45;">
            Research prototype only — not for automated KYC, hiring, legal, surveillance, or biometric verification decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)