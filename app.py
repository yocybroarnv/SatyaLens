import os
import sys
import subprocess

# Self-healing OpenCV installer to resolve Streamlit Cloud libGL/libglib apt package conflicts
try:
    import cv2
except ImportError:
    # Force uninstall standard OpenCV packages
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Reinstall headless version
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Clear Python module import cache
    for m in list(sys.modules.keys()):
        if m.startswith("cv2") or m.startswith("_cv2"):
            sys.modules.pop(m, None)
            
    import cv2

import json
import tempfile
import hashlib
import datetime
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
from PIL.ExifTags import TAGS

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
        
        .risk-badge-success {
            background-color: rgba(34, 197, 94, 0.1);
            color: #22C55E !important;
            border-color: rgba(34, 197, 94, 0.25);
        }
        
        .risk-badge-warning {
            background-color: rgba(245, 158, 11, 0.1);
            color: #F59E0B !important;
            border-color: rgba(245, 158, 11, 0.25);
        }
        
        .risk-badge-danger {
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
        .action-success {
            background-color: rgba(34, 197, 94, 0.05);
            color: #22C55E !important;
            border-color: rgba(34, 197, 94, 0.25);
        }
        .action-warning {
            background-color: rgba(245, 158, 11, 0.05);
            color: #F59E0B !important;
            border-color: rgba(245, 158, 11, 0.25);
        }
        .action-danger {
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
# MOCK UPLOAD OBJECT FOR DYNAMIC DEMO SAMPLES
# ---------------------------------------------------------
import io

class MockUploadedFile(io.BytesIO):
    def __init__(self, name, size, data):
        super().__init__(data)
        self.name = name
        self.size = size

# ---------------------------------------------------------
# HELPER ACTIONS / VERDICTS
# ---------------------------------------------------------
def estimate_risk_category(score):
    try:
        score = float(score)
    except Exception:
        score = 0.0

    if score < 0.30:
        return {
            "label": "Low Risk",
            "action": "Continue normal review, but do not treat this result as final proof.",
            "color": "success"
        }
    elif score < 0.67:
        return {
            "label": "Suspicious",
            "action": "Manual review and additional verification are recommended.",
            "color": "warning"
        }
    else:
        return {
            "label": "High Risk",
            "action": "Strong manual verification is recommended before taking any action.",
            "color": "danger"
        }

def get_risk_badge_html(risk_lvl, color):
    return f'<span class="risk-badge risk-badge-{color}">{risk_lvl}</span>'

def get_progress_bar_html(percentage, color_hex):
    return f"""
    <div class="progress-container">
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {percentage}%; background-color: {color_hex};"></div>
        </div>
    </div>
    """

def get_recommendation_box_html(risk_lvl, action, color):
    return f"""
    <div class="action-box action-{color}">
        <strong>Recommended Action ({risk_lvl}):</strong> {action}
    </div>
    """

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def format_file_size(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"

@st.cache_data
def load_demo_image_url(url):
    try:
        # Standard urllib fetching
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read()
    except Exception:
        return None

# ---------------------------------------------------------
# CORE MODEL LOADING WITH CACHE & WARM-UP
# ---------------------------------------------------------
@st.cache_resource
def load_deepfake_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights file {MODEL_PATH} is missing.")
    m = keras.models.load_model(MODEL_PATH)
    m.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
    return m

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
    st.sidebar.markdown('<div style="margin-top:0.35rem;"><span class="risk-badge risk-badge-warning" style="padding: 0.1rem 0.4rem; font-size:0.65rem; border-color:rgba(245,158,11,0.25);">Research Prototype</span></div>', unsafe_allow_html=True)
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
        st.sidebar.markdown('<span class="risk-badge risk-badge-success" style="padding: 0.15rem 0.5rem; font-size:0.75rem;">Model Loaded</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"Filename: `{MODEL_PATH}`")
    else:
        st.sidebar.markdown('<span class="risk-badge risk-badge-danger" style="padding: 0.15rem 0.5rem; font-size:0.75rem;">Model Missing</span>', unsafe_allow_html=True)
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
        <div class="glass-card" style="border-left: 4px solid #F59E0B; background-color: rgba(245, 158, 11, 0.05); padding: 1rem 1.25rem;">
            <strong style="color: #F59E0B; font-size: 0.9rem;">Research Prototype Disclaimer:</strong>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; line-height: 1.4; color: #CBD5E1;">
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

# Load Neural Network Model
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
        <div class="glass-card" style="border-left: 4px solid #EF4444; background-color: rgba(239, 68, 68, 0.05);">
            <strong style="color: #EF4444;">Core Model Weights Missing</strong>
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

# Initialize session state cache for results
if "cached_hash" not in st.session_state:
    st.session_state.cached_hash = None
    st.session_state.cached_prediction = None
    st.session_state.cached_metadata = None
    st.session_state.cached_gradcam = None

# ---------------------------------------------------------
# METADATA EXTRACTION HELPERS
# ---------------------------------------------------------
def get_image_metadata(uploaded_file, file_bytes):
    meta = {
        "name": uploaded_file.name,
        "size": format_file_size(len(file_bytes)),
        "extension": uploaded_file.name.split(".")[-1].upper(),
        "hash": get_file_hash(file_bytes)[:16],
        "type": "Image",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        img = Image.open(uploaded_file)
        meta.update({
            "width": img.width,
            "height": img.height,
            "aspect_ratio": f"{img.width / img.height:.2f}",
            "format": img.format,
            "mode": img.mode,
            "channels": len(img.getbands()) if hasattr(img, "getbands") else 3,
            "exif_present": "No",
            "exif_count": 0,
            "camera": "Unknown",
            "date_taken": "Unknown",
            "gps_present": "No"
        })
        
        exif_data = img.getexif() if hasattr(img, "getexif") else None
        if exif_data:
            meta["exif_present"] = "Yes"
            meta["exif_count"] = len(exif_data)
            
            camera_make = ""
            camera_model = ""
            for tag_id, val in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "Make":
                    camera_make = str(val)
                elif tag_name == "Model":
                    camera_model = str(val)
                elif tag_name in ["DateTime", "DateTimeOriginal"]:
                    meta["date_taken"] = str(val)
            
            if camera_make or camera_model:
                meta["camera"] = f"{camera_make} {camera_model}".strip()
            
            # Check for GPSInfo tag (34853)
            if 34853 in exif_data:
                meta["gps_present"] = "Yes"
    except Exception:
        pass
    return meta

def get_video_metadata(temp_video_path, file_bytes, uploaded_file):
    meta = {
        "name": uploaded_file.name,
        "size": format_file_size(len(file_bytes)),
        "extension": uploaded_file.name.split(".")[-1].upper(),
        "hash": get_file_hash(file_bytes)[:16],
        "type": "Video",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration": 0.0,
        "duration_formatted": "00:00",
        "codec": "Unknown",
        "readable": "No",
        "frame_extraction": "No"
    }
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if cap.isOpened():
            meta["readable"] = "Yes"
            meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
            meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if meta["fps"] > 0:
                meta["duration"] = round(meta["frame_count"] / meta["fps"], 2)
                mins = int(meta["duration"] // 60)
                secs = int(meta["duration"] % 60)
                meta["duration_formatted"] = f"{mins:02d}:{secs:02d}"
            
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_chars = [chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]
            meta["codec"] = "".join(codec_chars).strip()
            meta["frame_extraction"] = "Yes" if meta["frame_count"] > 0 else "No"
            
            cap.release()
    except Exception:
        pass
    return meta

def render_metadata_panel(meta, mode):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top: 0;'>Media Metadata & Audit Snapshot</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**File Name**: `{meta.get('name')}`")
        st.markdown(f"**Size**: `{meta.get('size')}`")
        st.markdown(f"**Extension**: `{meta.get('extension')}`")
        st.markdown(f"**SHA-256 Hash**: `{meta.get('hash')}`")
    with col2:
        st.markdown(f"**Upload Category**: `{meta.get('type')}`")
        st.markdown(f"**Analysis Timestamp**: `{meta.get('timestamp')}`")
        st.markdown(f"**Processing Mode**: `{mode}`")
        
    if meta.get("type") == "Image":
        st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.1);' />", unsafe_allow_html=True)
        col_im1, col_im2 = st.columns(2)
        with col_im1:
            st.markdown(f"**Dimensions**: `{meta.get('width')} x {meta.get('height')}`")
            st.markdown(f"**Aspect Ratio**: `{meta.get('aspect_ratio')}`")
            st.markdown(f"**Format**: `{meta.get('format')}`")
            st.markdown(f"**Color Mode**: `{meta.get('mode')}` (Channels: `{meta.get('channels')}`)")
        with col_im2:
            st.markdown(f"**EXIF Present**: `{meta.get('exif_present')}` (Tags: `{meta.get('exif_count')}`)")
            st.markdown(f"**Camera Info**: `{meta.get('camera')}`")
            st.markdown(f"**Date Taken**: `{meta.get('date_taken')}`")
            st.markdown(f"**GPS Metadata present**: `{meta.get('gps_present')}`")
    elif meta.get("type") == "Video":
        st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.1);' />", unsafe_allow_html=True)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown(f"**Resolution**: `{meta.get('width')} x {meta.get('height')}`")
            st.markdown(f"**FPS**: `{meta.get('fps'):.2f}`")
            st.markdown(f"**Total Frames**: `{meta.get('frame_count')}`")
        with col_v2:
            st.markdown(f"**Duration**: `{meta.get('duration')}s` (`{meta.get('duration_formatted')}`)")
            st.markdown(f"**Codec**: `{meta.get('codec')}`")
            st.markdown(f"**Readable**: `{meta.get('readable')}`")
            st.markdown(f"**Sampled Frames**: `{meta.get('sampled_frames_count', 0)}`")

    if mode == "Investigator Mode":
        st.markdown(" ")
        with st.expander("Raw Metadata Debug"):
            st.json(meta)
            
    st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Sharpness calculation
    sharpness = float(np.clip(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]) / 180.0, 0, 1))

    # Texture details
    texture = float(np.clip(
        np.mean([np.mean(cv2.Canny(g, 80, 160) > 0) for g in grays]) / 0.12,
        0,
        1
    ))

    # Lighting exposure
    brightness = float(np.mean([np.mean(g) for g in grays]))
    exposure = float(1 - np.clip(abs(brightness - 127.5) / 127.5, 0, 1))

    # Speed-optimized motion: downscale grays to 96x96 before frame differences
    small_grays = [cv2.resize(g, (96, 96)) for g in grays]
    if len(small_grays) > 1:
        diffs = [np.mean(cv2.absdiff(a, b)) for a, b in zip(small_grays[:-1], small_grays[1:])]
        motion = float(np.clip(np.mean(diffs) / 18.0, 0, 1))
    else:
        motion = 0.0

    # Color richness
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

def predict_video_aggregate(path, mode):
    n_frames = 8 if mode == "Recruiter Mode" else 16
    
    status_text = st.empty()
    
    status_text.text("Reading video...")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        status_text.empty()
        return None, []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    status_text.text("Sampling frames...")
    frames = read_video_frames(path, n_frames)
    if len(frames) == 0:
        status_text.empty()
        return None, []
    
    status_text.text("Running batch inference...")
    batch_faces = []
    detected_faces_count = 0
    
    for f in frames:
        cropped, detected, method = face_detector.detect_and_crop(f)
        resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
        arr = resized.astype("float32")
        batch_faces.append(arr)
        if detected:
            detected_faces_count += 1
            
    # Run batch inference in a single call for speed
    batch_arr = np.array(batch_faces)
    probs_batch = model.predict(batch_arr, verbose=0)
    probs = probs_batch[:, 0]
    
    status_text.text("Calculating liveness signals...")
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

    risk_data = estimate_risk_category(overall_risk)
    
    status_text.empty()

    return {
        "frames_used": len(frames),
        "faces_detected": detected_faces_count,
        "mean_fake_probability": round(mean_prob, 4),
        "median_fake_probability": round(median_prob, 4),
        "top3_fake_probability": round(top3_prob, 4),
        "high_risk_frame_ratio": round(high_risk_ratio, 4),
        "deepfake_video_risk": round(deepfake_risk, 4),
        "overall_identity_risk": round(overall_risk, 4),
        "risk": risk_data["label"],
        "action": risk_data["action"],
        "color": risk_data["color"],
        "liveness": liveness
    }, frames

# ---------------------------------------------------------
# GRAD-CAM EXPLAINABILITY HEATMAPS
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
st.markdown("<span class='muted-text'>Upload a face image or short video, or select a demo sample to scan.</span>", unsafe_allow_html=True)

demo_choice = st.selectbox(
    "Source Mode:",
    ["Upload My Own Media", "Demo Real Face (Unsplash)", "Demo Synthetic Pattern (Generated)"],
    index=0
)

uploaded_file = None
if demo_choice == "Upload My Own Media":
    uploaded_file = st.file_uploader(
        "Media File Upload",
        type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed"
    )
    st.markdown("<span style='font-size:0.8rem; color:#94A3B8;'>Supported formats: JPG, JPEG, PNG, WEBP, MP4, AVI, MOV, MKV, WEBM</span>", unsafe_allow_html=True)
elif demo_choice == "Demo Real Face (Unsplash)":
    unsplash_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?fit=crop&w=224&h=224&q=80"
    with st.spinner("Downloading Unsplash demo image..."):
        img_data = load_demo_image_url(unsplash_url)
    if img_data:
        uploaded_file = MockUploadedFile("unsplash_real_face.jpg", len(img_data), img_data)
    else:
        st.error("Could not fetch Unsplash demo image. Please upload a file manually.")
else:
    # Demo Synthetic Pattern (Generated)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(img, (112, 112), 80, (180, 180, 180), -1)
    cv2.circle(img, (85, 95), 12, (20, 20, 20), -1)
    cv2.circle(img, (139, 95), 12, (20, 20, 20), -1)
    for i in range(0, 224, 16):
        cv2.line(img, (i, 0), (i, 224), (100, 220, 100), 1)
        cv2.line(img, (0, i), (224, i), (100, 220, 100), 1)
    
    success, encoded = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if success:
        img_data = encoded.tobytes()
        uploaded_file = MockUploadedFile("synthetic_grid_pattern.png", len(img_data), img_data)

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
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0) # Reset stream pointer for PIL/CV2
    
    file_suffix = uploaded_file.name.lower().split(".")[-1]
    is_image = file_suffix in ["jpg", "jpeg", "png", "webp"]
    
    # Calculate file SHA-256 hash to check against session cache
    file_hash = get_file_hash(file_bytes)
    
    # Load or Compute Metadata & Prediction Results
    if st.session_state.cached_hash == file_hash:
        metadata_result = st.session_state.cached_metadata
        prediction_result = st.session_state.cached_prediction
    else:
        # Clear previous session state entries
        st.session_state.cached_hash = file_hash
        st.session_state.cached_gradcam = None
        
        # Extract Metadata
        if is_image:
            metadata_result = get_image_metadata(uploaded_file, file_bytes)
            
            # Predict Image
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
                raw_rgb = np.array(pil_image)
                fake_prob, cropped_face, face_found, det_method = predict_rgb(raw_rgb)
                risk_data = estimate_risk_category(fake_prob)
                prediction_result = {
                    "fake_probability": fake_prob,
                    "cropped_face": cropped_face,
                    "face_found": face_found,
                    "detection_method": det_method,
                    "verdict": "Fake" if fake_prob >= 0.5 else "Real",
                    "risk": risk_data["label"],
                    "action": risk_data["action"],
                    "color": risk_data["color"],
                    "raw_rgb": raw_rgb
                }
            except Exception as e:
                st.error(f"Image analysis failed: {e}")
                st.stop()
        else:
            # Video File Flow
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix="." + file_suffix)
            try:
                temp_video.write(file_bytes)
                temp_video_name = temp_video.name
                temp_video.close()
            except Exception as e:
                st.error(f"Failed to create temp video: {e}")
                st.stop()
                
            metadata_result = get_video_metadata(temp_video_name, file_bytes, uploaded_file)
            
            with st.spinner("Analyzing video frames and liveness signals..."):
                try:
                    prediction_result, frames = predict_video_aggregate(temp_video_name, dashboard_mode)
                    prediction_result["frames"] = frames
                except Exception as e:
                    st.error(f"Video analysis failed: {e}")
                    st.stop()
                finally:
                    try:
                        os.remove(temp_video_name)
                    except Exception:
                        pass
                        
            # Update metadata with correct sampled frames count
            metadata_result["sampled_frames_count"] = prediction_result.get("frames_used", 0)
            
        # Store results in cache
        st.session_state.cached_metadata = metadata_result
        st.session_state.cached_prediction = prediction_result

    # Render Media Metadata Section immediately after upload
    render_metadata_panel(metadata_result, dashboard_mode)

    # Setup variables for results & logs
    risk_lvl = ""
    fake_prob = 0.0
    result_dict = {}

    # ---------------------------------------------------------
    # RESULTS RENDERING SECTION
    # ---------------------------------------------------------
    if is_image:
        # IMAGE MODE RESULTS
        pred_lbl = prediction_result["verdict"]
        risk_lvl = prediction_result["risk"]
        rec_action = prediction_result["action"]
        risk_color = prediction_result["color"]
        fake_prob = prediction_result["fake_probability"]
        cropped_face = prediction_result["cropped_face"]
        face_found = prediction_result["face_found"]
        det_method = prediction_result["detection_method"]
        raw_rgb = prediction_result["raw_rgb"]
        
        result_dict = {"liveness": {"liveness_score": "N/A"}}

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
                            {get_risk_badge_html(risk_lvl, risk_color)}
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

                pb_color = "#22C55E" if risk_color == "success" else ("#F59E0B" if risk_color == "warning" else "#EF4444")
                st.markdown(get_progress_bar_html(fake_prob * 100, pb_color), unsafe_allow_html=True)
                st.markdown(get_recommendation_box_html(risk_lvl, rec_action, risk_color), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.warning("Image-only liveness is limited. Video provides stronger liveness signals.")

            # Explainability (Only on demand via click to improve speed)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Explainability Map")
            st.markdown("Highlighted areas show regions that influenced the model prediction. Grad-CAM is interpretability support, not forensic proof.")
            
            if st.session_state.cached_gradcam is not None:
                st.image(st.session_state.cached_gradcam, caption="Grad-CAM Overlay", use_container_width=True)
            else:
                if st.button("Generate Grad-CAM Explanation"):
                    with st.spinner("Analyzing explainability focus..."):
                        overlay_img = make_gradcam_overlay(raw_rgb)
                        if overlay_img is not None:
                            st.session_state.cached_gradcam = overlay_img
                            st.rerun()
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
                                {get_risk_badge_html(risk_lvl, risk_color)}
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
                    pb_color = "#22C55E" if risk_color == "success" else ("#F59E0B" if risk_color == "warning" else "#EF4444")
                    st.markdown(get_progress_bar_html(fake_prob * 100, pb_color), unsafe_allow_html=True)
                    st.markdown(get_recommendation_box_html(risk_lvl, rec_action, risk_color), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.warning("Image-only liveness is limited. Video provides stronger liveness signals.")

            with tab_gc:
                st.markdown("### Neural Activation Focus")
                st.markdown(
                    "Grad-CAM overlay highlights the final convolutional block layers to map visual "
                    "attention indicators during predictions. Red highlights indicate high model dependency."
                )
                
                if st.session_state.cached_gradcam is not None:
                    col_gc1, col_gc2 = st.columns(2)
                    with col_gc1:
                        st.image(cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE)), caption="Base Face Crop", use_container_width=True)
                    with col_gc2:
                        st.image(st.session_state.cached_gradcam, caption="Grad-CAM Focus Overlay", use_container_width=True)
                    st.info("Highlights around jaw borders, eye alignments, and frame lines are common in synthetically manipulated inputs.")
                else:
                    if st.button("Generate Grad-CAM Explanation", key="gc_investigator"):
                        with st.spinner("Computing activation gradients..."):
                            overlay_img = make_gradcam_overlay(raw_rgb)
                            if overlay_img is not None:
                                st.session_state.cached_gradcam = overlay_img
                                st.rerun()
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
        # VIDEO MODE RESULTS
        result = prediction_result
        frames = prediction_result.get("frames", [])
        
        result_dict = result
        risk_lvl = result["risk"]
        fake_prob = result["deepfake_video_risk"]
        
        # Recruiter Mode (Simplified Video Results)
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
                st.markdown(f"**Threat Assessment:** {get_risk_badge_html(result['risk'], result['color'])}", unsafe_allow_html=True)
                
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
                
                st.markdown(get_recommendation_box_html(result['risk'], result['action'], result['color']), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Sampled timeline gallery (Recruiter Mode: 4 frames)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Sampled Video Timeline Frames")
            st.markdown("extrapolated frames selected for aggregate pipeline predictions:")
            max_thumbs = min(4, len(frames))
            grid_cols_count = min(4, max_thumbs)
            rows = [frames[i:i + grid_cols_count] for i in range(0, max_thumbs, grid_cols_count)]
            for r_idx, row in enumerate(rows):
                grid_cols = st.columns(grid_cols_count)
                for c_idx, frame_rgb in enumerate(row):
                    global_idx = r_idx * grid_cols_count + c_idx
                    grid_cols[c_idx].image(frame_rgb, caption=f"Frame {global_idx+1}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Investigator Mode (Expert Video Results)
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
                    st.markdown(f"**Threat Assessment:** {get_risk_badge_html(result['risk'], result['color'])}", unsafe_allow_html=True)
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
                    st.markdown(get_recommendation_box_html(result['risk'], result['action'], result['color']), unsafe_allow_html=True)
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
                max_thumbs = min(8, len(frames))
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
    # INTERACTIVE REVIEW LOG NOTES & AUDIT CERTIFICATE
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top:0;'>Reviewer Audit Logs & Decisioning</h4>", unsafe_allow_html=True)
    st.markdown("Record your manual review decisions and log comments below.")
    
    review_verdict = st.radio(
        "Verification Verdict:", 
        ["Unverified", "Approved (Real)", "Flagged (Suspicious)", "Rejected (Fake)"], 
        horizontal=True
    )
    reviewer_notes = st.text_area(
        "Observations & Comments:", 
        placeholder="Type comments, suspicious artifacts, or verification confirmation details here...",
        key="reviewer_log_notes"
    )
    
    if st.button("Generate Security Audit Certificate"):
        cert_text = f"""==================================================
              SATYALENS AUDIT CERTIFICATE
==================================================
Analysis Date:  {metadata_result.get('timestamp')}
Target File:    {metadata_result.get('name')}
SHA-256 Hash:   {metadata_result.get('hash')}
Upload Type:    {metadata_result.get('type')}
--------------------------------------------------
MODEL ASSESSMENT RESULTS:
Neural Score:   {fake_prob:.4%}
Liveness:       {result_dict.get('liveness', {}).get('liveness_score', 'N/A')}/100
Risk Category:  {risk_lvl}
--------------------------------------------------
REVIEWER DECISION:
Status:         {review_verdict}
Notes:          {reviewer_notes if reviewer_notes else 'None'}
==================================================
        """
        st.code(cert_text, language="text")
        
    st.markdown('</div>', unsafe_allow_html=True)

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
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Acoustic Deepfake Detection</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Add voice and audio manipulation analysis.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">MediaPipe / RetinaFace Upgrade</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Improve face detection accuracy beyond Haar Cascades.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Robustness Stress Testing</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Test prediction stability under blur, crop, noise, exposure, and compression.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">PDF Risk Reports</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Generate downloadable review summaries.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
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
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Docker Deployment</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Containerize the app for reproducible deployment.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">ONNX / TFLite Export</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Support lightweight model deployment formats.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Bias and Fairness Testing</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Evaluate performance differences across demographics and data conditions.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right; border-color:rgba(245,158,11,0.25);">Planned</span>
            <strong style="font-size:0.95rem; color:#F8FAFC;">Confidence Calibration</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:#94A3B8;">
                Improve probability reliability.
            </p>
        </div>
        <div class="glass-card-secondary">
            <span class="risk-badge risk-badge-warning" style="font-size:0.65rem; padding: 0.15rem 0.45rem; color: #8B5CF6 !important; border-color: rgba(139, 92, 246, 0.25); background-color: rgba(139, 92, 246, 0.1); float:right;">Later</span>
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
# DEVELOPER FOOTER
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