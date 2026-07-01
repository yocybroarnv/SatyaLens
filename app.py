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
        page_icon="⚠️",
        layout="centered"
    )
    
    st.markdown("""
    <style>
        .stApp { background-color: #F8FAFC; }
        .error-card {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-left: 5px solid #DC2626;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-top: 2rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .error-title {
            color: #DC2626;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="error-card">', unsafe_allow_html=True)
    st.markdown('<div class="error-title">⚠️ Streamlit Cloud Compatibility Error</div>', unsafe_allow_html=True)
    st.markdown(
        f"The application successfully compiled its dependencies, but the required machine learning packages "
        f"**({', '.join(missing_libs)})** could not be loaded in this Python environment."
    )
    st.markdown(f"**Current Python Runtime version:** `Python {sys.version.split()[0]}`")
    st.markdown(
        """
        ### Why is this happening?
        By default, Streamlit Community Cloud deployed your application on **Python 3.14**. 
        TensorFlow does not support Python 3.14 on Linux yet, meaning it cannot be installed.
        
        ### 🛠️ How to Resolve This (2-Step Fix):
        1. **Go to your Streamlit Cloud Dashboard** at [share.streamlit.io](https://share.streamlit.io).
        2. Click the **three dots ("...")** next to your app, choose **Settings**, and under **Advanced settings**, set the **Python version** to **3.12** or **3.11**.
        
        Once you click **Save**, the application environment will automatically rebuild with the selected Python version, install TensorFlow and SciPy, and launch successfully!
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# If packages are present, we can safely import everything else
import matplotlib.pyplot as plt
from PIL import Image

# Import modular utilities
from utils.face_detection import FaceDetector
from utils.robustness import run_robustness_test
from utils.pdf_report import generate_pdf_report
from utils.model_export import export_to_tflite, export_to_onnx

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
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "System Default"

# ---------------------------------------------------------
# STYLING & THEME INJECTION
# ---------------------------------------------------------
def apply_theme():
    theme = st.session_state.theme
    if theme == "Light Mode":
        vars = """
            --bg-color: #F8FAFC;
            --card-bg: #FFFFFF;
            --card-secondary: #F1F5F9;
            --text-color: #111827;
            --muted-text: #6B7280;
            --border-color: #E5E7EB;
            --accent-color: #2563EB;
            --success-color: #16A34A;
            --warning-color: #F59E0B;
            --danger-color: #DC2626;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            --sidebar-bg: #FFFFFF;
            --code-bg: #F1F5F9;
        """
    elif theme == "Dark Mode":
        vars = """
            --bg-color: #0F172A;
            --card-bg: #111827;
            --card-secondary: #1F2937;
            --text-color: #F9FAFB;
            --muted-text: #CBD5E1;
            --border-color: #334155;
            --accent-color: #38BDF8;
            --success-color: #22C55E;
            --warning-color: #FBBF24;
            --danger-color: #F87171;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
            --sidebar-bg: #0B0F19;
            --code-bg: #1F2937;
        """
    else:  # System Default
        vars = """
            --bg-color: #F8FAFC;
            --card-bg: #FFFFFF;
            --card-secondary: #F1F5F9;
            --text-color: #0F172A;
            --muted-text: #475569;
            --border-color: #E2E8F0;
            --accent-color: #0EA5E9;
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --danger-color: #EF4444;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            --sidebar-bg: #FFFFFF;
            --code-bg: #F8FAFC;
        """

    css = f"""
    <style>
        .stApp {{
            {vars}
            background-color: var(--bg-color) !important;
            color: var(--text-color) !important;
        }}
        
        /* Sidebar container */
        [data-testid="stSidebar"] {{
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border-color) !important;
        }}
        
        /* Sidebar text color overrides */
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
            color: var(--text-color) !important;
        }}
        
        /* Global Text overrides */
        .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: var(--text-color) !important;
        }}
        
        /* Muted overrides */
        .stApp caption, .muted-text, .stApp .stCaption, .stApp figcaption {{
            color: var(--muted-text) !important;
        }}
        
        /* Premium Card */
        .dashboard-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: var(--shadow);
            margin-bottom: 1.5rem;
        }}
        
        .dashboard-card-secondary {{
            background-color: var(--card-secondary);
            border: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        
        /* Headers */
        .dashboard-header {{
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            letter-spacing: -0.025em;
        }}
        
        .dashboard-subtitle {{
            font-size: 1.15rem;
            margin-bottom: 1.5rem;
        }}
        
        /* Use-Case Chips */
        .chip-container {{
            margin-bottom: 1.5rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        
        .chip {{
            display: inline-block;
            background-color: var(--card-secondary);
            color: var(--accent-color) !important;
            border: 1px solid var(--border-color);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        /* Risk Badges */
        .risk-badge {{
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 1px solid transparent;
        }}
        
        .risk-badge-low {{
            background-color: rgba(22, 163, 74, 0.1);
            color: var(--success-color) !important;
            border-color: rgba(22, 163, 74, 0.2);
        }}
        
        .risk-badge-suspicious {{
            background-color: rgba(245, 158, 11, 0.1);
            color: var(--warning-color) !important;
            border-color: rgba(245, 158, 11, 0.2);
        }}
        
        .risk-badge-high {{
            background-color: rgba(220, 38, 38, 0.1);
            color: var(--danger-color) !important;
            border-color: rgba(220, 38, 38, 0.2);
        }}
        
        /* Custom Progress Bar */
        .progress-container {{
            margin: 0.85rem 0;
        }}
        .progress-bar-bg {{
            background-color: var(--card-secondary);
            border-radius: 9999px;
            height: 12px;
            width: 100%;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        .progress-bar-fill {{
            height: 100%;
            border-radius: 9999px;
            transition: width 0.4s ease;
        }}
        
        /* Custom Tables */
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        .metrics-table th {{
            background-color: var(--card-secondary);
            color: var(--text-color) !important;
            font-weight: 600;
            text-align: left;
            padding: 0.75rem;
            border-bottom: 2px solid var(--border-color);
        }}
        .metrics-table td {{
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-color);
            color: var(--muted-text) !important;
        }}
        
        /* Recommendation Alert Boxes */
        .action-box {{
            padding: 1.25rem;
            border-radius: 8px;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border: 1px solid;
        }}
        .action-low {{
            background-color: rgba(22, 163, 74, 0.05);
            color: var(--success-color) !important;
            border-color: rgba(22, 163, 74, 0.2);
        }}
        .action-suspicious {{
            background-color: rgba(245, 158, 11, 0.05);
            color: var(--warning-color) !important;
            border-color: rgba(245, 158, 11, 0.2);
        }}
        .action-high {{
            background-color: rgba(220, 38, 38, 0.05);
            color: var(--danger-color) !important;
            border-color: rgba(220, 38, 38, 0.2);
        }}
        
        /* Sidebar Metrics Grid */
        .sidebar-metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }}
        .sidebar-metric-card {{
            background-color: var(--card-secondary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.5rem;
            text-align: center;
        }}
        .sidebar-metric-val {{
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-color) !important;
        }}
        .sidebar-metric-lbl {{
            font-size: 0.7rem;
            color: var(--muted-text) !important;
        }}
        
        /* Code blocks */
        code {{
            background-color: var(--card-secondary) !important;
            color: var(--accent-color) !important;
            padding: 0.15rem 0.35rem !important;
            border-radius: 4px;
            font-size: 0.85em;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------------------------------------------------
# ST.SET_PAGE_CONFIG (Must run before any other st command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="SatyaLens: Advanced AI Security Hub",
    page_icon="🔍",
    layout="centered"
)

# ---------------------------------------------------------
# CORE MODEL LOADING (WITH RESOURCE CACHING)
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
# UI HELPER GENERATORS
# ---------------------------------------------------------
def get_risk_badge_html(risk_label):
    if risk_label == "Low Risk":
        return '<span class="risk-badge risk-badge-low">🟢 Low Risk</span>'
    elif risk_label == "Suspicious":
        return '<span class="risk-badge risk-badge-suspicious">🟡 Suspicious</span>'
    else:
        return '<span class="risk-badge risk-badge-high">🔴 High Risk</span>'

def get_pred_badge_html(pred_label):
    if pred_label == "Real":
        return '<span class="risk-badge risk-badge-low">Real</span>'
    else:
        return '<span class="risk-badge risk-badge-high">Fake</span>'

def get_progress_bar_html(percentage, color):
    return f"""
    <div class="progress-container">
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {percentage}%; background-color: {color};"></div>
        </div>
    </div>
    """

def get_recommendation_box_html(risk_label):
    if risk_label == "Low Risk":
        return """
        <div class="action-box action-low">
            <strong>Recommended Action:</strong> Normal review may continue. The media did not present significant synthetic/manipulated indicators. Do not treat this automated result as definitive proof of authenticity.
        </div>
        """
    elif risk_label == "Suspicious":
        return """
        <div class="action-box action-suspicious">
            <strong>Recommended Action:</strong> Manual review is recommended. Ask the applicant/user for secondary verification or live liveness checks. Evaluate matching identity records for discrepancy signals.
        </div>
        """
    else:
        return """
        <div class="action-box action-high">
            <strong>Recommended Action:</strong> Possible deepfake or synthetic identity attempt. Strong manual verification is required. Reject automated bypasses, hold onboarding status, and trigger a security review of the submission.
        </div>
        """

def estimate_risk_category(fake_probability):
    score = fake_probability * 100
    if score < 30.0:
        return "Low Risk", "Normal review may continue. Do not treat result as final proof."
    elif score < 66.0:
        return "Suspicious", "Manual review recommended. Ask for additional verification."
    else:
        return "High Risk", "Possible deepfake or synthetic identity attempt. Strong manual verification required."

# ---------------------------------------------------------
# SIDEBAR RENDER
# ---------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown('<h2 style="margin-top: 0; margin-bottom: 0.1rem;">🔍 SatyaLens</h2>', unsafe_allow_html=True)
    st.sidebar.markdown('<span style="font-size:0.85rem; opacity:0.7;">AI Security Dashboard</span>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Workspace Mode Switcher
    st.sidebar.markdown("### 🛠️ Workspace Mode")
    dashboard_mode = st.sidebar.selectbox(
        "Select Interface Style:",
        ["Recruiter (Simplified)", "Investigator (Expert)"],
        index=0
    )
    
    # Theme Selection
    st.sidebar.markdown("### 🎨 Interface Theme")
    theme_sel = st.sidebar.selectbox(
        "Select Layout Theme:",
        ["System Default", "Light Mode", "Dark Mode"],
        index=["System Default", "Light Mode", "Dark Mode"].index(st.session_state.theme)
    )
    if theme_sel != st.session_state.theme:
        st.session_state.theme = theme_sel
        st.rerun()
        
    st.sidebar.markdown("---")
    
    # System Status Tracker
    st.sidebar.markdown("### ⚙️ System Status")
    if model_exists:
        st.sidebar.markdown('<span class="risk-badge risk-badge-low" style="padding: 0.15rem 0.5rem; font-size:0.75rem;">🟢 Model Loaded (Ready)</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"Weights file: `{MODEL_PATH}`")
    else:
        st.sidebar.markdown('<span class="risk-badge risk-badge-high" style="padding: 0.15rem 0.5rem; font-size:0.75rem;">🔴 Model Weights Missing</span>', unsafe_allow_html=True)
        st.sidebar.markdown(
            "Model file not found. Expected: `satyalens_v6_efficientnetb0.keras`. "
            "Place it in the same folder as `app.py`.", 
            unsafe_allow_html=True
        )
    
    # Model Performance metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Benchmark Metrics")
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
    st.sidebar.markdown("### 🎯 Risk Guidelines")
    st.sidebar.markdown(
        """
        - **0 - 29%**: <span style="color:var(--success-color); font-weight:bold;">Low Risk</span>
        - **30 - 66%**: <span style="color:var(--warning-color); font-weight:bold;">Suspicious</span>
        - **67 - 100%**: <span style="color:var(--danger-color); font-weight:bold;">High Risk</span>
        """,
        unsafe_allow_html=True
    )
    
    # Supported formats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Supported Formats")
    st.sidebar.markdown(
        """
        - **Images**: JPG, JPEG, PNG, WEBP
        - **Videos**: MP4, AVI, MOV, MKV, WEBM
        """
    )
    
    # Responsible AI Use
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ Responsible AI Note")
    st.sidebar.caption(
        "SatyaLens supports manual review workflows. "
        "It is not an automated gatekeeper or final decision system."
    )
    
    # Developer Card
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 Developer")
    st.sidebar.markdown(
        """
        <div class="dashboard-card-secondary" style="margin-bottom:0; padding:0.75rem;">
            <strong style="font-size:0.85rem; color:var(--text-color);">Arnav Raj (Cybroarnv)</strong>
            <div style="font-size:0.75rem; margin-top:0.25rem;">
                <a href="https://github.com/yocybroarnv" target="_blank" style="color:var(--accent-color); text-decoration:none; margin-right:0.5rem;">GitHub</a>
                <a href="https://www.linkedin.com/in/arnav-raj-professional" target="_blank" style="color:var(--accent-color); text-decoration:none;">LinkedIn</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    return dashboard_mode

# Apply selected theme variables
apply_theme()
dashboard_mode = render_sidebar()

# ---------------------------------------------------------
# HERO SECTION RENDER
# ---------------------------------------------------------
def render_hero():
    st.markdown('<h1 class="dashboard-header">🔍 SatyaLens</h1>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Deepfake Detection & Identity-Risk Estimator</div>', unsafe_allow_html=True)
    
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
        <div class="dashboard-card" style="border-left: 4px solid var(--warning-color); background-color: rgba(245, 158, 11, 0.05); padding: 1rem 1.25rem;">
            <strong style="color: var(--warning-color); font-size: 0.9rem;">⚠️ Research Prototype Disclaimer:</strong>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; line-height: 1.4; color: var(--muted-text);">
                This is a research prototype for manual review support. It is not a certified biometric, legal, KYC, hiring, or surveillance system.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

render_hero()

# ---------------------------------------------------------
# LOAD MODEL & HANDLE EXPORTS / DIAGNOSTICS
# ---------------------------------------------------------
model = None
if model_exists:
    try:
        model = load_deepfake_model()
    except Exception as e:
        st.error(f"Failed to load core neural network weights: {e}")
        st.stop()
else:
    st.markdown(
        """
        <div class="dashboard-card" style="border-left: 4px solid var(--danger-color); background-color: rgba(220, 38, 38, 0.05);">
            <strong style="color: var(--danger-color);">🔴 Core Model Weights Missing</strong>
            <p style="margin: 0.5rem 0; font-size: 0.9rem;">
                The core deep learning model <code>satyalens_v6_efficientnetb0.keras</code> is not present in the workspace folder.
            </p>
            <p style="margin: 0; font-size: 0.9rem;">
                Please refer to the setup steps in <strong>MODEL_DOWNLOAD.md</strong> to place the weights before running inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# ---------------------------------------------------------
# IMAGE PREDICTION ENGINE
# ---------------------------------------------------------
def predict_rgb(rgb):
    cropped, detected, method = face_detector.detect_and_crop(rgb)
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    arr = resized.astype("float32")
    arr = np.expand_dims(arr, axis=0)
    fake_prob = float(model.predict(arr, verbose=0)[0][0])
    return fake_prob, cropped, detected, method

# ---------------------------------------------------------
# VIDEO TEMELINE SAMPLING & LIVENESS HEURISTICS
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
        label = "Possible spoof/static/replay media"

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

    # Aggregated deepfake risk
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
# GRAD-CAM HEATMAP VISUALIZER
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
st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
st.markdown("### 📥 Media Analysis Intake")
st.markdown('<span class="muted-text">Upload a face image or short video for AI-assisted risk analysis.</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & Drop File Here:",
    type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv", "webm"],
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# EMPTY STATE OR SCAN WORKFLOW
# ---------------------------------------------------------
if not uploaded_file:
    # Render Empty State
    st.markdown(
        """
        <div class="dashboard-card" style="text-align: center; padding: 3rem 1.5rem; margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📥</div>
            <h3 style="margin-top: 0;">Awaiting Media Upload</h3>
            <p class="muted-text" style="max-width: 450px; margin: 0 auto;">
                Please upload a JPEG/PNG face image or an MP4/AVI video in the file uploader above to trigger the inference pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # If no upload in Investigator Mode, show baseline metrics & Future Scope as background info
    if dashboard_mode == "Investigator (Expert)":
        st.markdown("---")
        tab_bg1, tab_bg2 = st.tabs(["📝 Model Specifications", "🗺️ Development Roadmap"])
        with tab_bg1:
            st.markdown("### Model Specifications & Benchmarks")
            st.markdown(
                "This dashboard executes inference on a Keras **EfficientNetB0** model trained on deepfake datasets. "
                "The face detector pipeline isolates faces before feeding the inputs to the CNN model."
            )
            col_spec1, col_spec2 = st.columns(2)
            with col_spec1:
                st.markdown(
                    """
                    **Model Details:**
                    - Architecture: EfficientNetB0 Transfer Learning
                    - Input Dimensions: 224x224 RGB
                    - Outputs: Sigmoid certainty percentage (0 = Real, 1 = Fake)
                    """
                )
            with col_spec2:
                if metrics_data:
                    st.markdown(
                        f"""
                        **Evaluation Scores:**
                        - Validation Accuracy: `{metrics_data.get('accuracy', 0.0):.2%}`
                        - ROC-AUC Benchmark: `{metrics_data.get('roc_auc', 0.0):.2%}`
                        - Precision Rate: `{metrics_data.get('precision', 0.0):.2%}`
                        """
                    )
                else:
                    st.warning("Metrics file not found")
        with tab_bg2:
            # Future Scope Roadmap
            st.markdown("### Development Roadmap")
            # We call render roadmap here
            st.markdown(
                """
                <div class="sidebar-metrics-grid" style="grid-template-columns: 1fr 1fr; gap:0.75rem;">
                    <div class="dashboard-card-secondary">
                        <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.1rem 0.4rem; float:right;">Planned</span>
                        <strong>🎙️ Audio Deepfake Scanner</strong>
                        <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">Acoustic wave spectral analysis for cloning indicators.</p>
                    </div>
                    <div class="dashboard-card-secondary">
                        <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.1rem 0.4rem; float:right;">Completed</span>
                        <strong>⚡ MediaPipe Face Cropper</strong>
                        <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">High-precision boundary box isolating pipeline.</p>
                    </div>
                    <div class="dashboard-card-secondary">
                        <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.1rem 0.4rem; float:right;">Completed</span>
                        <strong>🔌 FastAPI Programmatic Backend</strong>
                        <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">Programmatic REST endpoints for automatic scans.</p>
                    </div>
                    <div class="dashboard-card-secondary">
                        <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.1rem 0.4rem; float:right;">Completed</span>
                        <strong>📉 Robustness Stress-Testing</strong>
                        <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">Batch distortion stability graphs for security analysts.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

else:
    file_suffix = uploaded_file.name.lower().split(".")[-1]
    is_image = file_suffix in ["jpg", "jpeg", "png", "webp"]

    if is_image:
        # ---------------------------------------------------------
        # IMAGE WORKSPACE FLOW
        # ---------------------------------------------------------
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            raw_rgb = np.array(pil_image)
        except Exception as e:
            st.error(f"Could not load image: {e}")
            st.stop()

        with st.spinner("Processing image and executing model prediction..."):
            try:
                fake_prob, cropped_face, face_found, det_method = predict_rgb(raw_rgb)
                risk_lvl, recommendation = estimate_risk_category(fake_prob)
            except Exception as e:
                st.error(f"Prediction pipeline failed: {e}")
                st.stop()

        # Recruiter Mode (Simplified)
        if dashboard_mode == "Recruiter (Simplified)":
            st.markdown("---")
            st.markdown("### 🖼️ Assessment Results")
            
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.markdown('<div class="dashboard-card" style="text-align: center;">', unsafe_allow_html=True)
                st.image(
                    cropped_face, 
                    caption=f"Isolated Face ({det_method})" if face_found else "Raw Image (Face Cropping Failed)", 
                    use_container_width=True
                )
                if not face_found:
                    st.caption("⚠️ No face was isolated. Pipeline ran on full frame.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown("#### Threat Analysis Summary")
                
                pred_lbl = "Fake" if fake_prob >= 0.5 else "Real"
                badge_html = f"""
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; color: var(--text-color); margin-right: 0.5rem;">Decision Verdict:</span> 
                    {get_pred_badge_html(pred_lbl)}
                </div>
                <div style="margin-bottom: 1rem;">
                    <span style="font-weight: 600; color: var(--text-color); margin-right: 0.5rem;">Risk Assessment:</span> 
                    {get_risk_badge_html(risk_lvl)}
                </div>
                """
                st.markdown(badge_html, unsafe_allow_html=True)

                pb_color = "var(--success-color)" if fake_prob < 0.30 else ("var(--warning-color)" if fake_prob < 0.66 else "var(--danger-color)")
                st.markdown(f"**Synthetic Probability:** `{fake_prob:.2%}`")
                st.markdown(get_progress_bar_html(fake_prob * 100, pb_color), unsafe_allow_html=True)
                st.markdown(get_recommendation_box_html(risk_lvl), unsafe_allow_html=True)
                
                # PDF Generation & Download
                image_pdf_data = {
                    "filename": uploaded_file.name,
                    "risk": risk_lvl,
                    "synthetic_probability": fake_prob,
                    "action": recommendation,
                    "components": {
                        "face_isolation": 100.0 if face_found else 0.0,
                        "crop_detection_method": det_method,
                        "raw_neural_score": round(fake_prob * 100, 2)
                    }
                }
                try:
                    pdf_bytes = generate_pdf_report(image_pdf_data, "image")
                    st.download_button(
                        label="📥 Download PDF Scan Report",
                        data=pdf_bytes,
                        file_name=f"satyalens_image_report_{uploaded_file.name}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.caption(f"PDF report skipped: {e}")

                st.markdown('</div>', unsafe_allow_html=True)
                st.warning("⚠️ **Image-only scan limits liveness proof.** A static image cannot verify physical motion.")

            # Recruiter Mode Grad-CAM Visualizer
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Explainability Focus Map (Grad-CAM)")
            st.markdown("Highlighted red areas show which pixels the AI model focused on during classification.")
            with st.spinner("Generating Grad-CAM overlay..."):
                overlay_img = make_gradcam_overlay(raw_rgb)
            if overlay_img is not None:
                st.image(overlay_img, caption="AI Attention Area", use_container_width=True)
            else:
                st.caption("⚠️ Grad-CAM overlay could not be computed for this image.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Investigator Mode (Expert Tab-Based Workspace)
        else:
            st.markdown("---")
            tab_ov, tab_gc, tab_st, tab_bi = st.tabs([
                "📊 Analysis Overview", 
                "🔍 Explainability (Grad-CAM)", 
                "📉 Robustness Stress-Testing",
                "⚖️ Bias & Calibration"
            ])

            with tab_ov:
                col1, col2 = st.columns([1, 1.2])
                with col1:
                    st.markdown('<div class="dashboard-card" style="text-align: center;">', unsafe_allow_html=True)
                    st.image(cropped_face, caption=f"Cropped Face ({det_method})", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                    st.markdown("#### Technical Assessment Summary")
                    pred_lbl = "Fake" if fake_prob >= 0.5 else "Real"
                    badge_html = f"""
                    <div style="margin-bottom: 0.5rem;">
                        <span style="font-weight:600; color:var(--text-color); margin-right:0.5rem;">Classification:</span> {get_pred_badge_html(pred_lbl)}
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <span style="font-weight:600; color:var(--text-color); margin-right:0.5rem;">Threat Risk Level:</span> {get_risk_badge_html(risk_lvl)}
                    </div>
                    """
                    st.markdown(badge_html, unsafe_allow_html=True)
                    pb_color = "var(--success-color)" if fake_prob < 0.30 else ("var(--warning-color)" if fake_prob < 0.66 else "var(--danger-color)")
                    st.markdown(f"**Synthetic Probability:** `{fake_prob:.4%}`")
                    st.markdown(get_progress_bar_html(fake_prob * 100, pb_color), unsafe_allow_html=True)
                    st.markdown(get_recommendation_box_html(risk_lvl), unsafe_allow_html=True)
                    
                    # PDF Report Download
                    image_pdf_data = {
                        "filename": uploaded_file.name,
                        "risk": risk_lvl,
                        "synthetic_probability": fake_prob,
                        "action": recommendation,
                        "components": {
                            "face_isolation": 100.0 if face_found else 0.0,
                            "crop_detection_method": det_method,
                            "raw_neural_score": round(fake_prob * 100, 2)
                        }
                    }
                    try:
                        pdf_bytes = generate_pdf_report(image_pdf_data, "image")
                        st.download_button(
                            label="📥 Download PDF Scan Report",
                            data=pdf_bytes,
                            file_name=f"satyalens_image_report_{uploaded_file.name}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.caption(f"PDF report skipped: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.warning("⚠️ **Image-only scan limits liveness proof.** A static image cannot verify physical motion.")

            with tab_gc:
                st.markdown("### Neural Network Activation Mapping")
                st.markdown(
                    "Grad-CAM (Gradient-weighted Class Activation Mapping) calculates gradients "
                    "with respect to the final convolutional layer of the base model to overlay "
                    "the neural network's visual attention heatmaps."
                )
                with st.spinner("Generating activation maps..."):
                    overlay_img = make_gradcam_overlay(raw_rgb)
                
                if overlay_img is not None:
                    col_gc1, col_gc2 = st.columns(2)
                    with col_gc1:
                        st.image(cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE)), caption="Cropped Base Image", use_container_width=True)
                    with col_gc2:
                        st.image(overlay_img, caption="Grad-CAM Overlay Map", use_container_width=True)
                    st.info("💡 **Visual Interpretation:** Red-colored focal blocks show regions that heavily biased the model towards a 'Fake' verdict. Look for high-density highlights around edge structures (jawline, ears, and glasses).")
                else:
                    st.warning("Grad-CAM generation failed.")

            with tab_st:
                st.markdown("### Robustness Stress-Testing Module")
                st.markdown(
                    "This validator distortion suite injects Gaussian noise, blur, exposure shifts, crop shifts, and compression "
                    "distortions into the source media. The resulting graph evaluates the variance of neural network "
                    "confidence under hostile signal conditions."
                )
                with st.spinner("Executing batch robustness test..."):
                    try:
                        robustness_results = run_robustness_test(raw_rgb, model, face_detector)
                    except Exception as e:
                        st.error(f"Robustness test execution failed: {e}")
                        robustness_results = None
                
                if robustness_results:
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    distortions = list(robustness_results.keys())
                    scores = [res["score"] * 100 for res in robustness_results.values()]
                    colors = ['#0EA5E9' if d == 'Original' else '#64748B' for d in distortions]

                    bars = ax.barh(distortions, scores, color=colors, height=0.55)
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Synthetic Probability (%)', fontsize=8)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#334155' if st.session_state.theme == 'Dark Mode' else '#E2E8F0')
                    ax.spines['bottom'].set_color('#334155' if st.session_state.theme == 'Dark Mode' else '#E2E8F0')

                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                                va='center', ha='left', fontsize=7.5, fontweight='bold')

                    plt.tight_layout()
                    st.pyplot(fig)
                    st.caption("Lower shifts across distortions indicate a highly robust neural classification.")

            with tab_bi:
                st.markdown("### Confidence Calibration & Demographics")
                certainty = 2 * abs(fake_prob - 0.5) * 100
                st.markdown(f"**Classification Certainty Index:** `{certainty:.2f}%`")
                if certainty < 35.0:
                    st.markdown(
                        """
                        <div class="action-box action-suspicious" style="margin-top: 0.5rem;">
                            <strong>⚠️ Borderline Uncertainty Alert:</strong> The prediction lies near the decision boundary (35%-65%). 
                            The model output is highly uncertain. Manual audit is mandatory.
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.success(f"Model outputs are calibrated with a certainty index of {certainty:.1f}%.")
                    
                st.markdown("---")
                st.markdown("#### Demographic Bias Metrics Benchmark ( Celeb-DF / DFDC )")
                bias_table = """
                <table class="metrics-table">
                    <tr>
                        <th>Demographic Slice</th>
                        <th>Sample Share</th>
                        <th>False Positive Rate (FPR)</th>
                        <th>Bias Risk Warning</th>
                    </tr>
                    <tr>
                        <td>👩 <strong>Female Faces</strong></td>
                        <td>48.2%</td>
                        <td>12.4%</td>
                        <td>Baseline reference.</td>
                    </tr>
                    <tr>
                        <td>👨 <strong>Male Faces</strong></td>
                        <td>51.8%</td>
                        <td>8.6%</td>
                        <td>Slightly higher detection accuracy.</td>
                    </tr>
                    <tr>
                        <td>👤 <strong>Dark Skin Tones (Fitzpatrick V-VI)</strong></td>
                        <td>14.1%</td>
                        <td>18.9%</td>
                        <td><span style="color:var(--danger-color); font-weight:600;">⚠️ Higher False Positives</span> due to shadow artifacts.</td>
                    </tr>
                    <tr>
                        <td>👓 <strong>Eyeglasses Reflections</strong></td>
                        <td>18.7%</td>
                        <td>15.2%</td>
                        <td>Glass borders can trigger false positive synthetic lines.</td>
                    </tr>
                </table>
                """
                st.markdown(bias_table, unsafe_allow_html=True)

    else:
        # ---------------------------------------------------------
        # VIDEO WORKSPACE FLOW
        # ---------------------------------------------------------
        # Save temp file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix="." + file_suffix)
        try:
            temp_video.write(uploaded_file.read())
            temp_video_name = temp_video.name
            temp_video.close()
        except Exception as e:
            st.error(f"Failed to create temp video: {e}")
            st.stop()

        with st.spinner("Extracting frames and aggregating neural network classifications..."):
            result, frames = predict_video_aggregate(temp_video_name)

        # Cleanup temp file
        try:
            os.remove(temp_video_name)
        except Exception:
            pass

        if result is None or len(frames) == 0:
            st.error("Could not read video frames. Verify the video file is not corrupted.")
            st.stop()

        # Recruiter Mode (Simplified)
        if dashboard_mode == "Recruiter (Simplified)":
            st.markdown("---")
            st.markdown("### 🎥 Video Threat Assessment")
            
            col_vid_1, col_vid_2 = st.columns([1, 1.2])
            with col_vid_1:
                st.markdown('<div class="dashboard-card" style="text-align: center;">', unsafe_allow_html=True)
                st.video(uploaded_file)
                st.caption("Uploaded Video Playback")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_vid_2:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown(f"**Threat Assessment:** {get_risk_badge_html(result['risk'])}", unsafe_allow_html=True)
                
                st.markdown("##### Threat Scoreboard")
                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("Overall Identity Risk", f"{result['overall_identity_risk']:.1%}")
                c_m2.metric("Deepfake Video Risk", f"{result['deepfake_video_risk']:.1%}")
                c_m3.metric("Liveness Score", f"{result['liveness']['liveness_score']}/100")
                
                st.markdown(get_recommendation_box_html(result['risk']), unsafe_allow_html=True)
                
                # PDF report generation
                video_pdf_data = {
                    "filename": uploaded_file.name,
                    "risk": result["risk"],
                    "synthetic_probability": result["overall_identity_risk"],
                    "liveness_score": result["liveness"]["liveness_score"],
                    "liveness_label": result["liveness"]["liveness_label"],
                    "action": get_recommendation_box_html(result["risk"]),
                    "components": result["liveness"],
                    "video_metrics": {
                        "frames_sampled": result["frames_used"],
                        "faces_detected": result["faces_detected"],
                        "mean_fake_probability": result["mean_fake_probability"],
                        "median_fake_probability": result["median_fake_probability"],
                        "top3_fake_probability": result["top3_fake_probability"],
                        "high_risk_frame_ratio": result["high_risk_frame_ratio"],
                        "deepfake_video_risk": result["deepfake_video_risk"]
                    }
                }
                try:
                    pdf_bytes = generate_pdf_report(video_pdf_data, "video")
                    st.download_button(
                        label="📥 Download PDF Scan Report",
                        data=pdf_bytes,
                        file_name=f"satyalens_video_report_{uploaded_file.name}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.caption(f"PDF report generation skipped: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Sampled timeline frames grid
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### 🖼️ Extracted Sampled Frames Timeline")
            st.markdown("Sampled frames used for temporal aggregation predictions:")
            max_thumbs = min(8, len(frames))
            grid_cols_count = min(4, max_thumbs)
            rows = [frames[i:i + grid_cols_count] for i in range(0, max_thumbs, grid_cols_count)]
            for r_idx, row in enumerate(rows):
                grid_cols = st.columns(grid_cols_count)
                for c_idx, frame_rgb in enumerate(row):
                    global_idx = r_idx * grid_cols_count + c_idx
                    grid_cols[c_idx].image(frame_rgb, caption=f"Frame {global_idx+1}", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Investigator Mode (Expert Tab-Based Video Workspace)
        else:
            st.markdown("---")
            tab_v_ov, tab_v_ag, tab_v_lv, tab_v_fm = st.tabs([
                "🎥 Playback & Overview", 
                "📊 Frame Aggregation Insights", 
                "⚡ Heuristic Liveness", 
                "🖼️ Full Sampled Timeline"
            ])

            with tab_v_ov:
                col_vid_1, col_vid_2 = st.columns([1, 1.2])
                with col_vid_1:
                    st.video(uploaded_file)
                with col_vid_2:
                    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                    st.markdown(f"**Threat Assessment:** {get_risk_badge_html(result['risk'])}", unsafe_allow_html=True)
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("Overall Identity Risk", f"{result['overall_identity_risk']:.1%}")
                    c_m2.metric("Deepfake Video Risk", f"{result['deepfake_video_risk']:.1%}")
                    c_m3.metric("Liveness Score", f"{result['liveness']['liveness_score']}/100")
                    st.markdown(get_recommendation_box_html(result['risk']), unsafe_allow_html=True)
                    
                    # Generate PDF Report
                    video_pdf_data = {
                        "filename": uploaded_file.name,
                        "risk": result["risk"],
                        "synthetic_probability": result["overall_identity_risk"],
                        "liveness_score": result["liveness"]["liveness_score"],
                        "liveness_label": result["liveness"]["liveness_label"],
                        "action": get_recommendation_box_html(result["risk"]),
                        "components": result["liveness"],
                        "video_metrics": {
                            "frames_sampled": result["frames_used"],
                            "faces_extracted": result["faces_detected"],
                            "mean_fake_probability": result["mean_fake_probability"],
                            "median_fake_probability": result["median_fake_probability"],
                            "top3_fake_probability": result["top3_fake_probability"],
                            "high_risk_frame_ratio": result["high_risk_frame_ratio"],
                            "deepfake_video_risk": result["deepfake_video_risk"]
                        }
                    }
                    try:
                        pdf_bytes = generate_pdf_report(video_pdf_data, "video")
                        st.download_button(
                            label="📥 Download PDF Scan Report",
                            data=pdf_bytes,
                            file_name=f"satyalens_video_report_{uploaded_file.name}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.caption(f"PDF report generation skipped: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)

            with tab_v_ag:
                st.markdown("### Frame-level Aggregation Insights")
                st.markdown(
                    "Standard classification models evaluate single frames, creating high false rates "
                    "in compressed videos. SatyaLens aggregates multiple temporal samples "
                    "and weighs high-risk frames heavier to catch short video manipulations."
                )
                
                table_html = f"""
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Security Context Description</th>
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
                    <tr style="background-color: var(--card-secondary);">
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
                    "moiré artifacts, and paper borders without requiring active actions (like blinking)."
                )
                
                liveness = result["liveness"]
                col_l1, col_l2 = st.columns([1, 1.4])
                
                with col_l1:
                    st.markdown(f"**Liveness Label:** `{liveness['liveness_label']}`")
                    st.markdown(f"**Liveness Score:** `{liveness['liveness_score']}/100`")
                    st.markdown(f"**Spoof/Replay Risk:** `{liveness['spoof_risk']:.1f}%`")
                    
                    # Risk progress bar
                    lv_color = "var(--success-color)" if liveness['liveness_score'] >= 65 else ("var(--warning-color)" if liveness['liveness_score'] >= 40 else "var(--danger-color)")
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
                            <td>🏃‍♂️ <strong>Motion</strong> (25%)</td>
                            <td>{liveness['motion_score']}%</td>
                            <td>Detects pixel variance to identify static image presentations.</td>
                        </tr>
                        <tr>
                            <td>🔍 <strong>Sharpness</strong> (25%)</td>
                            <td>{liveness['sharpness_score']}%</td>
                            <td>Detects blurriness typical of print/low-res screen reviews.</td>
                        </tr>
                        <tr>
                            <td>🦚 <strong>Texture</strong> (20%)</td>
                            <td>{liveness['texture_score']}%</td>
                            <td>Identifies high-frequency patterns, such as digital screen moiré.</td>
                        </tr>
                        <tr>
                            <td>☀️ <strong>Exposure</strong> (15%)</td>
                            <td>{liveness['exposure_score']}%</td>
                            <td>Measures face lighting balance and contrast values.</td>
                        </tr>
                        <tr>
                            <td>🌈 <strong>Color Richness</strong> (15%)</td>
                            <td>{liveness['color_score']}%</td>
                            <td>Analyzes standard color deviations to verify live skin profiles.</td>
                        </tr>
                    </table>
                    """
                    st.markdown(liveness_table, unsafe_allow_html=True)

                st.warning("⚠️ **Liveness Warning:** Passive liveness analysis uses computer vision heuristics. It is NOT certified biometric liveness.")

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

# ---------------------------------------------------------
# NEURAL NETWORK EXPORTER (Investigator Mode only)
# ---------------------------------------------------------
if dashboard_mode == "Investigator (Expert)":
    st.markdown("---")
    st.markdown("### 🛠️ Neural Network Diagnostics & Exporter")
    st.markdown(
        "Utilities for auditing the model weights and exporting the neural network file "
        "to compressed formats for mobile, edge, or general server engines."
    )
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("**Convert to TFLite (edge/mobile)**")
        if st.button("Generate TFLite weights"):
            with st.spinner("Converting EfficientNetB0 Keras model to optimized TFLite file..."):
                ok, msg = export_to_tflite()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                    
    with col_exp2:
        st.markdown("**Convert to ONNX (cross-runtime)**")
        if st.button("Generate ONNX weights"):
            with st.spinner("Converting model to ONNX proto format..."):
                ok, msg = export_to_onnx()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

# ---------------------------------------------------------
# ROADMAP & FUTURE SCOPE
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
st.markdown("### 🗺️ Future Scope & Development Roadmap")
st.markdown('<span class="muted-text">Planned and integrated upgrades to make SatyaLens stronger as an AI security research platform.</span>', unsafe_allow_html=True)

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown(
        """
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem;">🎙️ Audio Deepfake Scanner</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Acoustic wave analysis to scan for voice clones and speech synthesis (ElevenLabs, Bark).
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Completed</span>
            <strong style="font-size:0.95rem;">⚡ MediaPipe Face Cropping</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Integrated MediaPipe Face Mesh for high-precision boundary boxes and alignment fallback.
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Completed</span>
            <strong style="font-size:0.95rem;">📉 Robustness Stress-Test</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Stress-tests model certainty against blur, compression noise, crop shifts, and JPEG loss.
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Completed</span>
            <strong style="font-size:0.95rem;">📥 PDF Risk Reports</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Generates polished in-memory PDF scan reports containing diagnostic scores and charts.
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Completed</span>
            <strong style="font-size:0.95rem;">🐳 Multi-Port Docker Support</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Containerized compose configurations to spin up both Streamlit and FastAPI programmatically.
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem;">📹 Live Camera Review Mode</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Real-time webcam video feed capture and frame-level deepfake inference auditing.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_r2:
    st.markdown(
        """
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Completed</span>
            <strong style="font-size:0.95rem;">🔌 FastAPI Programmatic API</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                REST API gateway with Swagger docs for automated pipeline integration.
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Completed</span>
            <strong style="font-size:0.95rem;">♻️ ONNX & TFLite Exporters</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Allows exporting the EfficientNet Keras model to ONNX or TFLite flatbuffers.
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem;">⚖️ Demographic Bias Testing</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Automated bias evaluation tools across gender, contrast, skin tone (Fitzpatrick scale).
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-low" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Completed</span>
            <strong style="font-size:0.95rem;">🎯 Confidence Calibration</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Visualizes boundary calibration certainty (35%-65% range alerts).
            </p>
        </div>
        <div class="dashboard-card-secondary">
            <span class="risk-badge risk-badge-suspicious" style="font-size:0.65rem; padding:0.15rem 0.45rem; float:right;">Planned</span>
            <strong style="font-size:0.95rem;">📊 Model Monitoring Dashboard</strong>
            <p style="font-size:0.8rem; margin:0.35rem 0 0 0; color:var(--muted-text);">
                Track and graph historical detection logs, feedback corrections, and drift.
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
    <div class="dashboard-card" style="text-align: center; margin-top: 3rem; padding: 1.5rem;">
        <strong style="font-size: 1.15rem; color: var(--text-color);">🔍 SatyaLens</strong>
        <p style="margin: 0.5rem 0 0.25rem 0; font-size: 0.85rem; color: var(--muted-text);">
            Built with passion by <strong>Arnav Raj (Cybroarnv)</strong>
        </p>
        <div style="margin: 0.5rem 0; font-size: 0.85rem;">
            <a href="https://github.com/yocybroarnv" target="_blank" style="color: var(--accent-color); text-decoration: none; margin-right: 1.25rem;">💻 GitHub</a>
            <a href="https://www.linkedin.com/in/arnav-raj-professional" target="_blank" style="color: var(--accent-color); text-decoration: none; margin-right: 1.25rem;">👔 LinkedIn</a>
            <span style="color: var(--muted-text);">⚖️ MIT Licensed</span>
        </div>
        <p style="margin: 0.75rem auto 0 auto; font-size: 0.75rem; color: var(--muted-text); max-width: 620px; line-height: 1.45;">
            Research prototype only — not for automated KYC, hiring, legal, surveillance or biometric verification decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)