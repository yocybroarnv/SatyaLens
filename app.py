import os
import cv2
import json
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# ---------------------------------------------------------
# CONSTANTS & SETUP
# ---------------------------------------------------------
IMG_SIZE = 224
MODEL_PATH = "satyalens_v6_efficientnetb0.keras"
METRICS_PATH = "satyalens_v6_metrics.json"

# Disable GPU warning spam for cleaner console execution
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(
    page_title="SatyaLens v7: Deepfake & Spoof Estimator",
    page_icon="🔍",
    layout="centered"
)

# ---------------------------------------------------------
# STYLING INJECTION (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Main Dashboard Cards */
    .dashboard-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    
    .dashboard-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.25rem;
        letter-spacing: -0.025em;
    }
    
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    
    /* Interactive Chips */
    .chip-container {
        margin-bottom: 1.5rem;
    }
    
    .chip {
        display: inline-block;
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.825rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .risk-badge-low {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    
    .risk-badge-suspicious {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    
    .risk-badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    
    /* Custom Progress Bar for High-Fidelity Feedback */
    .progress-container {
        margin: 0.85rem 0;
    }
    .progress-bar-bg {
        background-color: #E5E7EB;
        border-radius: 9999px;
        height: 12px;
        width: 100%;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }
    
    /* Responsive Info Tables */
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .metrics-table th {
        background-color: #F3F4F6;
        color: #374151;
        font-weight: 600;
        text-align: left;
        padding: 0.65rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .metrics-table td {
        padding: 0.65rem;
        border-bottom: 1px solid #E5E7EB;
        color: #4B5563;
    }
    
    /* Action / Recommendation Cards */
    .action-box {
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 1rem;
    }
    .action-low {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #DCFCE7;
    }
    .action-suspicious {
        background-color: #FFFDF2;
        color: #854D0E;
        border: 1px solid #FEF3C7;
    }
    .action-high {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FEE2E2;
    }
    
    /* Sidebar Specific Styling */
    .sidebar-section {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .sidebar-metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .sidebar-metric-card {
        background-color: #F9FAFB;
        border: 1px solid #F3F4F6;
        border-radius: 6px;
        padding: 0.5rem;
        text-align: center;
    }
    .sidebar-metric-val {
        font-size: 1rem;
        font-weight: 700;
        color: #1F2937;
    }
    .sidebar-metric-lbl {
        font-size: 0.7rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL & METRICS LOADER
# ---------------------------------------------------------
@st.cache_resource
def load_deepfake_model():
    """Loads the EfficientNetB0 Keras model weights with resource caching."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} is missing.")
    return keras.models.load_model(MODEL_PATH)

# Check model file existence first
model_exists = os.path.exists(MODEL_PATH)

# Load metrics configuration if present
metrics_data = None
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
    except Exception:
        pass

# ---------------------------------------------------------
# CV2 FACE CROPPING PIPELINE
# ---------------------------------------------------------
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def crop_largest_face_rgb(rgb, margin=0.25):
    """
    Detects and crops the largest face in an image.
    Returns: (cropped_image, face_detected_boolean)
    """
    if rgb is None or face_cascade.empty():
        return rgb, False

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        return rgb, False

    # Choose largest face box by area
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # Crop with margin padding
    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(rgb.shape[1], x + w + mx)
    y2 = min(rgb.shape[0], y + h + my)

    crop = rgb[y1:y2, x1:x2]

    if crop.size == 0:
        return rgb, False

    return crop, True

# ---------------------------------------------------------
# RISK CATEGORIZATION ENGINE
# ---------------------------------------------------------
def estimate_risk_category(fake_probability):
    """Maps fake probability score to risk level labels."""
    score = fake_probability * 100
    if score < 30.0:
        return "Low Risk", "Normal review may continue. Do not treat result as final proof."
    elif score < 66.0:
        return "Suspicious", "Manual review recommended. Ask for additional verification."
    else:
        return "High Risk", "Possible deepfake or synthetic identity attempt. Strong manual verification required."

# Helper functions for badge HTML rendering
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
            <strong>Recommended Action:</strong> Normal verification review may continue. The media did not present significant synthetic/manipulated indicators. Do not treat this automated result as definitive proof of authenticity.
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

# ---------------------------------------------------------
# IMAGE PREDICTION
# ---------------------------------------------------------
def predict_rgb(rgb, model):
    """Runs prediction on a face cropped from the raw image."""
    cropped, detected = crop_largest_face_rgb(rgb)
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    arr = resized.astype("float32")
    arr = np.expand_dims(arr, axis=0)
    
    # Run model prediction
    prediction_raw = model.predict(arr, verbose=0)
    fake_prob = float(prediction_raw[0][0])
    return fake_prob, cropped, detected

# ---------------------------------------------------------
# VIDEO TIMELINE SAMPLING & AGGREGATION
# ---------------------------------------------------------
def read_video_frames(video_path, max_frames=16):
    """Uniformly extracts frames from the video path."""
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

def calculate_passive_liveness_heuristics(frames):
    """Estimates fake-spoof heuristic components for video frames."""
    if len(frames) == 0:
        return {"liveness_score": 0.0, "spoof_risk": 100.0, "liveness_label": "Unknown"}

    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    # 1. Sharpness: Laplace variance normalizer
    sharpness = float(np.clip(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]) / 180.0, 0, 1))

    # 2. Texture: Canny edge density normalizer
    texture = float(np.clip(
        np.mean([np.mean(cv2.Canny(g, 80, 160) > 0) for g in grays]) / 0.12,
        0,
        1
    ))

    # 3. Exposure: Deviation from optimal exposure
    brightness = float(np.mean([np.mean(g) for g in grays]))
    exposure = float(1 - np.clip(abs(brightness - 127.5) / 127.5, 0, 1))

    # 4. Motion: Inter-frame change variance normalizer
    if len(grays) > 1:
        small = [cv2.resize(g, (96, 96)) for g in grays]
        diffs = [np.mean(cv2.absdiff(a, b)) for a, b in zip(small[:-1], small[1:])]
        motion = float(np.clip(np.mean(diffs) / 18.0, 0, 1))
    else:
        motion = 0.0

    # 5. Color richness: RGB standard deviation normalizer
    color = float(np.clip(np.mean([np.mean(np.std(f, axis=(0, 1))) for f in frames]) / 55.0, 0, 1))

    # Aggregate heuristics
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

def predict_video_aggregate(path, model):
    """Processes video, predicts each frame, and aggregates score with liveness."""
    frames = read_video_frames(path, 16)
    if len(frames) == 0:
        return None, []

    probs = []
    detected_faces_count = 0
    for f in frames:
        prob, _, detected = predict_rgb(f, model)
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

    # Heuristic liveness spoof calculations
    liveness = calculate_passive_liveness_heuristics(frames)
    
    # Overall risk integrates 70% model classification + 30% heuristic spoof risk
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
# GRAD-CAM EXPLAINABILITY
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

def make_gradcam_overlay(rgb, model, alpha=0.45):
    """Generates Grad-CAM activation heatmap overlay."""
    try:
        base = get_nested_base_model(model)
        if base is None:
            return None

        last_layer_name = get_last_conv_layer_name(base)
        if last_layer_name is None:
            return None

        rgb_crop, _ = crop_largest_face_rgb(rgb)
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
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown('<h2 style="margin-top: 0;">🔍 SatyaLens v7</h2>', unsafe_allow_html=True)
st.sidebar.caption("AI Security Deepfake Detection & Identity-Risk Estimator")

# 1. Model status indicator
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
if model_exists:
    st.sidebar.markdown('<span class="risk-badge risk-badge-low" style="padding: 0.15rem 0.5rem; font-size:0.75rem;">🟢 Model Loaded (Ready)</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"Weights file: `{MODEL_PATH}`")
else:
    st.sidebar.markdown('<span class="risk-badge risk-badge-high" style="padding: 0.15rem 0.5rem; font-size:0.75rem;">🔴 Model Weights Missing</span>', unsafe_allow_html=True)
    st.sidebar.markdown("Place the weights file in the app directory. Refer to [MODEL_DOWNLOAD.md](file:///c:/Users/arnav/Desktop/SatyaLens/MODEL_DOWNLOAD.md).")

# 2. Performance metrics from JSON
if metrics_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Evaluation Metrics")
    st.sidebar.markdown("Calculated from validation benchmarks:")
    
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
    st.sidebar.markdown("---")
    st.sidebar.warning("⚠️ Metrics metadata file (`satyalens_v6_metrics.json`) is missing.")

# 3. Threshold guidelines
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Risk Guidelines")
st.sidebar.markdown(
    """
    - **0 - 29%**: <span style="color:#16A34A; font-weight:bold;">Low Risk</span> (Proceed)
    - **30 - 66%**: <span style="color:#D97706; font-weight:bold;">Suspicious</span> (Manual Review)
    - **67 - 100%**: <span style="color:#DC2626; font-weight:bold;">High Risk</span> (Strong Verification)
    """,
    unsafe_allow_html=True
)

# 4. Supported uploads
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Supported Formats")
st.sidebar.markdown(
    """
    - **Images**: JPG, JPEG, PNG, WEBP
    - **Videos**: MP4, AVI, MOV, MKV, WEBM
    """
)

# 5. Ethical use warning
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚖️ Responsible AI Use")
st.sidebar.caption(
    "This tool is designed to support manual review workflows and threat research. "
    "It should not be used as an automated gating, screening, or final decision tool for "
    "KYC onboarding, hiring rejection, or legal cases."
)

# ---------------------------------------------------------
# MAIN APP BODY
# ---------------------------------------------------------

# Landing Section
st.markdown('<div class="dashboard-header">🔍 SatyaLens v7</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">AI Security-Focused Deepfake Detection & Identity-Risk Estimator</div>', unsafe_allow_html=True)

# Interactive use-case chips
st.markdown(
    """
    <div class="chip-container">
        <span class="chip">KYC Fraud Research</span>
        <span class="chip">Remote Hiring Vetting</span>
        <span class="chip">Synthetic Identity Detection</span>
        <span class="chip">Cybercrime Awareness</span>
        <span class="chip">AI Security Portfolio</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Disclaimer Card
st.markdown(
    """
    <div class="dashboard-card" style="border-left: 4px solid #F59E0B; background-color: #FFFDF5;">
        <strong style="color: #B45309;">⚠️ Research Prototype Notice:</strong>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem; color: #4B5563; line-height: 1.4;">
            SatyaLens provides statistical deepfake and passive spoof risk signals for decision support. 
            It is NOT a certified biometric liveness tool and should NOT be used for automated rejections. 
            Always escalate suspicious media to human review.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load model weights
model = None
if model_exists:
    try:
        model = load_deepfake_model()
    except Exception as e:
        st.error(f"Failed to load the model file. It may be corrupted or incompatible: {e}")
        st.stop()
else:
    st.markdown(
        f"""
        <div class="dashboard-card" style="border-left: 4px solid #DC2626; background-color: #FEF2F2;">
            <strong style="color: #B91C1C;">🔴 Core Model Weights Missing</strong>
            <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #4B5563;">
                The deep learning model file <strong><code>{MODEL_PATH}</code></strong> is not found in the project root directory.
            </p>
            <p style="margin: 0; font-size: 0.9rem; color: #4B5563;">
                Please refer to the <strong><a href="file:///c:/Users/arnav/Desktop/SatyaLens/MODEL_DOWNLOAD.md" target="_blank" style="color: #2563EB; text-decoration: underline;">MODEL_DOWNLOAD.md</a></strong> guide to resolve this and restart the application.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

# Drag-and-Drop Uploader
st.markdown("### 📥 Media Upload")
uploaded_file = st.file_uploader(
    "Upload a face image or a short video clip to scan for manipulation features:",
    type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv", "webm"]
)

# Processing uploads
if uploaded_file:
    file_suffix = uploaded_file.name.lower().split(".")[-1]
    is_image = file_suffix in ["jpg", "jpeg", "png", "webp"]

    if is_image:
        # IMAGE WORKFLOW
        st.markdown("---")
        st.markdown("### 🖼️ Image Scanning Workspace")
        
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            raw_rgb = np.array(pil_image)
        except Exception as e:
            st.error(f"Could not read image file. It may be corrupted: {e}")
            st.stop()
        
        with st.spinner("Isolating face features and running model inference..."):
            try:
                fake_prob, cropped_face, face_found = predict_rgb(raw_rgb, model)
                risk_lvl, recommendation = estimate_risk_category(fake_prob)
            except Exception as e:
                st.error(f"Inference pipeline error: {e}")
                st.stop()

        # Layout prediction results
        c1, c2 = st.columns([1, 1.2])

        with c1:
            st.markdown('<div class="dashboard-card" style="text-align: center;">', unsafe_allow_html=True)
            st.image(
                cropped_face, 
                caption="Extracted Face (Model Input)" if face_found else "Raw Image (Face Cropping Failed)", 
                use_container_width=True
            )
            if not face_found:
                st.warning("⚠️ No face was isolated. Processing was executed on the full image frame.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("#### Threat Assessment")
            
            # Badge Row
            pred_lbl = "Fake" if fake_prob >= 0.5 else "Real"
            badge_html = f"""
            <div style="margin-bottom: 0.75rem;">
                <span style="font-weight: 600; color: #4B5563; margin-right: 0.5rem;">Classification:</span> 
                {get_pred_badge_html(pred_lbl)}
            </div>
            <div style="margin-bottom: 0.75rem;">
                <span style="font-weight: 600; color: #4B5563; margin-right: 0.5rem;">Threat Class:</span> 
                {get_risk_badge_html(risk_lvl)}
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)

            # Fake probability score + progress bar
            pb_color = "#16A34A" if fake_prob < 0.30 else ("#F59E0B" if fake_prob < 0.66 else "#DC2626")
            st.markdown(f"**Synthetic Probability:** `{fake_prob:.2%}`")
            st.markdown(get_progress_bar_html(fake_prob * 100, pb_color), unsafe_allow_html=True)

            # Recommendation Box
            st.markdown(get_recommendation_box_html(risk_lvl), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.warning("⚠️ **Image-only scan limits liveness proof.** A static image cannot verify physical motion. Use a short video clip for passive liveness and screen-replay verification.")

        # Explainability Block (Grad-CAM)
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Model Attention Visualizer (Grad-CAM)")
        st.markdown(
            "Grad-CAM computes activation gradients on the model's last convolutional layer to show "
            "where the network focused its attention. High-focus areas are in red, while cold areas are in blue."
        )

        with st.spinner("Generating Grad-CAM overlays..."):
            overlay_img = make_gradcam_overlay(raw_rgb, model)

        if overlay_img is not None:
            st.image(overlay_img, caption="Grad-CAM Hotspot Overlay", use_container_width=True)
            st.markdown(
                """
                <p style="font-size: 0.85rem; color: #6B7280; margin-top: 0.5rem; line-height: 1.4;">
                    💡 <strong>Analysis Tip:</strong> Fake generators often struggle with realistic boundaries, nose-bridge highlights, 
                    ear asymmetries, and hair textures. Check if the red highlights focus on boundary artifacts or facial borders.
                </p>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Grad-CAM could not be generated. Image prediction details are still fully available.")

        st.caption("Grad-CAM outputs are interpretability support metrics, not forensic proof.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # VIDEO WORKFLOW
        st.markdown("---")
        st.markdown("### 🎥 Video Scanning Workspace")
        
        # Save video to temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix="." + file_suffix)
        try:
            temp_video.write(uploaded_file.read())
            temp_video_name = temp_video.name
            temp_video.close()
        except Exception as e:
            st.error(f"Failed to create temporary file for video processing: {e}")
            st.stop()

        with st.spinner("Sampling video frames and running prediction arrays..."):
            result, frames = predict_video_aggregate(temp_video_name, model)

        # Remove temporary video file right after processing to prevent leaks
        try:
            os.remove(temp_video_name)
        except Exception:
            pass

        if result is None or len(frames) == 0:
            st.error("Could not read video frames. Please check if the video is corrupted or encoded in an unsupported format.")
            st.stop()

        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("#### Aggregated Video Threat Scan")
        
        col_vid_1, col_vid_2 = st.columns([1, 1.2])

        with col_vid_1:
            st.video(uploaded_file)
            st.caption("Uploaded video source")

        with col_vid_2:
            # Main risk badge
            st.markdown(f"**Threat Assessment:** {get_risk_badge_html(result['risk'])}", unsafe_allow_html=True)
            
            # Big metrics blocks
            st.markdown("##### Threat Scoreboard")
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Overall Identity Risk", f"{result['overall_identity_risk']:.1%}")
            c_m2.metric("Deepfake Video Risk", f"{result['deepfake_video_risk']:.1%}")
            c_m3.metric("Liveness Score", f"{result['liveness']['liveness_score']}/100")
            
            # Recommendation
            st.markdown(get_recommendation_box_html(result['risk']), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Tabbed insights
        tab_agg, tab_live, tab_frames = st.tabs([
            "📊 Video Aggregation Metrics", 
            "⚡ Heuristic Passive Liveness", 
            "🖼️ Sampled Frame Details"
        ])

        with tab_agg:
            st.markdown("### Frame-level Aggregation Insights")
            st.markdown(
                "Aggregating predictions across multiple frames prevents false positives from single-frame lighting glitches. "
                "The overall deepfake risk combines statistical metrics calculated from the sampled frames."
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
                    <td>Average of top 3 high-risk frames. Captures short deepfake splice insertions.</td>
                </tr>
                <tr>
                    <td><strong>High-Risk Frame Ratio</strong></td>
                    <td>{result['high_risk_frame_ratio']:.2%}</td>
                    <td>Ratio of frames exceeding the 66% suspicious threat threshold.</td>
                </tr>
                <tr style="background-color: #F9FAFB;">
                    <td><strong>Deepfake Video Risk</strong></td>
                    <td><strong>{result['deepfake_video_risk']:.2%}</strong></td>
                    <td>Weighted combination: 50% Mean + 25% Median + 15% Top-3 + 10% Ratio.</td>
                </tr>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

        with tab_live:
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
                lv_color = "#16A34A" if liveness['liveness_score'] >= 65 else ("#F59E0B" if liveness['liveness_score'] >= 40 else "#DC2626")
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

            st.info("⚠️ **Liveness Warning:** Passive liveness analysis uses computer vision heuristics. It is NOT certified biometric check and sophisticated replays may bypass it. Do not rely on it as a stand-alone defense.")

        with tab_frames:
            st.markdown("### Sampled Video Timeline Frames")
            st.markdown("Review the extracted frames used for deepfake model predictions below:")
            
            # Show up to 8 frames in 4 columns
            max_thumbs = min(8, len(frames))
            grid_cols_count = min(4, max_thumbs)
            
            rows = [frames[i:i + grid_cols_count] for i in range(0, max_thumbs, grid_cols_count)]
            for r_idx, row in enumerate(rows):
                grid_cols = st.columns(grid_cols_count)
                for c_idx, frame_rgb in enumerate(row):
                    global_idx = r_idx * grid_cols_count + c_idx
                    grid_cols[c_idx].image(frame_rgb, caption=f"Frame {global_idx+1}", use_container_width=True)

# ---------------------------------------------------------
# TECHNICAL FOOTER & EXPLANATION
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📖 Methodologies & Dataset Limitations")

col_foot_1, col_foot_2 = st.columns(2)

with col_foot_1:
    st.markdown("#### Deepfake Video Aggregation")
    st.markdown(
        "Standard classification models evaluate single frames, creating high false rates "
        "in compressed videos. SatyaLens v7 aggregates multiple temporal samples "
        "and weighs high-risk frames heavier to catch short video manipulations."
    )
    
    st.markdown("#### Grad-CAM Explainability")
    st.markdown(
        "By tracing gradients back to the final convolutional layer of the EfficientNet base model, "
        "Grad-CAM visualizes what visual patterns (e.g. eye alignment, jawline boundaries) "
        "the model highlighted. This guides reviews to find forensic inconsistencies manually."
    )

with col_foot_2:
    st.markdown("#### Passive Liveness Heuristics")
    st.markdown(
        "Heuristic analysis scores pixel features like edge gradients, color channel statistics, "
        "and Laplace variance to identify flat paper printouts, tablet screen plays, and low-res "
        "manipulated media. This supports human review but does not substitute biometric certification."
    )
    
    st.markdown("#### Ethical Use Boundaries")
    st.markdown(
        "SatyaLens is a research tool. It should not be integrated into automated gatekeepers "
        "or background checks without manual override paths, as models can produce false positives "
        "under compressed streams, poor illumination, or unseen synthesis styles."
    )

st.divider()
st.caption(
    "SatyaLens v7 Research Prototype • Supported by TensorFlow, EfficientNetB0, and Streamlit. "
    "Not for production KYC, automated hiring, or legal verification."
)