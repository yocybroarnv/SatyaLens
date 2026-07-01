import os
import cv2
import tempfile
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Import local face detector and audio analyzer
from utils.face_detection import FaceDetector
from utils.audio_analysis import analyze_audio

# Disable GPU warning spam for cleaner log outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ---------------------------------------------------------
app = FastAPI(
    title="SatyaLens API Gateway",
    version="v7.0",
    description="REST API backend service for deepfake detection, video aggregation and acoustic liveness checks."
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# LIFESPAN AND SYSTEM STATE
# ---------------------------------------------------------
MODEL_PATH = "satyalens_v6_efficientnetb0.keras"
model = None
detector = FaceDetector()

@app.on_event("startup")
def startup_load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH)
            print("[INFO] SatyaLens EfficientNetB0 Keras model loaded successfully in API startup.")
        except Exception as e:
            print(f"[ERROR] Failed to load model weights on API startup: {e}")
    else:
        print(f"[WARNING] Model file {MODEL_PATH} not found. Model endpoints will report Service Unavailable.")

# ---------------------------------------------------------
# LOCAL REPLICAS OF VIDEO ANALYSIS ALGORITHMS
# ---------------------------------------------------------
# Replicated to avoid importing app.py which triggers Streamlit page configurations
def local_passive_liveness_score(frames):
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

    liveness = (
        0.25 * motion +
        0.25 * sharpness +
        0.20 * texture +
        0.15 * exposure +
        0.15 * color
    ) * 100

    liveness = float(np.clip(liveness, 0, 100))
    spoof_risk = 100 - liveness

    if liveness >= 65:
        label = "Live-like"
    elif liveness >= 40:
        label = "Uncertain"
    else:
        label = "Possible spoof/static/replay media"

    return {
        "liveness_score": round(liveness, 2),
        "spoof_risk": round(spoof_risk, 2),
        "liveness_label": label,
        "motion_score": round(motion * 100, 2),
        "sharpness_score": round(sharpness * 100, 2),
        "texture_score": round(texture * 100, 2),
        "exposure_score": round(exposure * 100, 2),
        "color_score": round(color * 100, 2)
    }

def local_risk_category(fake_probability):
    score = fake_probability * 100
    if score < 30:
        return "Low Risk", "Normal review may continue. Do not treat result as final proof."
    elif score < 66:
        return "Suspicious", "Manual review recommended. Ask for additional verification."
    else:
        return "High Risk", "Possible deepfake or synthetic identity attempt. Manual verification required."

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "SatyaLens API Gateway",
        "version": "v7.0",
        "documentation_docs": "/docs",
        "documentation_redoc": "/redoc"
    }

@app.get("/health")
def health_check():
    global model
    return {
        "status": "healthy",
        "model_weights_configured": os.path.exists(MODEL_PATH),
        "model_loaded_successfully": (model is not None)
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Core model weights are not loaded. Check server configuration.")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        rgb = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    try:
        cropped, detected, method = detector.detect_and_crop(rgb)
        resized = cv2.resize(cropped, (224, 224))
        arr = resized.astype("float32")
        arr = np.expand_dims(arr, axis=0)
        
        prob = float(model.predict(arr, verbose=0)[0][0])
        risk, action = local_risk_category(prob)
        
        return {
            "success": True,
            "filename": file.filename,
            "face_detected": detected,
            "detection_method": method,
            "fake_probability": round(prob, 4),
            "threat_class": risk,
            "recommended_action": action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Core model weights are not loaded. Check server configuration.")
        
    suffix = file.filename.lower().split(".")[-1]
    if suffix not in ["mp4", "avi", "mov", "mkv", "webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format. Upload MP4, AVI, MOV, MKV, or WEBM.")
        
    # Save uploaded file to temp file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix)
    try:
        contents = await file.read()
        temp.write(contents)
        temp_path = temp.name
        temp.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cache uploaded video: {str(e)}")
        
    try:
        # Uniformly sample up to 16 frames
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file.")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        if total_frames > 0:
            indices = np.linspace(0, max(total_frames - 1, 0), 16).astype(int)
            for idx in sorted(set(indices.tolist())):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        # Clean up video temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
            
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="No readable frames could be extracted from this video.")
            
        # Predict frames
        probs = []
        faces_detected = 0
        for f in frames:
            cropped, detected, _ = detector.detect_and_crop(f)
            if detected:
                faces_detected += 1
            resized = cv2.resize(cropped, (224, 224))
            arr = resized.astype("float32")
            arr = np.expand_dims(arr, axis=0)
            prob = float(model.predict(arr, verbose=0)[0][0])
            probs.append(prob)
            
        probs = np.array(probs)
        mean_prob = float(np.mean(probs))
        median_prob = float(np.median(probs))
        top3_prob = float(np.mean(np.sort(probs)[-min(3, len(probs)):]))
        high_risk_ratio = float(np.mean(probs >= 0.66))

        # Combined video risk
        deepfake_risk = (
            0.50 * mean_prob +
            0.25 * median_prob +
            0.15 * top3_prob +
            0.10 * high_risk_ratio
        )

        liveness = local_passive_liveness_score(frames)
        overall = 0.70 * deepfake_risk + 0.30 * (liveness["spoof_risk"] / 100)
        risk_lbl, action = local_risk_category(overall)
        
        return {
            "success": True,
            "filename": file.filename,
            "frames_used": len(frames),
            "faces_detected": faces_detected,
            "overall_identity_risk": round(overall, 4),
            "deepfake_video_risk": round(deepfake_risk, 4),
            "threat_class": risk_lbl,
            "recommended_action": action,
            "video_metrics": {
                "mean_fake_probability": round(mean_prob, 4),
                "median_fake_probability": round(median_prob, 4),
                "top3_fake_probability": round(top3_prob, 4),
                "high_risk_frame_ratio": round(high_risk_ratio, 4)
            },
            "passive_liveness": liveness
        }
    except Exception as e:
        # Final cleanup attempt
        try:
            os.remove(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Video aggregation processing failed: {str(e)}")

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    suffix = file.filename.lower().split(".")[-1]
    if suffix not in ["wav", "mp3", "m4a", "ogg"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Upload WAV, MP3, M4A, or OGG.")
        
    # Write to a temp file
    temp = tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix)
    try:
        contents = await file.read()
        temp.write(contents)
        temp_path = temp.name
        temp.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cache uploaded audio: {str(e)}")
        
    try:
        analysis_result = analyze_audio(temp_path, file.filename)
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
            
        return analysis_result
    except Exception as e:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
