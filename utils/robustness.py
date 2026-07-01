import cv2
import numpy as np

def run_robustness_test(rgb_image, model, face_detector):
    """
    Applies multiple visual distortions to the face crop and runs inference.
    Returns: A dictionary mapping distortion name to its model score and image.
    """
    # Isolate face from original image
    face, detected, _ = face_detector.detect_and_crop(rgb_image)
    h, w, c = face.shape
    
    results = {}
    
    # Helper to resize and predict
    def get_pred(face_img):
        try:
            resized = cv2.resize(face_img, (224, 224))
            arr = resized.astype("float32")
            arr = np.expand_dims(arr, axis=0)
            return float(model.predict(arr, verbose=0)[0][0])
        except Exception as e:
            print(f"Prediction error in robustness check: {e}")
            return 0.5

    # 1. Baseline
    baseline = get_pred(face)
    results["Original"] = {"score": baseline, "image": face}
    
    # 2. Gaussian Blur (Simulating camera lens defocus)
    ksize = 15 if h > 50 else 5
    blurred = cv2.GaussianBlur(face, (ksize, ksize), 0)
    results["Gaussian Blur"] = {"score": get_pred(blurred), "image": blurred}
    
    # 3. Low Light (Simulating poor illumination)
    dark = np.clip(face.astype(np.float32) * 0.4, 0, 255).astype(np.uint8)
    results["Low Light"] = {"score": get_pred(dark), "image": dark}
    
    # 4. Gaussian Noise (Simulating sensor noise in low-end hardware)
    np.random.seed(42) # Ensure deterministic test output
    noise = np.random.normal(0, 15, (h, w, c))
    noisy = np.clip(face.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    results["Gaussian Noise"] = {"score": get_pred(noisy), "image": noisy}
    
    # 5. JPEG Compression (Simulating social media upload compression)
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
        _, encimg = cv2.imencode('.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR), encode_param)
        decimg = cv2.imdecode(encimg, 1)
        compressed = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    except Exception:
        compressed = face
    results["JPEG Compression (Q=15)"] = {"score": get_pred(compressed), "image": compressed}
    
    # 6. Center Crop (Simulating alignment offsets or close-up cropping)
    h_start, h_end = int(h * 0.1), int(h * 0.9)
    w_start, w_end = int(w * 0.1), int(w * 0.9)
    cropped = face[h_start:h_end, w_start:w_end]
    if cropped.size > 0:
        cropped_resized = cv2.resize(cropped, (w, h))
    else:
        cropped_resized = face
    results["Center Crop (80%)"] = {"score": get_pred(cropped_resized), "image": cropped_resized}
    
    return results
