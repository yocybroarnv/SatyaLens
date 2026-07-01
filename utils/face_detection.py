import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Load Haar Cascade as fallback
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.mp_face_detection = None
        self.mp_loaded = False
        
        try:
            import mediapipe as mp
            import mediapipe.solutions.face_detection as mp_face
            self.mp_face_detection = mp_face
            self.mp_loaded = True
        except ImportError:
            pass

    def detect_and_crop(self, rgb, margin=0.25):
        """
        Attempts to detect and crop the largest face in an image.
        Returns: (cropped_rgb, success_boolean, method_name_string)
        """
        if rgb is None:
            return rgb, False, "None"

        # Try MediaPipe first if loaded
        if self.mp_loaded and self.mp_face_detection is not None:
            try:
                # Use model_selection=1 for general images (0-2 meters), min_detection_confidence=0.5
                with self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
                    # Convert to RGB if needed (MediaPipe expects RGB, which is our input format)
                    results = detector.process(rgb)
                    if results.detections:
                        h, w, _ = rgb.shape
                        largest_area = 0
                        best_bbox = None
                        
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            # Convert relative coordinates to pixels
                            x_px = int(bbox.xmin * w)
                            y_px = int(bbox.ymin * h)
                            w_px = int(bbox.width * w)
                            h_px = int(bbox.height * h)
                            
                            area = w_px * h_px
                            if area > largest_area:
                                largest_area = area
                                best_bbox = (x_px, y_px, w_px, h_px)
                        
                        if best_bbox:
                            x, y, w_box, h_box = best_bbox
                            mx = int(w_box * margin)
                            my = int(h_box * margin)
                            
                            x1 = max(0, x - mx)
                            y1 = max(0, y - my)
                            x2 = min(rgb.shape[1], x + w_box + mx)
                            y2 = min(rgb.shape[0], y + h_box + my)
                            
                            crop = rgb[y1:y2, x1:x2]
                            if crop.size > 0:
                                return crop, True, "MediaPipe Face Detection"
            except Exception:
                pass

        # Fallback to Haar Cascade if MediaPipe fails or finds no faces
        if not self.face_cascade.empty():
            try:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(faces) > 0:
                    x, y, w_box, h_box = max(faces, key=lambda box: box[2] * box[3])
                    mx = int(w_box * margin)
                    my = int(h_box * margin)
                    
                    x1 = max(0, x - mx)
                    y1 = max(0, y - my)
                    x2 = min(rgb.shape[1], x + w_box + mx)
                    y2 = min(rgb.shape[0], y + h_box + my)
                    
                    crop = rgb[y1:y2, x1:x2]
                    if crop.size > 0:
                        return crop, True, "Haar Cascade Face Detection"
            except Exception:
                pass

        return rgb, False, "None (Full image frame fallback)"
