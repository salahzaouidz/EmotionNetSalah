# app.py
import os
import cv2
import numpy as np
import time
import base64
import traceback
from io import BytesIO
from collections import deque, Counter
import logging
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB for multiple files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/temp', exist_ok=True)

# Model configurations
MODELS_CONFIG = {
    'model1': {
        'name': 'EmotionNetSalah V1',
        'path': 'bestModels/ModelV1.keras',
        'accuracy': '67.78%',
        'description': 'CNN trained on FER2013 with data augmentation',
        'emoji': '🧠'
    },
    'model2': {
        'name': 'EmotionNetSalah V2',
        'path': 'bestModels/ModelV2.keras',
        'accuracy': '69.18%',
        'description': 'Fine-tuned with 3-phase training',
        'emoji': '🤖'
    }
}

# Load models
loaded_models = {}
for model_key, model_config in MODELS_CONFIG.items():
    try:
        model_path = model_config['path']
        if not os.path.exists(model_path):
            logger.error(f"✗ Model not found: {model_path}")
            loaded_models[model_key] = {'model': None, 'config': model_config}
            continue
        
        logger.info(f"Loading {model_config['name']}...")
        try:
            model = load_model(model_path, compile=False)
        except:
            model = load_model(model_path, compile=True)
        
        # Warm up
        dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        
        loaded_models[model_key] = {'model': model, 'config': model_config}
        logger.info(f"✓ {model_config['name']} loaded")
    except Exception as e:
        logger.error(f"✗ {model_config['name']}: {e}")
        loaded_models[model_key] = {'model': None, 'config': model_config}

# Emotion mapping
EMOTION_MAP = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

EMOTION_COLORS_BGR = {
    'Angry': (68, 68, 255), 'Disgust': (65, 232, 98), 'Fear': (215, 10, 215),
    'Happy': (0, 215, 255), 'Sad': (225, 105, 65), 'Surprise': (10, 135, 237),
    'Neutral': (203, 203, 210)
}

EMOTION_COLORS_HEX = {
    'Angry': '#FF4444', 'Disgust': '#62E841', 'Fear': '#D70AD7',
    'Happy': '#FFD700', 'Sad': '#4169E1', 'Surprise': '#ED870A', 'Neutral': '#D2CBCB'
}

EMOTION_EMOJIS = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊',
    'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'
}

# Initialize face cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class AdvancedFaceDetector:
    """Professional-grade face detector with multiple strategies"""
    
    def __init__(self):
        self.cascades = [face_cascade, profile_cascade]
        self.last_faces = []
        self.last_detection_frame = -5
        self.detection_interval = 3
        
    def detect_faces_advanced(self, gray_frame, frame_count=0):
        """Multi-cascade face detection with confidence scoring"""
        faces_with_confidence = []
        
        # Strategy 1: Frontal face with multiple scales
        for cascade in self.cascades:
            if cascade.empty():
                continue
                
            for scale in [1.05, 1.1, 1.15]:
                for min_neighbors in [6, 5, 4]:
                    faces = cascade.detectMultiScale(
                        gray_frame,
                        scaleFactor=scale,
                        minNeighbors=min_neighbors,
                        minSize=(40, 40),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    for (x, y, w, h) in faces:
                        # Calculate confidence based on detection parameters
                        confidence = min(0.95, 0.6 + (min_neighbors - 3) * 0.1 + (1.2 - scale) * 0.5)
                        faces_with_confidence.append({
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'confidence': confidence,
                            'method': 'haar'
                        })
        
        # Strategy 2: Eye detection to validate faces
        validated_faces = []
        for face in faces_with_confidence:
            x, y, w, h = face['bbox']
            face_roi = gray_frame[max(0, y):min(gray_frame.shape[0], y+h),
                                  max(0, x):min(gray_frame.shape[1], x+w)]
            
            if face_roi.size > 0:
                eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(10, 10))
                if len(eyes) >= 1:
                    face['confidence'] += 0.1
                    face['method'] = 'haar+eyes'
                validated_faces.append(face)
        
        if not validated_faces:
            return []
        
        # Non-maximum suppression to remove overlapping detections
        final_faces = self._non_max_suppression(validated_faces)
        
        return final_faces
    
    def _non_max_suppression(self, faces, iou_threshold=0.3):
        """Remove overlapping face detections keeping highest confidence"""
        if len(faces) <= 1:
            return faces
        
        # Sort by confidence (descending) and area (descending)
        faces = sorted(faces, key=lambda f: (f['confidence'], f['bbox'][2] * f['bbox'][3]), reverse=True)
        
        kept = []
        for face in faces:
            x1, y1, w1, h1 = face['bbox']
            overlap = False
            
            for kept_face in kept:
                x2, y2, w2, h2 = kept_face['bbox']
                
                # Calculate IoU
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                
                if xi2 > xi1 and yi2 > yi1:
                    inter_area = (xi2 - xi1) * (yi2 - yi1)
                    union_area = (w1 * h1) + (w2 * h2) - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > iou_threshold:
                        overlap = True
                        break
            
            if not overlap:
                kept.append(face)
        
        return kept
    
    def detect(self, gray_frame, frame_count=0):
        """Main detection method with frame skipping"""
        if frame_count - self.last_detection_frame >= self.detection_interval:
            faces = self.detect_faces_advanced(gray_frame, frame_count)
            if faces:
                self.last_faces = faces
            self.last_detection_frame = frame_count
        else:
            faces = self.last_faces
        
        return faces

class MultiFaceEmotionPredictor:
    """Predict emotions for multiple faces simultaneously"""
    
    def __init__(self, model):
        self.model = model
        self.emotion_histories = {}  # Per-face emotion history
        self.face_id_counter = 0
        self.tracked_faces = {}  # Face tracking by position
        
    def preprocess_face(self, face_roi):
        """Preprocess face for model input"""
        if face_roi is None or face_roi.size == 0:
            return None
        
        # Enhance contrast
        face_eq = cv2.equalizeHist(face_roi)
        
        # Resize to 48x48
        face_resized = cv2.resize(face_eq, (48, 48))
        
        # Normalize
        face_norm = face_resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        return np.expand_dims(face_norm, axis=(0, -1))
    
    def get_face_id(self, bbox):
        """Track face by position to maintain history"""
        x, y, w, h = bbox
        center = (x + w//2, y + h//2)
        
        # Find closest tracked face
        best_id = None
        best_dist = float('inf')
        
        for face_id, tracked in self.tracked_faces.items():
            tx, ty = tracked['center']
            dist = np.sqrt((center[0] - tx)**2 + (center[1] - ty)**2)
            
            if dist < max(w, h) * 0.5 and dist < best_dist:
                best_dist = dist
                best_id = face_id
        
        if best_id is None:
            best_id = self.face_id_counter
            self.face_id_counter += 1
            self.emotion_histories[best_id] = deque(maxlen=15)
        
        # Update tracking
        self.tracked_faces[best_id] = {
            'center': center,
            'bbox': bbox,
            'last_seen': time.time()
        }
        
        # Clean old tracks
        current_time = time.time()
        expired = [fid for fid, data in self.tracked_faces.items() 
                   if current_time - data['last_seen'] > 2.0]
        for fid in expired:
            del self.tracked_faces[fid]
            if fid in self.emotion_histories:
                del self.emotion_histories[fid]
        
        return best_id
    
    def predict_face(self, face_input, face_id):
        """Predict emotion for a single face with history smoothing"""
        if face_input is None:
            return None
        
        predictions = self.model.predict(face_input, verbose=0)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        results = []
        for idx in top_indices:
            idx_int = int(idx)
            results.append({
                'emotion': EMOTION_MAP[idx_int],
                'probability': float(predictions[idx_int]),
                'color_hex': EMOTION_COLORS_HEX[EMOTION_MAP[idx_int]],
                'color_bgr': EMOTION_COLORS_BGR[EMOTION_MAP[idx_int]],
                'emoji': EMOTION_EMOJIS[EMOTION_MAP[idx_int]]
            })
        
        # Smooth top emotion using history
        if face_id in self.emotion_histories:
            self.emotion_histories[face_id].append(results[0]['emotion'])
            
            # Weighted majority vote (recent = higher weight)
            history = list(self.emotion_histories[face_id])
            emotion_counts = Counter(history)
            most_common = emotion_counts.most_common(1)[0]
            
            # Apply smoothing only if history is consistent
            consistency = emotion_counts[most_common[0]] / len(history)
            if consistency > 0.4:  # 40% consistency threshold
                results[0]['emotion'] = most_common[0]
                results[0]['probability'] = min(1.0, results[0]['probability'] * (0.7 + 0.3 * consistency))
        
        return results

class HighPerformanceVideoProcessor:
    """Production-grade video processor like professional detectors"""
    
    def __init__(self, model):
        self.face_detector = AdvancedFaceDetector()
        self.emotion_predictor = MultiFaceEmotionPredictor(model)
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        self.frame_count = 0
        self.current_results = []
        self.processing_resolution = 0.65  # Process at 65% resolution
        
    def process_frame(self, frame):
        """Process frame with professional-grade detection"""
        h, w = frame.shape[:2]
        self.frame_count += 1
        
        # Resize for faster processing while maintaining accuracy
        small_h = max(100, int(h * self.processing_resolution))
        small_w = max(100, int(w * self.processing_resolution))
        small_frame = cv2.resize(frame, (small_w, small_h))
        
        # Convert to grayscale and enhance
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Apply bilateral filter for noise reduction while preserving edges
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Detect faces with advanced detector
        detected_faces = self.face_detector.detect(gray, self.frame_count)
        
        results = []
        
        for face_data in detected_faces:
            x, y, fw, fh = face_data['bbox']
            confidence = face_data['confidence']
            
            # Scale coordinates back to original frame
            scale_x = w / small_w
            scale_y = h / small_h
            
            fx = int(x * scale_x)
            fy = int(y * scale_y)
            ffw = int(fw * scale_x)
            ffh = int(fh * scale_y)
            
            # Add padding (more for high-confidence detections)
            pad_pct = 0.08 + confidence * 0.07  # 8-15% padding
            pad_x = int(ffw * pad_pct)
            pad_y = int(ffh * pad_pct)
            
            fx = max(0, fx - pad_x)
            fy = max(0, fy - pad_y)
            ffw = min(w - fx, ffw + 2 * pad_x)
            ffh = min(h - fy, ffh + 2 * pad_y)
            
            if ffw < 20 or ffh < 20:
                continue
            
            # Extract face from ORIGINAL frame for best quality
            face_roi = frame[fy:fy+ffh, fx:fx+ffw]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.equalizeHist(face_gray)
            
            # Get face ID for tracking
            face_id = self.emotion_predictor.get_face_id([fx, fy, ffw, ffh])
            
            # Predict emotion
            face_input = self.emotion_predictor.preprocess_face(face_gray)
            if face_input is not None:
                predictions = self.emotion_predictor.predict_face(face_input, face_id)
                
                if predictions:
                    results.append({
                        'bbox': [fx, fy, ffw, ffh],
                        'predictions': predictions,
                        'face_id': face_id,
                        'confidence': confidence
                    })
        
        self.current_results = results
        
        # Calculate FPS
        current_time = time.time()
        dt = max(current_time - self.last_time, 0.001)
        fps = 1.0 / dt
        self.last_time = current_time
        self.fps_history.append(fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else fps
        
        return frame, results, avg_fps
    
    def get_all_emotions(self):
        """Get all detected face emotions for API"""
        if not self.current_results:
            return []
        
        all_emotions = []
        for result in self.current_results:
            top = result['predictions'][0]
            all_emotions.append({
                'face_id': result['face_id'],
                'bbox': result['bbox'],
                'emotion': top['emotion'],
                'emoji': top['emoji'],
                'confidence': round(top['probability'] * 100, 1),
                'color': top['color_hex'],
                'all_predictions': result['predictions'][:3]
            })
        
        return all_emotions

# Initialize processors
video_processors = {}
for model_key, model_data in loaded_models.items():
    if model_data['model'] is not None:
        video_processors[model_key] = HighPerformanceVideoProcessor(model_data['model'])

current_model_key = 'model2'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_with_crop(image_path, crop_coords=None):
    """
    Enhanced image preprocessing with manual crop support and advanced auto-detection
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image")
    
    h, w = img.shape[:2]
    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    gray_bilateral = cv2.bilateralFilter(gray_eq, 5, 50, 50)
    
    face_detected = False
    face = None
    face_coords = None
    detection_method = "None"
    all_faces = []
    
    # ===== STRATEGY 1: Manual Crop =====
    if crop_coords:
        x = max(0, int(crop_coords.get('x', 0)))
        y = max(0, int(crop_coords.get('y', 0)))
        cw = max(10, int(crop_coords.get('w', w // 3)))
        ch = max(10, int(crop_coords.get('h', h // 3)))
        
        x = min(x, w - 1)
        y = min(y, h - 1)
        cw = min(cw, w - x)
        ch = min(ch, h - y)
        
        face = gray[y:y+ch, x:x+cw]
        face = cv2.equalizeHist(face)
        
        face_detected = True
        face_coords = {'x': x, 'y': y, 'w': cw, 'h': ch}
        detection_method = "Manual Crop"
        all_faces.append({'bbox': [x, y, cw, ch], 'face': face})
    
    # ===== STRATEGY 2: Multi-Cascade Auto Detection =====
    if not face_detected:
        cascades_test = [
            (face_cascade, 1.1, 5),
            (face_cascade, 1.05, 6),
            (face_cascade, 1.15, 4),
        ]
        
        if not profile_cascade.empty():
            cascades_test.append((profile_cascade, 1.1, 5))
        
        detected_faces = []
        for cascade, scale, neighbors in cascades_test:
            if cascade.empty():
                continue
            
            faces = cascade.detectMultiScale(
                gray_bilateral,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (fx, fy, fw, fh) in faces:
                # Check for eyes to validate
                face_roi_check = gray[fy:fy+fh, fx:fx+fw]
                if face_roi_check.size > 0:
                    eyes = eye_cascade.detectMultiScale(face_roi_check, 1.1, 3, minSize=(8, 8))
                    confidence = 0.6 + len(eyes) * 0.1 + (neighbors - 3) * 0.05
                    detected_faces.append({
                        'bbox': [int(fx), int(fy), int(fw), int(fh)],
                        'confidence': min(0.95, confidence),
                        'has_eyes': len(eyes) > 0
                    })
        
        if detected_faces:
            # Sort by confidence and area
            detected_faces.sort(key=lambda f: (f['confidence'], f['bbox'][2] * f['bbox'][3]), reverse=True)
            
            # Process all detected faces
            for i, df in enumerate(detected_faces):
                fx, fy, fw, fh = df['bbox']
                
                # Adaptive padding based on confidence
                pad_pct = 0.05 + df['confidence'] * 0.1
                pad_x = int(fw * pad_pct)
                pad_y = int(fh * pad_pct)
                
                fx = max(0, fx - pad_x)
                fy = max(0, fy - pad_y)
                fw = min(w - fx, fw + 2 * pad_x)
                fh = min(h - fy, fh + 2 * pad_y)
                
                face_roi = gray[fy:fy+fh, fx:fx+fw]
                face_roi = cv2.equalizeHist(face_roi)
                
                all_faces.append({
                    'bbox': [fx, fy, fw, fh],
                    'face': face_roi,
                    'confidence': df['confidence']
                })
            
            # Use highest confidence face as primary
            if all_faces:
                best = max(all_faces, key=lambda f: f.get('confidence', 0))
                face = best['face']
                face_coords = {'x': best['bbox'][0], 'y': best['bbox'][1],
                              'w': best['bbox'][2], 'h': best['bbox'][3]}
                face_detected = True
                detection_method = f"Auto ({len(all_faces)} face(s))"
    
    # ===== STRATEGY 3: Center Crop Fallback =====
    if not face_detected or face is None:
        center_x, center_y = w // 2, h // 2
        crop_w = int(w * 0.6)
        crop_h = int(h * 0.75)
        
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        face = gray[y1:y2, x1:x2]
        face_coords = {'x': x1, 'y': y1, 'w': x2-x1, 'h': y2-y1}
        all_faces.append({'bbox': [x1, y1, x2-x1, y2-y1], 'face': face})
        detection_method = "Center Crop"
    
    # ===== Prepare display images =====
    _, gray_buffer = cv2.imencode('.png', gray)
    gray_base64 = base64.b64encode(gray_buffer).decode('utf-8')
    
    intermediate = {}
    
    # Face crop preview
    if face is not None and face.size > 0:
        _, face_buffer = cv2.imencode('.png', face)
        intermediate['face_crop'] = base64.b64encode(face_buffer).decode('utf-8')
    else:
        intermediate['face_crop'] = gray_base64
    
    # Draw rectangles on original for display
    display_img = original_img.copy()
    for af in all_faces:
        fx, fy, fw, fh = af['bbox']
        cv2.rectangle(display_img, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
    
    _, display_buffer = cv2.imencode('.png', display_img)
    intermediate['detected_faces'] = base64.b64encode(display_buffer).decode('utf-8')
    
    # Model preprocessing
    if face is None or face.size == 0:
        face = gray.copy()
    
    face_resized = cv2.resize(face, (48, 48))
    face_normalized = face_resized.astype(np.float32) / 255.0
    face_normalized = np.expand_dims(face_normalized, axis=(0, -1))
    
    # Scaled previews
    face_display_144 = cv2.resize(face_resized, (144, 144), interpolation=cv2.INTER_NEAREST)
    _, model_input_buffer = cv2.imencode('.png', face_display_144)
    intermediate['model_input'] = base64.b64encode(model_input_buffer).decode('utf-8')
    
    _, model_input_48_buffer = cv2.imencode('.png', face_resized)
    intermediate['model_input_48'] = base64.b64encode(model_input_48_buffer).decode('utf-8')
    
    if face_coords:
        face_coords = {k: int(v) for k, v in face_coords.items()}
    
    logger.info(f"🔍 Detection: {detection_method} | Faces found: {len(all_faces)}")
    
    return face_normalized, face_detected, gray_base64, intermediate, face_coords, all_faces

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def predict_emotion(model, face_array):
    """Get emotion predictions"""
    predictions = model.predict(face_array, verbose=0)[0]
    sorted_indices = np.argsort(predictions)[::-1]

    results = []
    for idx in sorted_indices[:7]:
        idx_int = int(idx)
        emotion_name = EMOTION_MAP[idx_int]
        results.append({
            'emotion': emotion_name,
            'probability': round(float(predictions[idx_int]), 4),
            'color': EMOTION_COLORS_HEX[emotion_name],
            'color_bgr': EMOTION_COLORS_BGR[emotion_name],
            'emoji': EMOTION_EMOJIS[emotion_name]
        })
    return results

def draw_multi_face_results(frame, results, fps):
    """Professional drawing for multiple faces like top detectors"""
    h, w = frame.shape[:2]
    
    for result in results:
        if 'bbox' not in result or not result['predictions']:
            continue
        
        x, y, fw, fh = result['bbox']
        top = result['predictions'][0]
        color_bgr = top['color_bgr']
        confidence = result.get('confidence', 0.9)
        
        # Draw face rectangle with rounded corners effect
        thickness = 2 if confidence > 0.7 else 1
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color_bgr, thickness)
        
        # Draw corner accents (like professional detectors)
        corner_len = min(fw, fh) // 4
        cv2.line(frame, (x, y), (x + corner_len, y), color_bgr, 3)
        cv2.line(frame, (x, y), (x, y + corner_len), color_bgr, 3)
        cv2.line(frame, (x + fw, y), (x + fw - corner_len, y), color_bgr, 3)
        cv2.line(frame, (x + fw, y), (x + fw, y + corner_len), color_bgr, 3)
        cv2.line(frame, (x, y + fh), (x + corner_len, y + fh), color_bgr, 3)
        cv2.line(frame, (x, y + fh), (x, y + fh - corner_len), color_bgr, 3)
        cv2.line(frame, (x + fw, y + fh), (x + fw - corner_len, y + fh), color_bgr, 3)
        cv2.line(frame, (x + fw, y + fh), (x + fw, y + fh - corner_len), color_bgr, 3)
        
        # Main label with shadow
        label = f"{top['emoji']} {top['emotion']}"
        prob_text = f"{top['probability']:.0%}"
        
        # Shadow
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x, y - lh - 25), (x + lw + 55, y), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y - lh - 25), (x + lw + 55, y), color_bgr, 2)
        
        cv2.putText(frame, label, (x + 5, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, prob_text, (x + lw + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 2)
        
        # Mini probability bars on the right
        bx = x + fw + 5
        by = y
        bw = min(100, w - bx - 10)
        bh = 14
        
        if bx + bw < w:
            for i, pred in enumerate(result['predictions'][:3]):
                py = by + i * (bh + 3)
                
                # Background
                cv2.rectangle(frame, (bx, py), (bx + bw, py + bh), (30, 30, 30), -1)
                cv2.rectangle(frame, (bx, py), (bx + bw, py + bh), (60, 60, 60), 1)
                
                # Fill
                fill_w = int(bw * pred['probability'])
                if fill_w > 0:
                    cv2.rectangle(frame, (bx, py), (bx + fill_w, py + bh), pred['color_bgr'], -1)
                
                # Text
                bar_text = f"{pred['emoji']} {pred['probability']:.0%}"
                cv2.putText(frame, bar_text, (bx + 3, py + bh - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # FPS counter (top right)
    fps_color = (0, 255, 0) if fps > 20 else (0, 200, 255) if fps > 15 else (0, 100, 255)
    cv2.putText(frame, f'FPS: {fps:.1f}', (w - 110, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
    
    # Face count (top left)
    face_count = len(results)
    cv2.putText(frame, f'Faces: {face_count}', (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_models_info')
def get_models_info():
    models_info = {}
    for model_key, model_data in loaded_models.items():
        config = model_data['config'].copy()
        config['loaded'] = model_data['model'] is not None
        models_info[model_key] = config
    return jsonify(models_info)

@app.route('/get_dataset_info')
def get_dataset_info():
    return jsonify({
        'name': 'FER2013',
        'samples': '35,887',
        'classes': '7 emotions',
        'resolution': '48×48 pixels'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle single or multiple image predictions"""
    files = request.files.getlist('files')
    model_key = request.form.get('model', 'model2')
    
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files uploaded'}), 400

    # Get manual crop coordinates
    crop_coords = {}
    for key in ['crop_x', 'crop_y', 'crop_w', 'crop_h']:
        val = request.form.get(key, type=float)
        if val is not None and val > 0:
            crop_coords[key.lstrip('crop_')] = val
    if not crop_coords:
        crop_coords = None

    selected_model = loaded_models.get(model_key)
    results_list = []
    
    for file in files:
        if not file or file.filename == '' or not allowed_file(file.filename):
            continue
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process image with advanced detection
            face, face_detected, gray_base64, intermediate, face_coords, all_faces = \
                preprocess_image_with_crop(filepath, crop_coords)
            
            # Predict
            if selected_model and selected_model['model'] is not None:
                model = selected_model['model']
                predictions = predict_emotion(model, face)
                model_name = selected_model['config']['name']
            else:
                predictions = [
                    {'emotion': 'Happy', 'probability': 0.85, 'color': '#FFD700', 'emoji': '😊', 'color_bgr': (0, 215, 255)},
                    {'emotion': 'Neutral', 'probability': 0.08, 'color': '#D2CBCB', 'emoji': '😐', 'color_bgr': (203, 203, 210)},
                    {'emotion': 'Surprise', 'probability': 0.04, 'color': '#ED870A', 'emoji': '😲', 'color_bgr': (10, 135, 237)}
                ]
                model_name = 'Demo Mode'
            
            # Count total faces
            total_faces = len([f for f in all_faces if f.get('confidence', 0) > 0.5])
            
            results_list.append({
                'filename': filename,
                'success': True,
                'image': image_to_base64(filepath),
                'gray_image': gray_base64,
                'face_crop': intermediate.get('face_crop', gray_base64),
                'detected_faces_image': intermediate.get('detected_faces', ''),
                'model_input': intermediate.get('model_input', gray_base64),
                'face_detected': bool(face_detected),
                'face_coords': face_coords,
                'total_faces_detected': total_faces,
                'top_emotion': predictions[0]['emotion'],
                'top_probability': predictions[0]['probability'],
                'top_color': predictions[0]['color'],
                'top_emoji': predictions[0]['emoji'],
                'results': predictions,
                'model_used': model_name
            })
        
        except Exception as e:
            traceback.print_exc()
            results_list.append({
                'filename': filename,
                'success': False,
                'error': str(e)
            })
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
    
    return jsonify({
        'success': True,
        'total_processed': len(results_list),
        'results': results_list
    })

@app.route('/video_feed')
def video_feed():
    model_key = request.args.get('model', 'model2')
    global current_model_key
    current_model_key = model_key
    
    processor = video_processors.get(model_key)
    if not processor:
        return jsonify({'error': 'Model not available'}), 400
    
    return Response(
        generate_frames(processor),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# Replace the generate_frames function in advanced.py with this:

def generate_frames(processor):
    """Generate video frames with emotion detection"""
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        logger.error("Cannot open camera")
        return
    
    # Optimize camera settings
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Warm up camera
    for _ in range(5):
        camera.read()
    
    logger.info("📹 Webcam stream started")
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                logger.warning("Failed to read frame from camera")
                break
            
            # Process frame with emotion detection
            processed_frame, results, fps = processor.process_frame(frame)
            
            # Draw results on frame
            display_frame = draw_multi_face_results(processed_frame, results, fps)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    except GeneratorExit:
        logger.info("📹 Webcam stream stopped (client disconnected)")
    except Exception as e:
        logger.error(f"❌ Webcam stream error: {e}")
    finally:
        camera.release()
        logger.info("📹 Camera released")

@app.route('/emotion_data')
def emotion_data():
    model_key = request.args.get('model', 'model2')
    processor = video_processors.get(model_key)
    
    if not processor:
        return jsonify({'faces': [], 'total_faces': 0, 'fps': 0})
    
    faces = processor.get_all_emotions()
    fps = sum(processor.fps_history) / len(processor.fps_history) if processor.fps_history else 0
    
    return jsonify({
        'faces': faces,
        'total_faces': len(faces),
        'fps': round(fps, 1)
    })

@app.route('/webcam_status')
def webcam_status():
    try:
        camera = cv2.VideoCapture(0)
        available = camera.isOpened()
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) if available else 0
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) if available else 0
        camera.release()
        return jsonify({
            'available': available,
            'width': width,
            'height': height
        })
    except:
        return jsonify({'available': False})

@app.route('/sample')
def get_sample():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 240
    cv2.circle(img, (150, 130), 90, (255, 215, 0), -1)
    cv2.circle(img, (115, 105), 12, (0, 0, 0), -1)
    cv2.circle(img, (185, 105), 12, (0, 0, 0), -1)
    cv2.circle(img, (110, 100), 4, (255, 255, 255), -1)
    cv2.circle(img, (180, 100), 4, (255, 255, 255), -1)
    cv2.ellipse(img, (150, 140), (30, 25), 0, 20, 160, (0, 0, 0), 3)
    cv2.circle(img, (95, 145), 15, (180, 180, 255), -1)
    cv2.circle(img, (205, 145), 15, (180, 180, 255), -1)

    _, buffer = cv2.imencode('.png', img)
    img_io = BytesIO(buffer)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    logger.info("\n" + "="*50)
    logger.info("EMOTION AI PRO - STARTING SERVER")
    logger.info("http://localhost:5000")
    logger.info("="*50)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)