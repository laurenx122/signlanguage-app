"""
fsl_static_inference.py - Updated for preprocessed landmarks
Real-time FSL recognition with proper normalization
"""

import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import cv2
import base64
from pathlib import Path
_FRAME_SKIP = 2        # run mediapipe every 2 frames
_frame_counter = 0
_last_landmarks = None
should_speak = False
# TTS import
try:
    from ..tts.tts_engine import speak
except ImportError:
    try:
        from src.tts.tts_engine import speak
    except ImportError:
        print("⚠️ TTS engine not available")
        def speak(text):
            pass

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm" / "best_fsl_lstm_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_hands = None
_inv_label_map = None
_last_prediction = None
_prediction_count = 0


class LSTMGestureModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMGestureModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        
        out = self.dropout(self.relu(self.batch_norm1(self.fc1(out))))
        out = self.dropout(self.relu(self.batch_norm2(self.fc2(out))))
        return self.fc3(out)


def is_left_hand(hand_landmarks, handedness):
    """Determine if detected hand is left hand"""
    if handedness and handedness.classification:
        label = handedness.classification[0].label
        return label == 'Left'
    return False


def mirror_landmarks_horizontal(landmarks):
    """Mirror landmarks horizontally for left->right conversion"""
    mirrored = landmarks.copy()
    for i in range(0, len(mirrored), 3):
        mirrored[i] = 1.0 - mirrored[i]
    return mirrored


def normalize_landmarks(landmarks):
    """
    Normalize landmarks (wrist-centered, scale-invariant)
    Same as preprocessing
    """
    landmarks = np.array(landmarks).reshape(-1, 3)
    
    # Center at wrist
    wrist = landmarks[0].copy()
    landmarks_centered = landmarks - wrist
    
    # Scale by hand size
    distances = np.linalg.norm(landmarks_centered, axis=1)
    hand_size = np.max(distances)
    
    if hand_size < 1e-6:
        hand_size = 1.0
    
    landmarks_normalized = landmarks_centered / hand_size
    
    return landmarks_normalized.flatten()


def initialize_fsl_model():
    global _model, _hands, _inv_label_map

    if _model is not None:
        return

    try:
        print("🔄 Loading FSL model...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        _inv_label_map = {v: k for k, v in checkpoint['class_to_idx'].items()}
        num_classes = len(checkpoint['classes'])
        feature_dim = checkpoint.get('feature_dim', 63)
        
        print(f"📊 Classes: {num_classes}")
        print(f"📊 Feature dimension: {feature_dim}")
        
        # Initialize model
        _model = LSTMGestureModel(
            input_size=feature_dim,
            hidden_size=checkpoint.get('hidden_size', 128),
            num_layers=checkpoint.get('num_layers', 2),
            num_classes=num_classes,
            dropout=checkpoint.get('dropout', 0.3)
        ).to(DEVICE)
        
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()
        print("✅ Model loaded successfully")

        # Initialize MediaPipe
        _hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,  # Detect both hands
            min_detection_confidence=0.5
        )
        print("✅ MediaPipe initialized (2 hands)")
        
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def extract_landmarks_from_frame(frame):
    """Extract and normalize landmarks from frame (supports 2 hands)"""
    global _hands,_frame_counter
    if _hands is None:
        initialize_fsl_model()
    
     # 🔑 Skip heavy MediaPipe work on some frames
    if _frame_counter % _FRAME_SKIP != 0 and _last_landmarks is not None:
        return _last_landmarks
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Initialize array for 2 hands (126 features)
    all_landmarks = []
    
    # Process up to 2 hands
    for i in range(min(len(results.multi_hand_landmarks), 2)):
        hand_landmarks = results.multi_hand_landmarks[i]
        handedness = results.multi_handedness[i] if results.multi_handedness else None
        
        # Extract landmarks
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Convert left to right
        if is_left_hand(hand_landmarks, handedness):
            landmarks = mirror_landmarks_horizontal(landmarks)
        
        # Normalize (same as preprocessing)
        landmarks = normalize_landmarks(landmarks)
        all_landmarks.extend(landmarks)
    
    # Pad to 126 if only one hand detected
    while len(all_landmarks) < 126:
        all_landmarks.extend([0.0] * 63)
    
        _last_landmarks = np.array(all_landmarks[:126], dtype=np.float32)
    return _last_landmarks


def predict_fsl_static(frame_base64, confidence_threshold=0.6):
    global _last_prediction, _prediction_count
    should_speak = False 
    try:
        if _model is None:
            initialize_fsl_model()
        
        # Decode image
        img_bytes = base64.b64decode(frame_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {
                'success': False,
                'prediction': "UNKNOWN",
                'confidence': 0.0,
                'message': "Failed to decode image"
            }
        
        # Extract landmarks
        landmarks = extract_landmarks_from_frame(frame)
        
        if landmarks is None:
            _last_prediction = None
            _prediction_count = 0
            return {
                'success': True,
                'prediction': "UNKNOWN",
                'confidence': 0.0,
                'message': "No hands detected"
            }
        
        # Predict
        tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = _model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, 1)
        
        pred_class = _inv_label_map[idx.item()]
        confidence = conf.item()
        
        print(f"🎯 Prediction: {pred_class}, Confidence: {confidence:.3f}")

        # ---------- FAST EXIT ----------
        if confidence >= 0.9:
            if pred_class != _last_prediction:
                _last_prediction = pred_class
                return {
                    'success': True,
                    'prediction': pred_class,
                    'confidence': confidence,
                    'should_speak': True
                }
        
        if confidence < confidence_threshold:
            pred_class = "UNKNOWN"
            _last_prediction = None
            _prediction_count = 0
        else:
            if confidence >= confidence_threshold:
                if pred_class != _last_prediction:
                    should_speak = True
                    _last_prediction = pred_class
            else:
                _last_prediction = None
        
        return {
            'success': True,
            'prediction': pred_class,
            'confidence': float(confidence),
            'message': "Success",
            'should_speak': should_speak
        }
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'prediction': "UNKNOWN",
            'confidence': 0.0,
            'message': f"Error: {str(e)}",
            'should_speak': should_speak
        }


def predict_fsl_batch(frames_base64, confidence_threshold=0.6):
    global _last_prediction, _prediction_count
    from collections import Counter
    
    predictions = []
    confidences = []
    
    for f in frames_base64:
        r = predict_fsl_static(f, confidence_threshold)
        if r['success'] and r['prediction'] != "UNKNOWN":
            predictions.append(r['prediction'])
            confidences.append(r['confidence'])
    
    if not predictions:
        _last_prediction = None
        _prediction_count = 0
        return {
            'success': True,
            'prediction': "UNKNOWN",
            'confidence': 0.0,
            'message': "No valid predictions"
        }
    
    # Most common prediction
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]
    
    # Average confidence
    avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == most_common])
    
    # Speak (batch is more stable)
    if most_common != _last_prediction:
        print(f"🔊 Speaking (batch): {most_common}")
        speak(f"Letter {most_common}")
        _last_prediction = most_common
        
    return {
        'success': True,
        'prediction': most_common,
        'confidence': float(avg_conf),
        'message': f"Predicted from {len(predictions)}/{len(frames_base64)} frames"
    }