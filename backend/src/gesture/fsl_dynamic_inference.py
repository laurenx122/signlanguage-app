"""
fsl_dynamic_inference.py
Real-time dynamic sign language recognition from video stream
"""

import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import cv2
from pathlib import Path
from collections import deque
import time

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
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_dynamic" / "best_fsl_dynamic_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sequence parameters
SEQUENCE_LENGTH = 30  # Must match training
BUFFER_SIZE = 30

_model = None
_hands = None
_label_mapping = None
_frame_buffer = None
_last_prediction = None
_prediction_count = 0


class DynamicLSTMModel(nn.Module):
    """Same architecture as training"""
    def __init__(self, input_size=126, hidden_size=256, num_layers=3, 
                 num_classes=10, dropout=0.4):
        super(DynamicLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.dropout(self.relu(self.batch_norm1(self.fc1(attended))))
        out = self.dropout(self.relu(self.batch_norm2(self.fc2(out))))
        return self.fc3(out)


def normalize_landmarks(landmarks):
    """Normalize landmarks (wrist-centered, scale-invariant)"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0].copy()
    landmarks_centered = landmarks - wrist
    distances = np.linalg.norm(landmarks_centered, axis=1)
    hand_size = np.max(distances)
    if hand_size < 1e-6:
        hand_size = 1.0
    landmarks_normalized = landmarks_centered / hand_size
    return landmarks_normalized.flatten()


def initialize_dynamic_model():
    global _model, _hands, _label_mapping, _frame_buffer
    
    if _model is not None:
        return
    
    try:
        print("🔄 Loading FSL Dynamic model...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        _label_mapping = checkpoint['label_mapping']
        num_classes = len(_label_mapping['label_to_idx'])
        
        print(f"📊 Classes: {list(_label_mapping['label_to_idx'].keys())}")
        
        _model = DynamicLSTMModel(
            input_size=126,
            hidden_size=checkpoint.get('hidden_size', 256),
            num_layers=checkpoint.get('num_layers', 3),
            num_classes=num_classes,
            dropout=checkpoint.get('dropout', 0.4)
        ).to(DEVICE)
        
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()
        print("✅ Dynamic model loaded successfully")
        
        # Initialize MediaPipe
        _hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize frame buffer
        _frame_buffer = deque(maxlen=BUFFER_SIZE)
        
        print("✅ MediaPipe initialized for video stream")
        
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def extract_frame_landmarks(frame):
    """Extract landmarks from single frame"""
    global _hands
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb_frame)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Process up to 2 hands
    frame_landmarks = []
    for i in range(min(len(results.multi_hand_landmarks), 2)):
        hand_landmarks = results.multi_hand_landmarks[i]
        
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        landmarks_norm = normalize_landmarks(landmarks)
        frame_landmarks.extend(landmarks_norm)
    
    # Pad to 126 if only one hand
    while len(frame_landmarks) < 126:
        frame_landmarks.extend([0.0] * 63)
    
    return frame_landmarks[:126]


def predict_dynamic_sign(confidence_threshold=0.7):
    """Predict from current frame buffer"""
    global _model, _frame_buffer, _label_mapping, _last_prediction, _prediction_count
    
    if len(_frame_buffer) < SEQUENCE_LENGTH:
        return {
            'success': True,
            'prediction': "COLLECTING",
            'confidence': 0.0,
            'message': f"Buffering frames: {len(_frame_buffer)}/{SEQUENCE_LENGTH}"
        }
    
    # Get last SEQUENCE_LENGTH frames
    sequence = list(_frame_buffer)[-SEQUENCE_LENGTH:]
    sequence = np.array(sequence, dtype=np.float32)
    
    # Predict
    tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    
    pred_label = _label_mapping['idx_to_label'][idx.item()]
    confidence = conf.item()
    
    print(f"🎯 Prediction: {pred_label}, Confidence: {confidence:.3f}")
    
    if confidence < confidence_threshold:
        pred_label = "UNKNOWN"
        _last_prediction = None
        _prediction_count = 0
    else:
        # Stabilize predictions
        if pred_label == _last_prediction:
            _prediction_count += 1
        else:
            _last_prediction = pred_label
            _prediction_count = 1
        
        # Speak after 2 consecutive predictions (faster for phrases)
        if _prediction_count == 2:
            print(f"🔊 Speaking: {pred_label}")
            speak(pred_label)
    
    return {
        'success': True,
        'prediction': pred_label,
        'confidence': float(confidence),
        'message': "Success"
    }


def add_frame_to_buffer(frame):
    """Add frame landmarks to buffer"""
    global _frame_buffer
    
    if _frame_buffer is None:
        initialize_dynamic_model()
    
    landmarks = extract_frame_landmarks(frame)
    
    if landmarks is not None:
        _frame_buffer.append(landmarks)
        return True
    
    return False


def reset_buffer():
    """Reset frame buffer"""
    global _frame_buffer, _last_prediction, _prediction_count
    _frame_buffer.clear()
    _last_prediction = None
    _prediction_count = 0