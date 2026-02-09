"""
fsl_dynamic_inference.py
Inference module for FSL dynamic sign language recognition.
"""

import json
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'lstm_dynamic_14' / 'final_model_complete.pth'
LABEL_MAP_PATH = PROJECT_ROOT / 'models' / 'lstm_dynamic_14' / 'label_mapping.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Architecture ---
class ImprovedLSTMModel(nn.Module):
    """LSTM with Conv1D preprocessing, bidirectional layers, and attention"""
    def __init__(self, input_size=126, hidden_size=256, num_layers=3, num_classes=44, dropout=0.4):
        super(ImprovedLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm1d(256)
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn_conv(self.conv1(x)))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        return self.fc(pooled)

# --- Global State ---
model = None
classes = None
label_mapping = None  # ID → Label mapping
mp_hands = None
hands = None
frame_buffer = None
SEQUENCE_LENGTH = 30

def normalize_landmarks(landmarks_array):
    """Normalizes landmarks relative to the wrist (Landmark 0)"""
    coords = landmarks_array.reshape(21, 3)
    wrist = coords[0]
    centered_coords = coords - wrist
    
    max_val = np.max(np.abs(centered_coords))
    if max_val > 0:
        normalized = centered_coords / max_val
    else:
        normalized = centered_coords
        
    return normalized.flatten()

def load_label_mapping():
    """Load label mapping from JSON file"""
    global label_mapping
    
    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        print(f"✅ Loaded label mapping: {len(label_mapping)} labels")
        print(f"   Sample: {list(label_mapping.items())[:3]}")
    else:
        print(f"⚠️  Label mapping not found at {LABEL_MAP_PATH}")
        print(f"   Run: python src/gesture/create_label_mapping.py")
        label_mapping = {}

def get_label(folder_name):
    """Convert folder name to human-readable label"""
    global label_mapping
    
    if label_mapping and folder_name in label_mapping:
        return label_mapping[folder_name]
    
    # Fallback
    return f"SIGN_{folder_name}"

def initialize_dynamic_model():
    """Load trained model and initialize MediaPipe"""
    global model, classes, mp_hands, hands, frame_buffer
    
    print("="*60)
    print("🚀 Initializing FSL Dynamic Recognition System")
    print("="*60)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
    
    # Load label mapping
    load_label_mapping()
    
    # Load model checkpoint
    print(f"📂 Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Extract metadata
    classes = checkpoint['classes']
    num_classes = checkpoint['num_classes']
    
    print(f"📊 Model Info:")
    print(f"   - Classes: {num_classes}")
    print(f"   - Device: {DEVICE}")
    print(f"   - Test F1: {checkpoint['test_metrics']['f1']:.4f}")
    
    # Initialize model
    model = ImprovedLSTMModel(num_classes=num_classes, dropout=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    # Initialize frame buffer
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    print("✅ Initialization complete!")
    print(f"📹 Buffer size: {SEQUENCE_LENGTH} frames")
    print("="*60 + "\n")

def add_frame_to_buffer(frame):
    """Process a frame and add features to buffer"""
    global frame_buffer, hands
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    frame_feats = np.zeros(126, dtype=np.float32)
    hands_detected = False
    
    if results.multi_hand_landmarks and results.multi_handedness:
        hands_detected = True
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            
            raw_lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            norm_lms = normalize_landmarks(raw_lms)
            
            if label == 'Left':
                frame_feats[0:63] = norm_lms
            else:
                frame_feats[63:126] = norm_lms
            
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp.solutions.hands.HAND_CONNECTIONS
            )
    
    frame_buffer.append(frame_feats)
    return hands_detected

def predict_dynamic_sign():
    """Predict sign from current buffer"""
    global model, classes, frame_buffer
    
    if len(frame_buffer) < SEQUENCE_LENGTH:
        return {
            'top1_label': 'Buffering...',
            'top1_conf': 0.0,
            'top3_labels': [],
            'top3_confs': [],
            'is_ready': False
        }
    
    sequence = np.array(frame_buffer, dtype=np.float32)
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        top3_probs, top3_indices = torch.topk(probabilities, k=min(3, len(classes)), dim=1)
        
        top3_probs = top3_probs.cpu().numpy()[0]
        top3_indices = top3_indices.cpu().numpy()[0]
        
        # Map folder names to labels
        result = {
            'top1_label': get_label(classes[top3_indices[0]]),
            'top1_conf': float(top3_probs[0]),
            'top3_labels': [get_label(classes[idx]) for idx in top3_indices],
            'top3_confs': [float(prob) for prob in top3_probs],
            'is_ready': True
        }
    
    return result

def get_buffer_info():
    """Get current buffer status"""
    global frame_buffer, SEQUENCE_LENGTH
    
    current = len(frame_buffer)
    percentage = (current / SEQUENCE_LENGTH) * 100
    
    return {
        'current': current,
        'max': SEQUENCE_LENGTH,
        'percentage': percentage
    }

def reset_buffer():
    """Clear the frame buffer"""
    global frame_buffer
    frame_buffer.clear()
    print("🔄 Buffer reset")

def get_model_info():
    """Get loaded model information"""
    global classes, DEVICE
    
    if model is None:
        return {'status': 'not_initialized'}
    
    return {
        'status': 'ready',
        'num_classes': len(classes),
        'classes': classes,
        'device': str(DEVICE),
        'sequence_length': SEQUENCE_LENGTH
    }