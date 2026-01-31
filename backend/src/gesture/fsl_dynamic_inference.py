"""
fsl_dynamic_inference_14.py
Inference for 14-gesture dynamic model
Location: D:/SMS/backend/src/gesture/fsl_dynamic_inference_14.py
"""

import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import cv2
from pathlib import Path
import pickle
from collections import deque

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'lstm_dynamic_12' / 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables
_model = None
_label_mapping = None
_hands = None
_frame_buffer = None
_sequence_length = None


class ImprovedLSTMModel(nn.Module):
    """Same architecture as training"""
    def __init__(self, input_size=126, hidden_size=256, num_layers=3, 
                 num_classes=14, dropout=0.4):
        super(ImprovedLSTMModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, 256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn_conv = nn.BatchNorm1d(256)
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_t = x.transpose(1, 2)
        x_t = torch.relu(self.conv1(x_t))
        x_t = torch.relu(self.conv2(x_t))
        x_t = self.bn_conv(x_t)
        x_t = x_t.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x_t)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        
        out = self.fc1(pooled)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out


def normalize_landmarks(landmarks):
    """Normalize landmarks (wrist-centered, scale-invariant)"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0].copy()
    landmarks_centered = landmarks - wrist
    distances = np.linalg.norm(landmarks_centered, axis=1)
    hand_size = np.max(distances)
    if hand_size < 1e-6:
        hand_size = 1.0
    return (landmarks_centered / hand_size).flatten()


def initialize_dynamic_model():
    """Initialize the 14-gesture model"""
    global _model, _label_mapping, _hands, _frame_buffer, _sequence_length
    
    if _model is not None:
        return
    
    print("="*70)
    print("🔄 Loading 14-Gesture Dynamic Model...")
    print("="*70)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    _label_mapping = checkpoint['label_mapping']
    _sequence_length = checkpoint['sequence_length']
    num_classes = len(_label_mapping['idx_to_label'])
    
    print(f"✅ Model info:")
    print(f"   Classes: {num_classes}")
    print(f"   Sequence length: {_sequence_length}")
    print(f"   Hidden size: {checkpoint['hidden_size']}")
    print(f"   Num layers: {checkpoint['num_layers']}")
    
    if 'val_top1_acc' in checkpoint:
        print(f"   Validation Top-1: {checkpoint['val_top1_acc']:.2f}%")
        print(f"   Validation Top-3: {checkpoint['val_top3_acc']:.2f}%")
        print(f"   Validation F1: {checkpoint['val_f1']:.4f}")
    
    # Initialize model
    _model = ImprovedLSTMModel(
        input_size=126,
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        num_classes=num_classes,
        dropout=checkpoint['dropout']
    ).to(DEVICE)
    
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()
    
    print(f"✅ Model loaded successfully")
    
    # Initialize MediaPipe
    _hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize frame buffer
    _frame_buffer = deque(maxlen=_sequence_length)
    
    print(f"✅ MediaPipe initialized")
    print(f"✅ Frame buffer ready (max: {_sequence_length} frames)")
    print("="*70)


def add_frame_to_buffer(frame):
    """
    Process frame and add to buffer
    Returns: True if hands detected, False otherwise
    """
    global _frame_buffer, _hands
    
    if _hands is None:
        initialize_dynamic_model()
    
    # Process frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb)
    
    frame_feats = np.zeros(126, dtype=np.float32)
    hands_detected = False
    
    if results.multi_hand_landmarks and results.multi_handedness:
        hands_detected = True
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            
            lm = []
            for landmark in hand_landmarks.landmark:
                lm.extend([landmark.x, landmark.y, landmark.z])
            
            norm_lm = normalize_landmarks(lm)
            
            if label == 'Left':
                frame_feats[0:63] = norm_lm
            else:
                frame_feats[63:126] = norm_lm
    
    # Only add frames with hands detected (active filtering)
    if hands_detected:
        _frame_buffer.append(frame_feats)
    
    return hands_detected


def predict_dynamic_sign():
    """
    Predict from current buffer
    Returns dict with top-1 and top-3 predictions
    """
    global _model, _frame_buffer, _label_mapping, _sequence_length
    
    if _model is None:
        initialize_dynamic_model()
    
    if len(_frame_buffer) == 0:
        return {
            'top1_label': 'UNKNOWN',
            'top1_conf': 0.0,
            'top3': [('UNKNOWN', 0.0)] * 3
        }
    
    # Create sequence from buffer
    sequence = list(_frame_buffer)
    
    # Pad to sequence_length
    while len(sequence) < _sequence_length:
        sequence.append(np.zeros(126, dtype=np.float32))
    
    # Truncate if too long
    if len(sequence) > _sequence_length:
        sequence = sequence[:_sequence_length]
    
    sequence = np.array(sequence, dtype=np.float32)
    
    # Predict
    with torch.no_grad():
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)
        
        # Top-1
        top1_conf, top1_idx = torch.max(probs, 1)
        top1_label = _label_mapping['idx_to_label'][top1_idx.item()]
        top1_conf = top1_conf.item()
        
        # Top-3
        top3_confs, top3_idxs = torch.topk(probs, k=min(3, len(_label_mapping['idx_to_label'])), dim=1)
        top3 = [
            (_label_mapping['idx_to_label'][idx.item()], conf.item())
            for idx, conf in zip(top3_idxs[0], top3_confs[0])
        ]
    
    return {
        'top1_label': top1_label,
        'top1_conf': top1_conf,
        'top3': top3
    }


def reset_buffer():
    """Reset the frame buffer"""
    global _frame_buffer
    if _frame_buffer is not None:
        _frame_buffer.clear()


def get_buffer_info():
    """Get current buffer status"""
    global _frame_buffer, _sequence_length
    
    if _frame_buffer is None:
        return {
            'current': 0,
            'max': 0,
            'percentage': 0.0
        }
    
    return {
        'current': len(_frame_buffer),
        'max': _sequence_length,
        'percentage': 100.0 * len(_frame_buffer) / _sequence_length if _sequence_length > 0 else 0.0
    }


if __name__ == '__main__':
    # Test
    print("Testing inference module...")
    initialize_dynamic_model()
    print("\n✅ Inference module ready!")
    print(f"   Available gestures: {list(_label_mapping['label_to_idx'].keys())}")