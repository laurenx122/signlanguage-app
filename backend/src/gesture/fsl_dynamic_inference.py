"""
fsl_dynamic_inference.py
Inference module for FSL dynamic sign language recognition.

UPDATED:
- Segment-based inference (collect a whole gesture sequence, then predict once)
- Resample collected frames to SEQUENCE_LENGTH (matches dataset extraction logic)
- Avoid filling buffer with zeros when no hands are present
- Start/stop gesture based on consecutive hand presence/absence
- Optional cooldown to prevent rapid repeated triggers
"""

import json
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque
import time

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'lstm_dynamic_final' / 'final_model_complete.pth'
LABEL_MAP_PATH = PROJECT_ROOT / 'models' / 'lstm_dynamic_final' / 'label_mapping.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 252
SEQUENCE_LENGTH = 30

# Gesture segmentation thresholds (tune these)
START_HAND_FRAMES = 4     # need this many consecutive frames with hands to "start" gesture
END_NOHAND_FRAMES = 6     # need this many consecutive frames without hands to "end" gesture
MIN_GESTURE_FRAMES = 12   # ignore gestures shorter than this
COOLDOWN_SECONDS = 0.6    # after a prediction, ignore triggers briefly

# --- Model Architecture ---
class ImprovedLSTMModel(nn.Module):
    """LSTM with Conv1D preprocessing, bidirectional layers, and attention"""
    def __init__(self, input_size=126, hidden_size=256, num_layers=3, num_classes=44, dropout=0.4):
        super(ImprovedLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm1d(256)
        self.lstm = nn.LSTM(
            256, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=8, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
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

# Segment collection state
collecting = False
gesture_frames = []          # list[np.ndarray] (each is 126,)
consec_hand = 0
consec_nohand = 0
last_prediction_time = 0.0

def normalize_landmarks(landmarks_array: np.ndarray) -> np.ndarray:
    """Normalizes landmarks relative to the wrist (Landmark 0) and scales."""
    coords = landmarks_array.reshape(21, 3)
    wrist = coords[0]
    centered_coords = coords - wrist
    max_val = np.max(np.abs(centered_coords))
    if max_val > 0:
        centered_coords = centered_coords / max_val
    return centered_coords.flatten().astype(np.float32)

def load_label_mapping():
    """Load label mapping from JSON file"""
    global label_mapping

    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        print(f"✅ Loaded label mapping: {len(label_mapping)} labels")
    else:
        print(f"⚠️  Label mapping not found at {LABEL_MAP_PATH}")
        print(f"   Run: python src/gesture/create_label_mapping.py")
        label_mapping = {}

def get_label(folder_name: str) -> str:
    """Convert folder name to human-readable label"""
    global label_mapping
    if label_mapping and folder_name in label_mapping:
        return label_mapping[folder_name]
    return f"SIGN_{folder_name}"

def initialize_dynamic_model():
    """Load trained model and initialize MediaPipe"""
    global model, classes, mp_hands, hands
    global collecting, gesture_frames, consec_hand, consec_nohand, last_prediction_time

    print("=" * 60)
    print("🚀 Initializing FSL Dynamic Recognition System")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

    load_label_mapping()

    print(f"📂 Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    classes = checkpoint['classes']
    num_classes = checkpoint['num_classes']

    print("📊 Model Info:")
    print(f"   - Classes: {num_classes}")
    print(f"   - Device:  {DEVICE}")
    print(f"   - Test F1: {checkpoint['test_metrics']['f1']:.4f}")

    # IMPORTANT: use dropout=0.0 (or keep 0.4) doesn't matter in eval, but keep consistent
    # model = ImprovedLSTMModel(num_classes=num_classes, dropout=0.4)
    model = ImprovedLSTMModel(input_size=INPUT_SIZE, num_classes=num_classes, dropout=0.4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # reset state
    collecting = False
    gesture_frames = []
    consec_hand = 0
    consec_nohand = 0
    last_prediction_time = 0.0

    print("✅ Initialization complete!")
    print("=" * 60 + "\n")

def extract_frame_features(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Extract 126-dim features for one frame.
    Returns: (features, hands_detected)
    """
    global hands

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    frame_feats = np.zeros(INPUT_SIZE, dtype=np.float32)
    hands_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:
        hands_detected = True

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label

            raw_lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            norm_lms = normalize_landmarks(raw_lms)

            if label == 'Left':
                frame_feats[0:63] = norm_lms
            else:
                frame_feats[63:126] = norm_lms

            # optional drawing for UI
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    return frame_feats, hands_detected

def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a variable-length sequence to target_len using linear index spacing.
    Similar idea to your dataset extraction (linspace sampling over frames).
    seq: (T, 126)
    returns: (target_len, 126)
    """
    T = seq.shape[0]
    if T == 0:
        return np.zeros((target_len, INPUT_SIZE), dtype=np.float32)

    if T == target_len:
        return seq.astype(np.float32)

    # choose indices spaced over [0..T-1]
    idxs = np.linspace(0, T - 1, target_len).astype(int)
    return seq[idxs].astype(np.float32)

def predict_from_sequence(sequence_30: np.ndarray) -> dict:
    """Run model on a (30, 126) sequence, adds velocity, returns top results."""
    global model, classes

    # Only add velocity if not already added (check if shape is still 126)
    if sequence_30.shape[1] == 126:
        velocity = np.zeros_like(sequence_30)
        velocity[1:] = sequence_30[1:] - sequence_30[:-1]
        sequence_30 = np.concatenate([sequence_30, velocity], axis=1)  # (30, 252)

    sequence_tensor = torch.from_numpy(sequence_30).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(sequence_tensor)
        probs = torch.softmax(output, dim=1)
        top3_probs, top3_idx = torch.topk(probs, k=min(3, len(classes)), dim=1)

        top3_probs = top3_probs.cpu().numpy()[0]
        top3_idx = top3_idx.cpu().numpy()[0]

    return {
        'top1_label': get_label(classes[top3_idx[0]]),
        'top1_conf': float(top3_probs[0]),
        'top3_labels': [get_label(classes[i]) for i in top3_idx],
        'top3_confs': [float(p) for p in top3_probs],
        'is_ready': True
    }

def update_and_maybe_predict(frame: np.ndarray) -> dict:
    """
    Call this for every camera frame.
    It collects a gesture segment, and ONLY predicts when the gesture ends.
    """
    global collecting, gesture_frames, consec_hand, consec_nohand, last_prediction_time

    feats, hands_detected = extract_frame_features(frame)

    # cooldown to prevent spam predictions
    now = time.time()
    in_cooldown = (now - last_prediction_time) < COOLDOWN_SECONDS

    if hands_detected:
        consec_hand += 1
        consec_nohand = 0
    else:
        consec_nohand += 1
        consec_hand = 0

    # Start collecting when hands are stable for a few frames (and not in cooldown)
    if (not collecting) and (not in_cooldown) and (consec_hand >= START_HAND_FRAMES):
        collecting = True
        gesture_frames = []
        # include current frame as part of gesture
        gesture_frames.append(feats)

    elif collecting:
        # While collecting, only append frames when hands detected
        # (keeps gesture cleaner; avoids padding with zeros during gaps)
        if hands_detected:
            gesture_frames.append(feats)

        # End gesture when no hands for a few frames
        if consec_nohand >= END_NOHAND_FRAMES:
            collecting = False

            # decide if gesture is long enough
            if len(gesture_frames) >= MIN_GESTURE_FRAMES:
                seq = np.stack(gesture_frames, axis=0)  # (T,126)
                seq30 = resample_sequence(seq, SEQUENCE_LENGTH)
                result = predict_from_sequence(seq30)
                last_prediction_time = now
                # reset
                gesture_frames = []
                return result
            else:
                # too short → ignore
                gesture_frames = []
                return {
                    'top1_label': 'Too short / ignored',
                    'top1_conf': 0.0,
                    'top3_labels': [],
                    'top3_confs': [],
                    'is_ready': False
                }

    # default: not predicting yet
    status = "Collecting..." if collecting else "Waiting..."
    return {
        'top1_label': status,
        'top1_conf': 0.0,
        'top3_labels': [],
        'top3_confs': [],
        'is_ready': False,
        'debug': {
            'collecting': collecting,
            'frames_collected': int(len(gesture_frames)),
            'consec_hand': int(consec_hand),
            'consec_nohand': int(consec_nohand)
        }
    }

def reset_buffer():
    """Manual reset gesture collection."""
    global collecting, gesture_frames, consec_hand, consec_nohand
    collecting = False
    gesture_frames = []
    consec_hand = 0
    consec_nohand = 0
    print("🔄 Gesture state reset")

def get_model_info():
    """Get loaded model information"""
    global classes
    if model is None:
        return {'status': 'not_initialized'}

    return {
        'status': 'ready',
        'num_classes': len(classes),
        'device': str(DEVICE),
        'sequence_length': SEQUENCE_LENGTH,
        'segmentation': {
            'START_HAND_FRAMES': START_HAND_FRAMES,
            'END_NOHAND_FRAMES': END_NOHAND_FRAMES,
            'MIN_GESTURE_FRAMES': MIN_GESTURE_FRAMES,
            'COOLDOWN_SECONDS': COOLDOWN_SECONDS
        }
    }