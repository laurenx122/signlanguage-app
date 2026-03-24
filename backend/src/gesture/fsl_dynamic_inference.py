"""
fsl_dynamic_inference.py
Inference module for FSL dynamic sign language recognition.

UPDATED:
- Segment-based inference (collect a whole gesture sequence, then predict once)
- Resample collected frames to SEQUENCE_LENGTH (matches dataset extraction logic)
- Avoid filling buffer with zeros when no hands are present
- Start/stop gesture based on consecutive hand presence/absence
- Optional cooldown to prevent rapid repeated triggers

ARCHITECTURE FIX:
- [FIX 1] ImprovedLSTMModel updated to match trained checkpoint:
    hidden_size 256 → 128
    num_layers  3   → 2
    conv1 channels  256 → 128
    Removed MultiheadAttention — uses mean pooling instead
    Added input_dropout layer
    fc layer input  512 → 256, hidden 256 → 128
- [FIX 2] input_size default corrected: 126 → 252 (position + velocity)
- [FIX 3] initialize_dynamic_model() now reads dropout from checkpoint metadata
          so architecture is always in sync with the saved model
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

INPUT_SIZE = 252        # 126 position + 126 velocity  [FIX 2: was 126]
SEQUENCE_LENGTH = 30

# Gesture segmentation thresholds
START_HAND_FRAMES  = 4    # consecutive frames with hands to start gesture
END_NOHAND_FRAMES  = 6    # consecutive frames without hands to end gesture
MIN_GESTURE_FRAMES = 12   # ignore gestures shorter than this
COOLDOWN_SECONDS   = 0.6  # after a prediction, ignore triggers briefly


# ---------------------------------------------------------------------------
# [FIX 1] Updated model architecture — must match train_dynamic_fsl.py exactly
# ---------------------------------------------------------------------------
class ImprovedLSTMModel(nn.Module):
    """
    Compact BiLSTM with Conv1D preprocessing.
    hidden_size=128, num_layers=2, no MultiheadAttention.
    Must be identical to the class used during training.
    """
    def __init__(
        self,
        input_size  = INPUT_SIZE,   # 252
        hidden_size = 128,          # [FIX 1] was 256
        num_layers  = 2,            # [FIX 1] was 3
        num_classes = 44,
        dropout     = 0.5,
    ):
        super().__init__()

        # [FIX 1] Input dropout layer (was missing in old inference file)
        self.input_dropout = nn.Dropout(p=0.1)

        # Conv1D — output channels 128 [FIX 1: was 256]
        self.conv1   = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm1d(128)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            128, hidden_size, num_layers,
            batch_first  = True,
            dropout      = dropout,
            bidirectional= True,
        )

        # [FIX 1] No MultiheadAttention — mean pooling is used in forward()
        # Classifier: hidden*2=256 → 128 → num_classes  [FIX 1: was 512→256→num_classes]
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # [FIX 1] Apply input dropout
        x = self.input_dropout(x)

        # Conv preprocessing: (B, T, F) → (B, F, T) → conv → (B, T, 128)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn_conv(self.conv1(x)))
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # [FIX 1] Mean pool — no attention
        pooled = torch.mean(lstm_out, dim=1)

        return self.fc(pooled)


# --- Global State ---
model            = None
classes          = None
label_mapping    = None
mp_hands         = None
hands            = None

# Segment collection state
collecting           = False
gesture_frames       = []
consec_hand          = 0
consec_nohand        = 0
last_prediction_time = 0.0


def normalize_landmarks(landmarks_array: np.ndarray) -> np.ndarray:
    """Normalizes landmarks relative to the wrist (Landmark 0) and scales."""
    coords = landmarks_array.reshape(21, 3)
    wrist  = coords[0]
    centered = coords - wrist
    max_val  = np.max(np.abs(centered))
    if max_val > 0:
        centered = centered / max_val
    return centered.flatten().astype(np.float32)


def load_label_mapping():
    """Load label mapping from JSON file."""
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
    """Convert folder name (numeric class id) to human-readable label."""
    global label_mapping
    if label_mapping and folder_name in label_mapping:
        return label_mapping[folder_name]
    return f"SIGN_{folder_name}"


def initialize_dynamic_model():
    """Load trained model and initialize MediaPipe."""
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

    classes     = checkpoint['classes']
    num_classes = checkpoint['num_classes']
    input_size  = checkpoint.get('input_size', INPUT_SIZE)

    print("📊 Model Info:")
    print(f"   - Classes: {num_classes}")
    print(f"   - Device:  {DEVICE}")
    print(f"   - Test F1: {checkpoint['test_metrics']['f1']:.4f}")

    # [FIX 3] Read dropout from checkpoint so architecture always matches.
    # best_config is saved by train_dynamic_fsl.py inside the checkpoint.
    best_config = checkpoint.get('best_config', {})
    dropout     = best_config.get('Dropout', 0.5)

    # [FIX 1] Instantiate with the new smaller architecture
    model = ImprovedLSTMModel(
        input_size  = input_size,
        num_classes = num_classes,
        dropout     = dropout,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    print(f"   - Architecture: hidden=128, layers=2, dropout={dropout}, no attention")

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode       = False,
        max_num_hands           = 2,
        min_detection_confidence= 0.6,
        min_tracking_confidence = 0.6,
    )

    # Reset state
    collecting           = False
    gesture_frames       = []
    consec_hand          = 0
    consec_nohand        = 0
    last_prediction_time = 0.0

    print("✅ Initialization complete!")
    print("=" * 60 + "\n")


def extract_frame_features(frame: np.ndarray) -> tuple:
    """
    Extract 126-dim position features for one frame.
    Velocity is added later in predict_from_sequence().
    Returns: (features_126, hands_detected)
    """
    global hands

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # NOTE: frame_feats is 126 here (position only).
    # Velocity is computed in predict_from_sequence() over the full sequence.
    frame_feats    = np.zeros(126, dtype=np.float32)
    hands_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:
        hands_detected = True

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label

            raw_lms  = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32
            )
            norm_lms = normalize_landmarks(raw_lms)

            if label == 'Left':
                frame_feats[0:63]   = norm_lms
            else:
                frame_feats[63:126] = norm_lms

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

    return frame_feats, hands_detected


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample variable-length sequence to target_len using linear index spacing.
    Matches the linspace sampling used in extract_dynamic_features.py.
    seq: (T, 126)  →  returns: (target_len, 126)
    """
    T = seq.shape[0]
    if T == 0:
        return np.zeros((target_len, 126), dtype=np.float32)
    if T == target_len:
        return seq.astype(np.float32)

    idxs = np.linspace(0, T - 1, target_len).astype(int)
    return seq[idxs].astype(np.float32)


def predict_from_sequence(sequence_30: np.ndarray) -> dict:
    """
    Run model on a (30, 126) position sequence.
    Adds velocity features to produce (30, 252) before inference.
    Returns top-3 predictions.
    """
    global model, classes

    # Add velocity: concatenate frame-to-frame differences
    # Input:  (30, 126)  →  Output: (30, 252)
    if sequence_30.shape[1] == 126:
        velocity    = np.zeros_like(sequence_30)
        velocity[1:]= sequence_30[1:] - sequence_30[:-1]
        sequence_30 = np.concatenate([sequence_30, velocity], axis=1)

    sequence_tensor = torch.from_numpy(sequence_30).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output     = model(sequence_tensor)
        probs      = torch.softmax(output, dim=1)
        top3_probs, top3_idx = torch.topk(probs, k=min(3, len(classes)), dim=1)

        top3_probs = top3_probs.cpu().numpy()[0]
        top3_idx   = top3_idx.cpu().numpy()[0]

    return {
        'top1_label' : get_label(classes[top3_idx[0]]),
        'top1_conf'  : float(top3_probs[0]),
        'top3_labels': [get_label(classes[i]) for i in top3_idx],
        'top3_confs' : [float(p) for p in top3_probs],
        'is_ready'   : True,
    }


def update_and_maybe_predict(frame: np.ndarray) -> dict:
    """
    Call this for every camera frame.
    Collects a gesture segment and predicts only when the gesture ends.
    """
    global collecting, gesture_frames, consec_hand, consec_nohand, last_prediction_time

    feats, hands_detected = extract_frame_features(frame)

    now         = time.time()
    in_cooldown = (now - last_prediction_time) < COOLDOWN_SECONDS

    if hands_detected:
        consec_hand  += 1
        consec_nohand = 0
    else:
        consec_nohand += 1
        consec_hand   = 0

    # Start collecting when hands stable for START_HAND_FRAMES (and not in cooldown)
    if (not collecting) and (not in_cooldown) and (consec_hand >= START_HAND_FRAMES):
        collecting     = True
        gesture_frames = []
        gesture_frames.append(feats)

    elif collecting:
        if hands_detected:
            gesture_frames.append(feats)

        # End gesture when no hands for END_NOHAND_FRAMES consecutive frames
        if consec_nohand >= END_NOHAND_FRAMES:
            collecting = False

            if len(gesture_frames) >= MIN_GESTURE_FRAMES:
                seq    = np.stack(gesture_frames, axis=0)   # (T, 126)
                seq30  = resample_sequence(seq, SEQUENCE_LENGTH)
                result = predict_from_sequence(seq30)
                last_prediction_time = now
                gesture_frames = []
                return result
            else:
                gesture_frames = []
                return {
                    'top1_label' : 'Too short / ignored',
                    'top1_conf'  : 0.0,
                    'top3_labels': [],
                    'top3_confs' : [],
                    'is_ready'   : False,
                }

    # Default: not predicting yet
    return {
        'top1_label' : "Collecting..." if collecting else "Waiting...",
        'top1_conf'  : 0.0,
        'top3_labels': [],
        'top3_confs' : [],
        'is_ready'   : False,
        'debug'      : {
            'collecting'      : collecting,
            'frames_collected': int(len(gesture_frames)),
            'consec_hand'     : int(consec_hand),
            'consec_nohand'   : int(consec_nohand),
        }
    }


def reset_buffer():
    """Manual reset of gesture collection state."""
    global collecting, gesture_frames, consec_hand, consec_nohand
    collecting     = False
    gesture_frames = []
    consec_hand    = 0
    consec_nohand  = 0
    print("🔄 Gesture state reset")


def get_model_info() -> dict:
    """Return loaded model information."""
    global classes
    if model is None:
        return {'status': 'not_initialized'}

    return {
        'status'        : 'ready',
        'num_classes'   : len(classes),
        'device'        : str(DEVICE),
        'sequence_length': SEQUENCE_LENGTH,
        'segmentation'  : {
            'START_HAND_FRAMES' : START_HAND_FRAMES,
            'END_NOHAND_FRAMES' : END_NOHAND_FRAMES,
            'MIN_GESTURE_FRAMES': MIN_GESTURE_FRAMES,
            'COOLDOWN_SECONDS'  : COOLDOWN_SECONDS,
        }
    }