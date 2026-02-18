"""
test_webcam.py
Test FSL recognition using webcam
Real-time hand detection and sign prediction
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from pathlib import Path
import time
from collections import deque, Counter


# Optional TTS (speaks letters after a pause)
try:
    from tts.tts_engine import speak
except Exception:
    try:
        from src.tts.tts_engine import speak
    except Exception:
        def speak(_text: str):
            return


# ================= CONFIG =================
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_static" / "best_fsl_lstm_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Buffer settings
PREDICTION_BUFFER_SIZE = 10
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)   # stores top1 labels (raw)
confidence_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)   # stores top1 confidences (raw)

# Speaking behavior
pause_to_speak_s = 0.80

# Green-buffer rule (>=65% during hold)
confidence_threshold = 0.65

# NEW: Hold-to-queue rule
HOLD_TO_QUEUE_S = 0.2  # how long the same GREEN buffered letter must be held before queueing it

# Timing
commit_cooldown_s = 0.30      # minimum time between commits
gap_s = 0.12                  # NO-HANDS duration to allow repeated letters (e.g., "LL")

# State
letter_queue = deque()
last_committed = None
last_commit_time = 0.0

# No-hands tracking (spelling finished)
nohands_start = None
nohands_seen_since = None
gap_seen = True

# Hold tracking for queueing (prevents transition commits)
hold_letter = None
hold_start = None
hold_committed_for_letter = False  # ensures one queue per hold

_model = None
_hands = None
_inv_label_map = None
_mp_drawing = None


# ================= MODEL (MUST MATCH CHECKPOINT) =================
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

        # Must exist to load your checkpoint
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


# ================= LANDMARK HELPERS =================
def is_left_hand(hand_landmarks, handedness):
    if handedness and handedness.classification:
        return handedness.classification[0].label == "Left"
    return False


def mirror_landmarks_horizontal(landmarks):
    mirrored = landmarks.copy()
    for i in range(0, len(mirrored), 3):
        mirrored[i] = 1.0 - mirrored[i]
    return mirrored


def normalize_landmarks(landmarks):
    """Wrist-centered + scale normalization (matches your preprocessing/inference style)"""
    landmarks = np.array(landmarks).reshape(-1, 3)

    wrist = landmarks[0].copy()
    centered = landmarks - wrist

    distances = np.linalg.norm(centered, axis=1)
    hand_size = np.max(distances)
    if hand_size < 1e-6:
        hand_size = 1.0

    normed = centered / hand_size
    return normed.flatten()


# ================= INIT =================
def initialize_model():
    global _model, _hands, _inv_label_map, _mp_drawing

    print("🔄 Loading FSL model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    _inv_label_map = {v: k for k, v in checkpoint["class_to_idx"].items()}
    num_classes = len(checkpoint["classes"])
    feature_dim = checkpoint.get("feature_dim", 126)

    _model = LSTMGestureModel(
        input_size=feature_dim,
        hidden_size=checkpoint.get("hidden_size", 128),
        num_layers=checkpoint.get("num_layers", 2),
        num_classes=num_classes,
        dropout=checkpoint.get("dropout", 0.3),
    ).to(DEVICE)

    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    print("✅ Model loaded successfully")
    print(f"💾 Model path: {MODEL_PATH}")

    _hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe initialized")


def extract_landmarks(results):
    """2-hand extraction, mirrored left->right, wrist+scale normalized, padded to 126."""
    if not results.multi_hand_landmarks:
        return None

    all_landmarks = []

    for i in range(min(len(results.multi_hand_landmarks), 2)):
        hand_landmarks = results.multi_hand_landmarks[i]
        handedness = results.multi_handedness[i] if results.multi_handedness else None

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks, dtype=np.float32)

        if is_left_hand(hand_landmarks, handedness):
            landmarks = mirror_landmarks_horizontal(landmarks)

        landmarks = normalize_landmarks(landmarks)
        all_landmarks.extend(landmarks)

    while len(all_landmarks) < 126:
        all_landmarks.extend([0.0] * 63)

    return np.array(all_landmarks[:126], dtype=np.float32)


def predict_with_top2(landmarks):
    """Return top1/top2 labels and probabilities."""
    tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    top2_idx = np.argsort(probs)[-2:][::-1]
    i1, i2 = int(top2_idx[0]), int(top2_idx[1])

    p1, p2 = float(probs[i1]), float(probs[i2])
    c1, c2 = _inv_label_map[i1], _inv_label_map[i2]
    return c1, p1, c2, p2


def update_motion(_landmarks_vec):
    """
    Kept only because your UI displays Motion.
    We keep it but do NOT use it to block queueing.
    """
    return 0.0


def draw_info(frame, buffered_sign, top1, p1, top2, p2, motion, fps):
    # ---- UI MUST NOT CHANGE ----
    cv2.rectangle(frame, (10, 10), (900, 220), (0, 0, 0), -1)

    cv2.putText(frame, f"Buffered: {buffered_sign}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0) if buffered_sign != "UNKNOWN" else (0, 165, 255), 3)

    cv2.putText(frame, f"Top1: {top1} ({p1:.2f})   Top2: {top2} ({p2:.2f})   Margin: {(p1-p2):.2f}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Motion: {motion:.4f}   (threshold {0.020})",
                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    queued_text = "".join(list(letter_queue)) if len(letter_queue) else ""
    cv2.putText(frame, f"Queued: {queued_text}", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, "Press 'Q' to quit", (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    global last_committed, last_commit_time
    global nohands_start, nohands_seen_since, gap_seen
    global hold_letter, hold_start, hold_committed_for_letter

    initialize_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_counter = 0
    fps_start = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _hands.process(rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                _mp_drawing.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)

        landmarks = extract_landmarks(results)
        no_hands = (landmarks is None)

        top1, p1, top2, p2 = "UNKNOWN", 0.0, "UNKNOWN", 0.0
        motion = 0.0

        if not no_hands:
            top1, p1, top2, p2 = predict_with_top2(landmarks)
            motion = update_motion(landmarks)

            # Buffer RAW top1 + RAW confidence
            prediction_buffer.append(top1)
            confidence_buffer.append(p1)
        else:
            # No hands -> force UNKNOWN and make "no-hands pause" immediate
            prediction_buffer.clear()
            confidence_buffer.clear()
            prediction_buffer.append("UNKNOWN")
            confidence_buffer.append(0.0)

        # ---- Compute buffered label ----
        buffered_sign_raw = Counter(prediction_buffer).most_common(1)[0][0] if len(prediction_buffer) else "UNKNOWN"

        # ---- Compute buffered confidence for that buffered label ----
        conf_vals = [
            c for (lbl, c) in zip(prediction_buffer, confidence_buffer)
            if lbl == buffered_sign_raw
        ]
        buffered_conf = float(np.mean(conf_vals)) if conf_vals else 0.0

        # ---- "Green buffered" rule ----
        buffered_sign = buffered_sign_raw if (buffered_sign_raw != "UNKNOWN" and buffered_conf >= confidence_threshold) else "UNKNOWN"

        now = time.time()

        # ---------- NO-HANDS tracking (ONLY this can end spelling / speak / clear) ----------
        if no_hands:
            if nohands_seen_since is None:
                nohands_seen_since = now
            elif (now - nohands_seen_since) >= gap_s:
                gap_seen = True

            if nohands_start is None:
                nohands_start = now
        else:
            nohands_seen_since = None
            nohands_start = None

        # ---------- HOLD-TO-QUEUE (0.8s) ----------
        # We only start/continue the hold timer when buffered_sign is GREEN.
        if buffered_sign != "UNKNOWN" and not no_hands:
            if buffered_sign != hold_letter:
                hold_letter = buffered_sign
                hold_start = now
                hold_committed_for_letter = False
            else:
                # same letter still green; check hold duration
                if hold_start is not None:
                    held_for = now - hold_start
                    if (not hold_committed_for_letter) and held_for >= HOLD_TO_QUEUE_S:
                        # queue it once for this hold (with cooldown + repeat rules)
                        if (now - last_commit_time) >= commit_cooldown_s:
                            if buffered_sign != last_committed or gap_seen:
                                letter_queue.append(buffered_sign)
                                last_committed = buffered_sign
                                last_commit_time = now
                                gap_seen = False
                        hold_committed_for_letter = True
        else:
            # Not green (or no hands): reset hold timer so transitions don't commit
            hold_letter = None
            hold_start = None
            hold_committed_for_letter = False

        # ---------- Speak + CLEAR queue ONLY after sustained NO HANDS ----------
        if nohands_start is not None and (now - nohands_start) >= pause_to_speak_s:
            if len(letter_queue) > 0:
                to_speak = list(letter_queue)
                letter_queue.clear()
                for ch in to_speak:
                    speak(ch)
            nohands_start = None

        # FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()

        draw_info(frame, buffered_sign, top1, p1, top2, p2, motion, fps)
        cv2.imshow("FSL Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    _hands.close()


if __name__ == "__main__":
    main()
