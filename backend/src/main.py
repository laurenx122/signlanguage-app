import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
from collections import deque
from tts.tts_engine import speak
import torch.nn.functional as F

# =========================
# LANDMARK NORMALIZATION
# =========================
def normalize_hand_landmarks(hand_landmarks):
    coords = []

    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    coords = np.array(coords)

    # Center at wrist
    wrist = coords[0]
    coords = coords - wrist

    # Scale by hand size
    hand_size = np.linalg.norm(coords[9][:2])
    if hand_size > 0:
        coords[:, :3] /= hand_size

    return coords.flatten().tolist()

# =========================
# CONFIG
# =========================
SEQ_LEN = 60
FEATURES = 126
CONF_THRESHOLD = 0.70
SPEECH_DELAY = 1.5
SMOOTHING_WINDOW = 8
UNKNOWN_LABEL = "Unknown"
STABLE_FRAMES_REQUIRED = 12   # ~0.4 sec @ 30 FPS
MOTION_THRESHOLD = 0.015     # lower = stricter


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/lstm/lstm_model.pth"
LABELS_CSV = "data/raw/fsl105/labels.csv"

df = pd.read_csv(LABELS_CSV)

# id (int) -> label (text)
IDX_TO_LABEL = dict(zip(df["id"], df["label"]))



# LABEL_MAP_PATH = "models/lstm/label_map.npy"

# # =========================
# # LOAD LABEL MAP
# # =========================
# label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
# IDX_TO_LABEL = {v: k for k, v in label_map.items()}

# =========================
# HELPER FUNCTIONS
# =========================
#HAND MOTION MEASUREMNT
def compute_motion(curr, prev):
    if prev is None:
        return float("inf")
    curr = np.array(curr)
    prev = np.array(prev)
    return np.mean(np.abs(curr - prev))


# =========================
# MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = LSTMModel(FEATURES, 128, len(IDX_TO_LABEL))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded")

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# =========================
# BUFFERS
# =========================
sequence = deque(maxlen=SEQ_LEN)
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
last_spoken = ""
last_spoken_time = 0
stable_count = 0
last_predicted_idx = None
prev_features = None


# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
print("🎥 Webcam started — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    features = []

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            if handedness.classification[0].label == "Left":
                left_hand = hand_lm
            else:
                right_hand = hand_lm

    #IF NOT LEFT AND RIGHT HANDS
    if not left_hand and not right_hand:
        prediction_buffer.clear()
        predicted_text = UNKNOWN_LABEL
        conf = 0.0


    # LEFT HAND
    if left_hand:
        mp_draw.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
        features.extend(normalize_hand_landmarks(left_hand))
    else:
        features.extend([0.0] * 63)

    # RIGHT HAND
    if right_hand:
        mp_draw.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)
        features.extend(normalize_hand_landmarks(right_hand))
    else:
        features.extend([0.0] * 63)

    sequence.append(features)

    predicted_text = UNKNOWN_LABEL
    conf = 0.0

    # =========================
    # PREDICTION
    # =========================
    if len(sequence) == SEQ_LEN:
        X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = F.softmax(model(X), dim=1)
            conf_tensor, idx_tensor = torch.max(probs, dim=1)

        conf = conf_tensor.item()
        idx = idx_tensor.item()

        # Motion estimation
        motion = compute_motion(features, prev_features)
        prev_features = features

        if conf >= CONF_THRESHOLD:
            if idx == last_predicted_idx and motion < MOTION_THRESHOLD:
                stable_count += 1
            else:
                stable_count = 0

            last_predicted_idx = idx
        else:
            stable_count = 0
            last_predicted_idx = None
            predicted_text = UNKNOWN_LABEL

        # CONFIRM SIGN
        if stable_count >= STABLE_FRAMES_REQUIRED:
            predicted_text = IDX_TO_LABEL.get(idx, UNKNOWN_LABEL)

            current_time = time.time()
            if (
                predicted_text != last_spoken
                and current_time - last_spoken_time > SPEECH_DELAY
            ):
                speak(predicted_text)
                last_spoken = predicted_text
                last_spoken_time = current_time
        else:
            predicted_text = UNKNOWN_LABEL


    else:
        # No hands → reset everything
        prediction_buffer.clear()
        predicted_text = UNKNOWN_LABEL
        conf = 0.0

    # =========================
    # SPEECH OUTPUT
    # =========================
    current_time = time.time()

    if (
        predicted_text != UNKNOWN_LABEL
        and predicted_text != last_spoken
        and conf >= 0.60
        and current_time - last_spoken_time > 1.5
    ):
        speak(predicted_text)
        last_spoken = predicted_text
        last_spoken_time = current_time


    # =========================
    # DISPLAY
    # =========================
    color = (0, 255, 0) if conf > CONF_THRESHOLD else (0, 0, 255)

    cv2.putText(
        frame,
        f"Sign: {predicted_text} ({conf*100:.1f}%)",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Sign Language to Speech", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
