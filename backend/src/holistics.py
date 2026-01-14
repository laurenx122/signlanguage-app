import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import torch.nn.functional as F
from collections import deque

from tts.tts_engine import speak

# =========================
# CONFIG
# =========================
SEQ_LEN = 30
FEATURES = 255          # Pose + 2 Hands + Face subset
CONF_THRESHOLD = 0.70
STABLE_FRAMES = 8
SPEECH_DELAY = 1.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/lstm/lstm_model.pth"
LABELS_CSV = "data/raw/fsl105/labels.csv"

# =========================
# LOAD LABEL MAP
# =========================
df = pd.read_csv(LABELS_CSV)
LABEL_MAP = dict(zip(df["id"], df["label"]))

# =========================
# LSTM MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = LSTMModel(FEATURES, 128, len(LABEL_MAP))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded")

# =========================
# MEDIAPIPE HOLISTIC
# =========================
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# BUFFERS
# =========================
sequence = deque(maxlen=SEQ_LEN)
pred_buffer = deque(maxlen=STABLE_FRAMES)

last_spoken = ""
last_spoken_time = 0

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
    res = holistic.process(rgb)

    # =========================
    # DRAW LANDMARKS (VERY IMPORTANT)
    # =========================
    if res.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

    if res.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

    if res.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

    if res.face_landmarks:
        mp_draw.draw_landmarks(
            frame, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION
        )

    # =========================
    # FEATURE EXTRACTION
    # =========================
    features = []

    # POSE (33 x 3)
    if res.pose_landmarks:
        for lm in res.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 99)

    # HANDS (21 x 3 x 2)
    for hand in [res.left_hand_landmarks, res.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)

    # FACE (10 landmarks x 3)
    FACE_IDX = [1, 13, 14, 61, 291, 199, 152, 10, 9, 8]
    if res.face_landmarks:
        for idx in FACE_IDX:
            lm = res.face_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 30)

    sequence.append(features)

    predicted_text = ""

    # =========================
    # PREDICTION
    # =========================
    if len(sequence) == SEQ_LEN:
        X = torch.tensor(sequence, dtype=torch.float32)\
                .unsqueeze(0)\
                .to(DEVICE)

        with torch.no_grad():
            probs = F.softmax(model(X), dim=1)
            conf, idx = torch.max(probs, dim=1)

        if conf.item() >= CONF_THRESHOLD:
            predicted_text = LABEL_MAP.get(idx.item(), "")
            pred_buffer.append(predicted_text)
        else:
            pred_buffer.clear()

        # =========================
        # SPEECH (STABLE PREDICTION)
        # =========================
        if (
            len(pred_buffer) == STABLE_FRAMES
            and len(set(pred_buffer)) == 1
        ):
            current_time = time.time()
            if (
                predicted_text != last_spoken
                and current_time - last_spoken_time > SPEECH_DELAY
            ):
                speak(predicted_text)
                last_spoken = predicted_text
                last_spoken_time = current_time

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(
        frame,
        f"Sign: {predicted_text}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language to Speech", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
