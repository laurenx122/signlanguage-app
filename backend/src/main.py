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
# CONFIG
# =========================
SEQ_LEN = 60
FEATURES = 126
CONF_THRESHOLD = 0.70
SPEECH_DELAY = 1.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/lstm/lstm_model.pth"
LABELS_CSV = "data/raw/fsl105/labels.csv"

# =========================
# LOAD LABELS
# =========================
df = pd.read_csv(LABELS_CSV)
LABEL_MAP = dict(zip(df["id"], df["label"]))

# =========================
# MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
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

    # LEFT HAND
    if left_hand:
        mp_draw.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
        base = left_hand.landmark[0]
        for lm in left_hand.landmark:
            features.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])
    else:
        features.extend([0.0] * 63)

    # RIGHT HAND
    if right_hand:
        mp_draw.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)
        base = right_hand.landmark[0]
        for lm in right_hand.landmark:
            features.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])
    else:
        features.extend([0.0] * 63)

    sequence.append(features)
    predicted_text = ""

    if len(sequence) == SEQ_LEN:
        X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = F.softmax(model(X), dim=1)
            conf, idx = torch.max(probs, dim=1)

        if conf.item() > CONF_THRESHOLD:
            predicted_text = LABEL_MAP.get(idx.item(), "")

            current_time = time.time()
            if (
                predicted_text
                and predicted_text != last_spoken
                and current_time - last_spoken_time > SPEECH_DELAY
            ):
                speak(predicted_text)
                last_spoken = predicted_text
                last_spoken_time = current_time

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
