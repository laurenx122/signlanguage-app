import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
DATASET_PATH = "data/raw/fsl105"
CSV_PATH = os.path.join(DATASET_PATH, "train.csv")
OUTPUT_PATH = "data/processed_sequences"

SEQ_LEN = 60
FEATURES = 126  # 2 hands × 21 landmarks × 3 coords

os.makedirs(OUTPUT_PATH, exist_ok=True)

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

df = pd.read_csv(CSV_PATH)

# =========================
# PROCESS VIDEOS
# =========================
for _, row in df.iterrows():
    video_path = os.path.join(DATASET_PATH, row["vid_path"])
    label = str(row["id_label"])

    if not os.path.exists(video_path):
        continue

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Initialize hands
        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label_hand = handedness.classification[0].label
                if label_hand == "Left":
                    left_hand = hand_landmarks
                elif label_hand == "Right":
                    right_hand = hand_landmarks

        features = []

        # -------- LEFT HAND --------
        if left_hand:
            base = left_hand.landmark[0]
            for lm in left_hand.landmark:
                features.extend([
                    lm.x - base.x,
                    lm.y - base.y,
                    lm.z - base.z
                ])
        else:
            features.extend([0.0] * 63)

        # -------- RIGHT HAND --------
        if right_hand:
            base = right_hand.landmark[0]
            for lm in right_hand.landmark:
                features.extend([
                    lm.x - base.x,
                    lm.y - base.y,
                    lm.z - base.z
                ])
        else:
            features.extend([0.0] * 63)

        frames.append(features)

    cap.release()

    # =========================
    # TEMPORAL SAMPLING
    # =========================
    if len(frames) < SEQ_LEN:
        continue

    # Uniform sampling (keeps motion)
    idxs = np.linspace(0, len(frames) - 1, SEQ_LEN).astype(int)
    sequence = np.array([frames[i] for i in idxs])

    # =========================
    # SAVE
    # =========================
    label_dir = os.path.join(OUTPUT_PATH, label)
    os.makedirs(label_dir, exist_ok=True)

    out_path = os.path.join(label_dir, f"{row.name}.npy")
    np.save(out_path, sequence)

print("✅ Two-hand SEQUENTIAL landmark extraction complete")
