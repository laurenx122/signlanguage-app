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

SEQ_LEN = 30

POSE_FEATURES = 33 * 3
HAND_FEATURES = 21 * 3 * 2
FACE_FEATURES = 10 * 3   # selected points only

TOTAL_FEATURES = POSE_FEATURES + HAND_FEATURES + FACE_FEATURES

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# MEDIAPIPE HOLISTIC
# =========================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Face indices (mouth + chin + forehead)
FACE_IDX = [1, 13, 14, 61, 291, 199, 152, 10, 9, 8]

df = pd.read_csv(CSV_PATH)

print("🚀 Starting holistic landmark extraction...")

for _, row in df.iterrows():
    video_path = os.path.join(DATASET_PATH, row["vid_path"])
    label = str(row["id_label"])

    if not os.path.exists(video_path):
        continue

    cap = cv2.VideoCapture(video_path)
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        frame_features = []

        # =========================
        # POSE
        # =========================
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_features.extend([lm.x, lm.y, lm.z])
        else:
            frame_features.extend([0.0] * POSE_FEATURES)

        # =========================
        # HANDS
        # =========================
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                for lm in hand_landmarks.landmark:
                    frame_features.extend([lm.x, lm.y, lm.z])
            else:
                frame_features.extend([0.0] * 63)

        # =========================
        # FACE (subset)
        # =========================
        if results.face_landmarks:
            for idx in FACE_IDX:
                lm = results.face_landmarks.landmark[idx]
                frame_features.extend([lm.x, lm.y, lm.z])
        else:
            frame_features.extend([0.0] * FACE_FEATURES)

        sequence.append(frame_features)

    cap.release()

    if len(sequence) < SEQ_LEN:
        continue

    sequence = np.array(sequence[:SEQ_LEN])

    label_dir = os.path.join(OUTPUT_PATH, label)
    os.makedirs(label_dir, exist_ok=True)

    np.save(
        os.path.join(label_dir, f"{row.name}.npy"),
        sequence
    )

print("✅ Holistic landmark extraction complete")
print("Final feature size:", TOTAL_FEATURES)
