"""
extract_video_frames.py
Extract landmark sequences from FSL video dataset
TRAIN.CSV driven (correct version)
"""

import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle

# =====================================================
# PATHS
# =====================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent

VIDEO_ROOT = PROJECT_ROOT / "data/raw/fsl105"
LABELS_CSV = VIDEO_ROOT / "labels.csv"
TRAIN_CSV  = VIDEO_ROOT / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/fsl_dynamic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# CONFIG
# =====================================================
TARGET_CLASSES = None        # None = all labels
AUGMENT_WITH_MIRRORING = True

MAX_SEQUENCE_LENGTH = 30
TARGET_FPS = 10

# =====================================================
# MEDIAPIPE
# =====================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# LABEL LOADING (STABLE)
# =====================================================
def load_label_mapping():
    df = pd.read_csv(LABELS_CSV)

    label_to_idx = {}
    idx_to_label = {}

    for _, row in df.iterrows():
        label_id = int(row["id"])
        label = row["label"]

        if TARGET_CLASSES is not None and label_id not in TARGET_CLASSES:
            continue

        label_to_idx[label] = label_id
        idx_to_label[label_id] = label

    print(f"📋 Loaded {len(label_to_idx)} labels")
    return label_to_idx, idx_to_label

# =====================================================
# NORMALIZATION
# =====================================================
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0].copy()
    landmarks -= wrist

    scale = np.max(np.linalg.norm(landmarks, axis=1))
    if scale < 1e-6:
        scale = 1.0

    return (landmarks / scale).flatten()

# =====================================================
# MIRRORING
# =====================================================
def mirror_hand(landmarks):
    if np.all(landmarks == 0):
        return landmarks
    mirrored = landmarks.copy()
    mirrored[0::3] *= -1
    return mirrored

def mirror_sequence(sequence):
    out = []
    for frame in sequence:
        h1 = mirror_hand(frame[:63])
        h2 = mirror_hand(frame[63:])
        out.append(np.concatenate([h1, h2]))
    return np.array(out, dtype=np.float32)

# =====================================================
# VIDEO → SEQUENCE
# =====================================================
def extract_sequence(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, int(fps / TARGET_FPS))

    seq = []
    frame_idx = 0

    while len(seq) < MAX_SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if not res.multi_hand_landmarks:
            continue

        frame_feats = []
        for hand in res.multi_hand_landmarks[:2]:
            lm = []
            for p in hand.landmark:
                lm.extend([p.x, p.y, p.z])
            frame_feats.extend(normalize_landmarks(lm))

        while len(frame_feats) < 126:
            frame_feats.extend([0.0] * 63)

        seq.append(frame_feats[:126])

    cap.release()

    if len(seq) == 0:
        return None

    while len(seq) < MAX_SEQUENCE_LENGTH:
        seq.append([0.0] * 126)

    return np.array(seq, dtype=np.float32)

# =====================================================
# MAIN PIPELINE
# =====================================================
def process_dataset():
    label_to_idx, idx_to_label = load_label_mapping()
    train_df = pd.read_csv(TRAIN_CSV)

    X, y = [], []
    failed = 0

    print("\n🎬 Processing TRAIN.CSV videos")

    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        label = row["label"]
        if label not in label_to_idx:
            continue

        label_idx = label_to_idx[label]
        video_path = VIDEO_ROOT / row["vid_path"]

        seq = extract_sequence(video_path)
        if seq is None:
            failed += 1
            continue

        X.append(seq)
        y.append(label_idx)

        if AUGMENT_WITH_MIRRORING:
            X.append(mirror_sequence(seq))
            y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    np.save(OUTPUT_DIR / "sequences_X.npy", X)
    np.save(OUTPUT_DIR / "labels_y.npy", y)

    with open(OUTPUT_DIR / "label_mapping.pkl", "wb") as f:
        pickle.dump({
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label
        }, f)

    print("\n" + "="*60)
    print("✅ EXTRACTION COMPLETE")
    print("="*60)
    print(f"Samples: {len(X)}")
    print(f"Classes: {len(label_to_idx)}")
    print(f"Failed videos: {failed}")
    print(f"Shape: {X.shape}")
    print("="*60)

    hands.close()

# =====================================================
if __name__ == "__main__":
    process_dataset()
