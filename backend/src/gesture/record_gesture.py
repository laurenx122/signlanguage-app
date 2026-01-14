import cv2
import mediapipe as mp
import numpy as np
import os

# =========================
# CONFIG
# =========================
SEQ_LEN = 60
FEATURES = 126  # 2 hands × 21 landmarks × 3 coords
DATA_PATH = "data/processed_sequences"

# =========================
# Input label
# =========================
LABEL = input("Enter label for this gesture (e.g., A, B, C): ").strip()
LABEL_PATH = os.path.join(DATA_PATH, LABEL)
os.makedirs(LABEL_PATH, exist_ok=True)

# =========================
# Initialize MediaPipe Hands
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# Start webcam
# =========================
cap = cv2.VideoCapture(0)  # 0 = default webcam

frames = []

print("Recording gesture. Press 'q' to stop recording.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    features = []

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label_hand = handedness.classification[0].label
            if label_hand == "Left":
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks

    # LEFT HAND
    if left_hand:
        base = left_hand.landmark[0]
        for lm in left_hand.landmark:
            features.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])
    else:
        features.extend([0.0]*63)

    # RIGHT HAND
    if right_hand:
        base = right_hand.landmark[0]
        for lm in right_hand.landmark:
            features.extend([lm.x - base.x, lm.y - base.y, lm.z - base.z])
    else:
        features.extend([0.0]*63)

    frames.append(features)

    # Display webcam
    cv2.putText(frame, f"Recording {LABEL}...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Webcam", frame)

    # Stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# Check if enough frames
# =========================
if len(frames) < SEQ_LEN:
    print("❌ Not enough frames recorded! Try again.")
else:
    # Uniform sampling to SEQ_LEN
    idxs = np.linspace(0, len(frames)-1, SEQ_LEN).astype(int)
    sequence = np.array([frames[i] for i in idxs])

    # Save with next available index
    existing_files = [f for f in os.listdir(LABEL_PATH) if f.endswith(".npy")]
    existing_indexes = [int(os.path.splitext(f)[0]) for f in existing_files if f.split(".")[0].isdigit()]
    next_index = max(existing_indexes) + 1 if existing_indexes else 0
    filename = os.path.join(LABEL_PATH, f"{next_index}.npy")

    np.save(filename, sequence)
    print(f"✅ Gesture saved: {filename}")
