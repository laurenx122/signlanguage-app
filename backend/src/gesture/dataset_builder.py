import os
import cv2
import mediapipe as mp
import numpy as np

DATASET_PATH = "data/raw/fsl-dataset"
OUTPUT_PATH = "data/processed"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

os.makedirs(OUTPUT_PATH, exist_ok=True)

labels = sorted(os.listdir(DATASET_PATH))
label_map = {label: idx for idx, label in enumerate(labels)}

X = []
y = []

total_images = 0
detected = 0

for label in labels:
    class_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(class_dir):
        continue

    print(f"Processing class: {label}")

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        total_images += 1

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            detected += 1
            hand = results.multi_hand_landmarks[0]

            keypoints = []
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            X.append(keypoints)
            y.append(label_map[label])

print("Total images read:", total_images)
print("Hands detected:", detected)

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("❌ No hand landmarks detected. Dataset not suitable for MediaPipe.")
else:
    np.save(os.path.join(OUTPUT_PATH, "X.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)
    print("✅ Dataset processing complete")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
