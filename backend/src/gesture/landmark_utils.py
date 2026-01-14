import numpy as np

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(21, 3)

    wrist = landmarks[0, :2]
    landmarks[:, :2] -= wrist

    hand_size = np.linalg.norm(landmarks[9, :2])
    if hand_size > 0:
        landmarks[:, :2] /= hand_size

    landmarks[:, 2] -= landmarks[0, 2]

    return landmarks.flatten()
