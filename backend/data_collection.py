import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def collect_sign_data(sign_name, num_samples=100):
    """
    Collect hand landmark data for a specific sign
    """
    cap = cv2.VideoCapture(0)
    data = []
    
    print(f"Collecting data for: {sign_name}")
    print("Press 'c' to capture, 'q' to quit")
    
    while len(data) < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Pad with zeros if only one hand detected
            if len(landmarks) < 126:  # 21 landmarks * 3 coordinates * 2 hands
                landmarks.extend([0] * (126 - len(landmarks)))
            
            data.append(landmarks)
            print(f"Collected {len(data)}/{num_samples}")
        
        # Display
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save data
    np.save(f'backend/data/{sign_name}.npy', data)
    cap.release()
    cv2.destroyAllWindows()

# Usage: collect_sign_data('hello', 100)