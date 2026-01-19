"""
test_webcam.py
Test FSL recognition using webcam
Real-time hand detection and sign prediction
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from pathlib import Path
import time

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm" / "best_fsl_lstm_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables
_model = None
_hands = None
_inv_label_map = None
_mp_drawing = None


class LSTMGestureModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMGestureModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        
        out = self.dropout(self.relu(self.batch_norm1(self.fc1(out))))
        out = self.dropout(self.relu(self.batch_norm2(self.fc2(out))))
        return self.fc3(out)


def is_left_hand(hand_landmarks, handedness):
    """Determine if detected hand is left hand"""
    if handedness and handedness.classification:
        label = handedness.classification[0].label
        return label == 'Left'
    return False


def mirror_landmarks_horizontal(landmarks):
    """Mirror landmarks horizontally for left->right conversion"""
    mirrored = landmarks.copy()
    for i in range(0, len(mirrored), 3):
        mirrored[i] = 1.0 - mirrored[i]
    return mirrored


def normalize_landmarks(landmarks):
    """Normalize landmarks (wrist-centered, scale-invariant)"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    
    # Center at wrist
    wrist = landmarks[0].copy()
    landmarks_centered = landmarks - wrist
    
    # Scale by hand size
    distances = np.linalg.norm(landmarks_centered, axis=1)
    hand_size = np.max(distances)
    
    if hand_size < 1e-6:
        hand_size = 1.0
    
    landmarks_normalized = landmarks_centered / hand_size
    
    return landmarks_normalized.flatten()


def initialize_model():
    """Initialize model and MediaPipe"""
    global _model, _hands, _inv_label_map, _mp_drawing
    
    print("🔄 Loading FSL model...")
    
    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    _inv_label_map = {v: k for k, v in checkpoint['class_to_idx'].items()}
    num_classes = len(checkpoint['classes'])
    feature_dim = checkpoint.get('feature_dim', 126)
    
    print(f"📊 Classes: {checkpoint['classes']}")
    print(f"📊 Feature dimension: {feature_dim}")
    
    _model = LSTMGestureModel(
        input_size=feature_dim,
        hidden_size=checkpoint.get('hidden_size', 128),
        num_layers=checkpoint.get('num_layers', 2),
        num_classes=num_classes,
        dropout=checkpoint.get('dropout', 0.3)
    ).to(DEVICE)
    
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()
    print("✅ Model loaded successfully")
    
    # Initialize MediaPipe
    _hands = mp.solutions.hands.Hands(
        static_image_mode=False,  # Video mode
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    _mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe initialized")


def extract_landmarks(results):
    """Extract landmarks from MediaPipe results"""
    if not results.multi_hand_landmarks:
        return None
    
    all_landmarks = []
    
    # Process up to 2 hands
    for i in range(min(len(results.multi_hand_landmarks), 2)):
        hand_landmarks = results.multi_hand_landmarks[i]
        handedness = results.multi_handedness[i] if results.multi_handedness else None
        
        # Extract landmarks
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Convert left to right
        if is_left_hand(hand_landmarks, handedness):
            landmarks = mirror_landmarks_horizontal(landmarks)
        
        # Normalize
        landmarks = normalize_landmarks(landmarks)
        all_landmarks.extend(landmarks)
    
    # Pad to 126 if only one hand
    while len(all_landmarks) < 126:
        all_landmarks.extend([0.0] * 63)
    
    return np.array(all_landmarks[:126], dtype=np.float32)


def predict(landmarks, confidence_threshold=0.6):
    """Make prediction from landmarks"""
    tensor = torch.FloatTensor(landmarks).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    
    pred_class = _inv_label_map[idx.item()]
    confidence = conf.item()
    
    if confidence < confidence_threshold:
        return "UNKNOWN", confidence
    
    return pred_class, confidence


def draw_info(frame, prediction, confidence, fps):
    """Draw prediction info on frame"""
    # Background for text
    cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
    
    # Prediction
    color = (0, 255, 0) if prediction != "UNKNOWN" else (0, 165, 255)
    cv2.putText(frame, f"Sign: {prediction}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Confidence
    cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press 'Q' to quit", (frame.shape[1] - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    """Main webcam testing loop"""
    global _hands, _mp_drawing
    
    print("="*60)
    print("🎥 FSL Webcam Testing")
    print("="*60)
    
    # Initialize model
    initialize_model()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n📹 Webcam opened successfully")
    print("👋 Show your FSL signs to the camera")
    print("🔤 Press 'Q' to quit\n")
    
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Flip frame horizontally (mirror view)
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = _hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                _mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp.solutions.hands.HAND_CONNECTIONS,
                    _mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    _mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        
        # Extract landmarks and predict
        prediction = "UNKNOWN"
        confidence = 0.0
        
        landmarks = extract_landmarks(results)
        if landmarks is not None:
            prediction, confidence = predict(landmarks, confidence_threshold=0.6)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Draw info on frame
        draw_info(frame, prediction, confidence, fps)
        
        # Display
        cv2.imshow('FSL Webcam Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    _hands.close()
    
    print("\n✅ Webcam test completed")
    print("="*60)


if __name__ == '__main__':
    main()