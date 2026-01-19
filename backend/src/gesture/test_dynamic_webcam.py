"""
test_dynamic_webcam.py
Test dynamic sign recognition with webcam
"""

import cv2
import time
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.gesture.fsl_dynamic_inference import (
    initialize_dynamic_model,
    add_frame_to_buffer,
    predict_dynamic_sign,
    reset_buffer
)


def main():
    print("="*60)
    print("🎥 FSL Dynamic Sign Recognition - Webcam Test")
    print("="*60)
    
    # Initialize model
    initialize_dynamic_model()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n📹 Webcam opened")
    print("👋 Perform FSL signs continuously")
    print("🔤 Press 'Q' to quit")
    print("🔄 Press 'R' to reset buffer")
    print("="*60)
    
    fps_counter = 0
    fps_start = time.time()
    fps = 0
    
    current_prediction = "WAITING"
    current_confidence = 0.0
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Add frame to buffer
        added = add_frame_to_buffer(frame)
        
        # Predict every 5 frames
        if frame_count % 5 == 0:
            result = predict_dynamic_sign(confidence_threshold=0.7)
            current_prediction = result['prediction']
            current_confidence = result['confidence']
        
        frame_count += 1
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:
            fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()
        
        # Draw info
        cv2.rectangle(frame, (10, 10), (500, 150), (0, 0, 0), -1)
        
        color = (0, 255, 0) if current_prediction not in ["UNKNOWN", "COLLECTING", "WAITING"] else (0, 165, 255)
        cv2.putText(frame, f"Sign: {current_prediction}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.putText(frame, f"Confidence: {current_confidence:.2%}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Q: Quit | R: Reset", (frame.shape[1] - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('FSL Dynamic Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("🔄 Resetting buffer...")
            reset_buffer()
            current_prediction = "WAITING"
            current_confidence = 0.0
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed")


if __name__ == '__main__':
    main()