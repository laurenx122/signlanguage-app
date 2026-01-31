"""
test_dynamic_webcam_14.py
Real-time testing for 14-gesture dynamic model
Location: D:/SMS/backend/src/gesture/test_dynamic_webcam_14.py
"""

import cv2
import time
import sys
from pathlib import Path
from collections import deque, Counter
import numpy as np

# Import inference functions
from fsl_dynamic_inference import (
    initialize_dynamic_model,
    add_frame_to_buffer,
    predict_dynamic_sign,
    reset_buffer,
    get_buffer_info
)


class LongGestureStabilizer:
    """
    Stabilizer for multi-movement gestures (optimized for 14 gestures)
    """
    
    def __init__(
        self,
        confidence_threshold=0.50,      # Adjusted threshold
        stability_window=20,
        min_stable_count=12,
        hold_duration=1.5,
        min_gesture_frames=15,
        max_no_hands_frames=10
    ):
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.min_stable_count = min_stable_count
        self.hold_duration = hold_duration
        self.min_gesture_frames = min_gesture_frames
        self.max_no_hands_frames = max_no_hands_frames
        
        # Prediction tracking
        self.prediction_history = deque(maxlen=stability_window)
        self.confidence_history = deque(maxlen=stability_window)
        
        # Current stable result
        self.current_stable = None
        self.stable_since = None
        self.stable_confidence = 0.0
        
        # Gesture state
        self.gesture_active = False
        self.gesture_start_time = None
        self.no_hands_counter = 0
        self.total_frames_seen = 0
        
    def stabilize(self, prediction, confidence, top3, hands_detected=True):
        """Stabilize predictions with top-3 awareness"""
        current_time = time.time()
        
        self.total_frames_seen += 1
        
        # Hands detection state
        if hands_detected:
            self.no_hands_counter = 0
            if not self.gesture_active:
                self.gesture_active = True
                self.gesture_start_time = current_time
                self.total_frames_seen = 0
        else:
            self.no_hands_counter += 1
        
        gesture_ended = (self.no_hands_counter > self.max_no_hands_frames)
        
        # Holding stable result
        if self.current_stable and self.stable_since:
            time_held = current_time - self.stable_since
            
            if time_held < self.hold_duration:
                return {
                    'prediction': self.current_stable,
                    'confidence': self.stable_confidence,
                    'status': 'LOCKED',
                    'time_remaining': self.hold_duration - time_held,
                    'gesture_duration': current_time - self.gesture_start_time if self.gesture_start_time else 0,
                    'raw_prediction': prediction,
                    'raw_confidence': confidence,
                    'top3': top3
                }
            else:
                if gesture_ended:
                    self._reset_for_new_gesture()
                
                return {
                    'prediction': self.current_stable,
                    'confidence': self.stable_confidence,
                    'status': 'COMPLETE',
                    'raw_prediction': prediction,
                    'raw_confidence': confidence,
                    'top3': top3
                }
        
        # Gesture ended - analyze
        if gesture_ended and self.gesture_active and len(self.prediction_history) >= self.min_gesture_frames:
            result = self._analyze_gesture()
            
            if result:
                self.current_stable = result['prediction']
                self.stable_since = current_time
                self.stable_confidence = result['confidence']
                self.gesture_active = False
                
                return {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'status': 'STABLE',
                    'votes': result['votes'],
                    'gesture_duration': current_time - self.gesture_start_time if self.gesture_start_time else 0,
                    'raw_prediction': prediction,
                    'raw_confidence': confidence,
                    'top3': top3
                }
        
        # Reset if hands gone
        if gesture_ended and self.gesture_active:
            self._reset_for_new_gesture()
            return {
                'prediction': 'WAITING',
                'confidence': 0.0,
                'status': 'RESET',
                'raw_prediction': prediction,
                'raw_confidence': confidence,
                'top3': top3
            }
        
        # Collecting predictions
        if self.gesture_active and hands_detected:
            if confidence >= self.confidence_threshold and prediction != "UNKNOWN":
                self.prediction_history.append(prediction)
                self.confidence_history.append(confidence)
            
            if len(self.prediction_history) > 0:
                vote_counts = Counter(self.prediction_history)
                most_common, vote_count = vote_counts.most_common(1)[0]
                
                return {
                    'prediction': most_common,
                    'confidence': confidence,
                    'status': 'COLLECTING',
                    'buffer': f"{len(self.prediction_history)}/{self.min_gesture_frames}",
                    'votes': f"{vote_count}/{len(self.prediction_history)}",
                    'gesture_duration': current_time - self.gesture_start_time if self.gesture_start_time else 0,
                    'raw_prediction': prediction,
                    'raw_confidence': confidence,
                    'top3': top3
                }
        
        return {
            'prediction': 'WAITING',
            'confidence': 0.0,
            'status': 'WAITING',
            'raw_prediction': prediction,
            'raw_confidence': confidence,
            'top3': top3
        }
    
    def _analyze_gesture(self):
        """Analyze complete gesture"""
        if len(self.prediction_history) < self.min_gesture_frames:
            return None
        
        # Majority vote
        vote_counts = Counter(self.prediction_history)
        most_common, vote_count = vote_counts.most_common(1)[0]
        
        consensus_ratio = vote_count / len(self.prediction_history)
        
        if consensus_ratio >= 0.6:
            avg_confidence = np.mean([
                conf for pred, conf in zip(self.prediction_history, self.confidence_history)
                if pred == most_common
            ])
            
            return {
                'prediction': most_common,
                'confidence': avg_confidence,
                'votes': f"{vote_count}/{len(self.prediction_history)}"
            }
        
        # Weighted voting (second half more important)
        second_half_start = len(self.prediction_history) // 2
        second_half = list(self.prediction_history)[second_half_start:]
        
        if len(second_half) >= self.min_gesture_frames // 2:
            vote_counts_2nd = Counter(second_half)
            most_common_2nd, vote_count_2nd = vote_counts_2nd.most_common(1)[0]
            
            if vote_count_2nd / len(second_half) >= 0.55:
                avg_confidence = np.mean([
                    conf for pred, conf in zip(
                        list(self.prediction_history)[second_half_start:],
                        list(self.confidence_history)[second_half_start:]
                    ) if pred == most_common_2nd
                ])
                
                return {
                    'prediction': most_common_2nd,
                    'confidence': avg_confidence,
                    'votes': f"{vote_count_2nd}/{len(second_half)} (2nd half)"
                }
        
        return None
    
    def _reset_for_new_gesture(self):
        """Reset for new gesture"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.gesture_active = False
        self.gesture_start_time = None
        self.no_hands_counter = 0
        self.total_frames_seen = 0
    
    def manual_reset(self):
        """Manual reset"""
        self._reset_for_new_gesture()
        self.current_stable = None
        self.stable_since = None
        self.stable_confidence = 0.0


def draw_ui(frame, result, hands_detected, fps, buffer_info):
    """Draw UI with enhanced information"""
    h, w, _ = frame.shape

    prediction = result['prediction']
    confidence = result['confidence']
    status = result['status']
    top3 = result.get('top3', [])

    status_colors = {
        'WAITING': (180, 180, 180),
        'COLLECTING': (0, 165, 255),
        'STABLE': (0, 255, 0),
        'LOCKED': (0, 255, 0),
        'COMPLETE': (100, 200, 100),
        'RESET': (0, 200, 255)
    }

    color = status_colors.get(status, (255, 255, 255))

    # Top progress bar
    if 'buffer' in result:
        cur, total = map(int, result['buffer'].split('/'))
        progress = min(cur / total, 1.0)
        bar_w = int(w * 0.6 * progress)

        cv2.rectangle(frame, (20, 15), (20 + int(w * 0.6), 35), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 15), (20 + bar_w, 35), (0, 165, 255), -1)
        cv2.putText(frame, f"{cur}/{total}", (25, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Main prediction box
    box_x, box_y = 20, h - 230
    box_w, box_h = 500, 200

    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)

    # Main prediction
    cv2.putText(frame, prediction, (box_x + 15, box_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"{confidence:.1%}", (box_x + 15, box_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    cv2.putText(frame, status, (box_x + 350, box_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Top-3 predictions
    if len(top3) > 0:
        cv2.putText(frame, "Top-3:", (box_x + 15, box_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        y_offset = 140
        for i, (label, conf) in enumerate(top3[:3]):
            text = f"{i+1}. {label} ({conf:.1%})"
            cv2.putText(frame, text, (box_x + 25, box_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_offset += 20

    # Buffer info
    cv2.putText(frame, f"Buffer: {buffer_info['current']}/{buffer_info['max']}", 
                (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # FPS + Hand indicator
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    cv2.circle(frame, (120, 55), 8, hand_color, -1)


def main():
    print("="*70)
    print("🎥 FSL - 14 GESTURE DYNAMIC RECOGNITION")
    print("   GOOD MORNING, HELLO, THANK YOU, etc.")
    print("="*70)
    
    # Initialize model
    initialize_dynamic_model()
    
    # Create stabilizer
    stabilizer = LongGestureStabilizer(
        confidence_threshold=0.50,
        stability_window=30,
        min_stable_count=18,
        hold_duration=1.5,
        min_gesture_frames=15,
        max_no_hands_frames=15
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n✅ Webcam ready")
    print("\n📋 Instructions:")
    print("  1. Position hands in view")
    print("  2. Perform complete gesture")
    print("  3. System collects predictions")
    print("  4. Drop hands when done")
    print("  5. Result shown after analysis")
    print("\n💡 Tips:")
    print("  • Complete ALL movements")
    print("  • Keep hands visible")
    print("  • Natural gesture speed")
    print("\n⌨️  Controls:")
    print("  Q - Quit")
    print("  R - Reset")
    print("="*70)
    
    fps_counter, fps_start, fps = 0, time.time(), 0
    frame_count = 0
    
    result = {
        'prediction': 'WAITING',
        'confidence': 0.0,
        'status': 'WAITING',
        'top3': []
    }
    hands_detected = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Add frame to buffer
        hands_detected = add_frame_to_buffer(frame)
        
        # Get buffer info
        buffer_info = get_buffer_info()
        
        # Predict every 2 frames
        if frame_count % 2 == 0:
            raw_result = predict_dynamic_sign()
            
            top1_label = raw_result['top1_label']
            top1_conf = raw_result['top1_conf']
            top3 = raw_result['top3']
            
            # Stabilize
            result = stabilizer.stabilize(
                top1_label,
                top1_conf,
                top3,
                hands_detected=hands_detected
            )
        
        frame_count += 1
        
        # FPS calculation
        fps_counter += 1
        if fps_counter >= 30:
            fps = fps_counter / (time.time() - fps_start)
            fps_counter, fps_start = 0, time.time()
        
        # Draw UI
        draw_ui(frame, result, hands_detected, fps, buffer_info)
        
        cv2.imshow('FSL - 14 Gesture Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            stabilizer.manual_reset()
            reset_buffer()
            print("🔄 Reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done!")


if __name__ == '__main__':
    main()