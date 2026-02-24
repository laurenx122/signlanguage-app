"""
test_dynamic_webcam.py
Real-time FSL dynamic sign language recognition with TTS + sentence building
"""
import sys
import os
import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque, Counter

sys.path.append(str(Path(__file__).parent.parent))
from tts.tts_engine import CoquiTTS

# Import inference functions
from fsl_dynamic_inference import (
    initialize_dynamic_model,
    add_frame_to_buffer,
    predict_dynamic_sign,
    reset_buffer,
    get_buffer_info
)

# ✅ NEW: sentence builder
from sentence_builder import SentenceBuilder


class GestureStabilizer:
    """
    Stabilizer for dynamic gestures with improved prediction stability
    """
    def __init__(
        self,
        confidence_threshold=0.55,
        stability_window=12,
        min_stable_count=8,
        hold_duration=1.5,
        min_gesture_frames=8,
        max_no_hands_frames=5
    ):
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.min_stable_count = min_stable_count
        self.hold_duration = hold_duration
        self.min_gesture_frames = min_gesture_frames
        self.max_no_hands_frames = max_no_hands_frames

        self.prediction_history = deque(maxlen=stability_window)
        self.confidence_history = deque(maxlen=stability_window)

        self.current_stable = None
        self.stable_since = None
        self.stable_confidence = 0.0

        self.gesture_active = False
        self.gesture_start_time = None
        self.no_hands_counter = 0
        self.total_frames_seen = 0

        self.last_announced = None

    def stabilize(self, prediction, confidence, top3_labels, top3_confs, hands_detected=True):
        current_time = time.time()
        self.total_frames_seen += 1

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
                    'top3_labels': top3_labels,
                    'top3_confs': top3_confs,
                    'should_announce': False
                }
            else:
                if gesture_ended:
                    self._reset_for_new_gesture()
                    return {
                        'prediction': 'WAITING',
                        'confidence': 0.0,
                        'status': 'WAITING',
                        'raw_prediction': prediction,
                        'raw_confidence': confidence,
                        'top3_labels': top3_labels,
                        'top3_confs': top3_confs,
                        'should_announce': False
                    }

                self.current_stable = None
                self.stable_since = None

        # Gesture ended - analyze
        if gesture_ended and self.gesture_active and len(self.prediction_history) >= self.min_gesture_frames:
            result = self._analyze_gesture()

            if result:
                self.current_stable = result['prediction']
                self.stable_since = current_time
                self.stable_confidence = result['confidence']
                self.gesture_active = False

                should_announce = (self.current_stable != self.last_announced)
                if should_announce:
                    self.last_announced = self.current_stable

                return {
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'status': 'STABLE',
                    'votes': result['votes'],
                    'gesture_duration': current_time - self.gesture_start_time if self.gesture_start_time else 0,
                    'raw_prediction': prediction,
                    'raw_confidence': confidence,
                    'top3_labels': top3_labels,
                    'top3_confs': top3_confs,
                    'should_announce': should_announce
                }

        if gesture_ended:
            self.manual_reset()
            return {
                'prediction': 'WAITING',
                'confidence': 0.0,
                'status': 'WAITING',
                'raw_prediction': prediction,
                'raw_confidence': confidence,
                'top3_labels': [],
                'top3_confs': [],
                'should_announce': False
            }

        # Collecting predictions
        if self.gesture_active and hands_detected:
            if confidence >= self.confidence_threshold and prediction not in ["UNKNOWN", "Buffering..."]:
                self.prediction_history.append(prediction)
                self.confidence_history.append(confidence)

            if len(self.prediction_history) < self.min_gesture_frames:
                return {
                    'prediction': 'COLLECTING...',
                    'confidence': confidence,
                    'status': 'COLLECTING',
                    'buffer': f"{len(self.prediction_history)}/{self.min_gesture_frames}",
                    'raw_prediction': prediction,
                    'raw_confidence': confidence,
                    'top3_labels': top3_labels,
                    'top3_confs': top3_confs,
                    'should_announce': False
                }

            vote_counts = Counter(self.prediction_history)
            dominant_label, vote_count = vote_counts.most_common(1)[0]
            dominance = vote_count / len(self.prediction_history)

            if dominance >= 0.6:
                self.current_stable = dominant_label
                self.stable_confidence = float(np.mean(self.confidence_history)) if self.confidence_history else confidence
                self.stable_since = current_time
                self.prediction_history.clear()
                self.confidence_history.clear()

                should_announce = (dominant_label != self.last_announced)
                if should_announce:
                    self.last_announced = dominant_label

                return {
                    'prediction': dominant_label,
                    'confidence': self.stable_confidence,
                    'status': 'STABLE',
                    'raw_prediction': prediction,
                    'raw_confidence': confidence,
                    'top3_labels': top3_labels,
                    'top3_confs': top3_confs,
                    'should_announce': should_announce
                }

            return {
                'prediction': self.current_stable if self.current_stable else dominant_label,
                'confidence': confidence,
                'status': 'COLLECTING',
                'buffer': f"{len(self.prediction_history)}/{self.min_gesture_frames}",
                'raw_prediction': prediction,
                'raw_confidence': confidence,
                'top3_labels': top3_labels,
                'top3_confs': top3_confs,
                'should_announce': False
            }

        return {
            'prediction': 'WAITING',
            'confidence': 0.0,
            'status': 'WAITING',
            'raw_prediction': prediction,
            'raw_confidence': confidence,
            'top3_labels': top3_labels,
            'top3_confs': top3_confs,
            'should_announce': False
        }

    def _analyze_gesture(self):
        if len(self.prediction_history) < self.min_gesture_frames:
            return None

        vote_counts = Counter(self.prediction_history)
        most_common, vote_count = vote_counts.most_common(1)[0]
        consensus_ratio = vote_count / len(self.prediction_history)

        if consensus_ratio >= 0.60:
            avg_confidence = float(np.mean([
                conf for pred, conf in zip(self.prediction_history, self.confidence_history)
                if pred == most_common
            ])) if self.confidence_history else 0.0

            return {'prediction': most_common, 'confidence': avg_confidence, 'votes': f"{vote_count}/{len(self.prediction_history)}"}

        second_half_start = len(self.prediction_history) // 2
        second_half = list(self.prediction_history)[second_half_start:]

        if len(second_half) >= self.min_gesture_frames // 2:
            vote_counts_2nd = Counter(second_half)
            most_common_2nd, vote_count_2nd = vote_counts_2nd.most_common(1)[0]

            if vote_count_2nd / len(second_half) >= 0.55:
                avg_confidence = float(np.mean([
                    conf for pred, conf in zip(
                        list(self.prediction_history)[second_half_start:],
                        list(self.confidence_history)[second_half_start:]
                    ) if pred == most_common_2nd
                ])) if self.confidence_history else 0.0

                return {'prediction': most_common_2nd, 'confidence': avg_confidence, 'votes': f"{vote_count_2nd}/{len(second_half)} (2nd half)"}

        return None

    def _reset_for_new_gesture(self):
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.gesture_active = False
        self.gesture_start_time = None
        self.no_hands_counter = 0
        self.total_frames_seen = 0

    def manual_reset(self):
        self._reset_for_new_gesture()
        self.current_stable = None
        self.stable_since = None
        self.stable_confidence = 0.0
        self.last_announced = None


def draw_ui(frame, result, hands_detected, fps, buffer_info, current_sentence_raw="", current_sentence_eng=""):
    h, w, _ = frame.shape

    prediction = result['prediction']
    confidence = result['confidence']
    status = result['status']
    top3_labels = result.get('top3_labels', [])
    top3_confs = result.get('top3_confs', [])

    status_colors = {
        'WAITING': (180, 180, 180),
        'COLLECTING': (0, 165, 255),
        'STABLE': (0, 255, 0),
        'LOCKED': (0, 255, 0),
        'COMPLETE': (100, 200, 100),
        'RESET': (0, 200, 255)
    }
    color = status_colors.get(status, (255, 255, 255))

    if 'buffer' in result:
        cur, total = map(int, result['buffer'].split('/'))
        progress = min(cur / total, 1.0)
        bar_w = int(w * 0.6 * progress)

        cv2.rectangle(frame, (20, 15), (20 + int(w * 0.6), 35), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 15), (20 + bar_w, 35), (0, 165, 255), -1)
        cv2.putText(frame, f"{cur}/{total} frames", (25, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    box_x, box_y = 20, h - 260
    box_w, box_h = 700, 230

    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)

    cv2.putText(frame, prediction, (box_x + 15, box_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(frame, f"{confidence:.1%}", (box_x + 15, box_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    cv2.putText(frame, status, (box_x + 520, box_y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if len(top3_labels) > 0:
        cv2.putText(frame, "Top-3:", (box_x + 15, box_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        y_offset = 140
        for i, (label, conf) in enumerate(zip(top3_labels[:3], top3_confs[:3])):
            text = f"{i+1}. {label} ({conf:.1%})"
            cv2.putText(frame, text, (box_x + 25, box_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_offset += 20

    # ✅ Sentence display
    if current_sentence_raw:
        cv2.putText(frame, f"RAW: {current_sentence_raw}", (box_x + 15, box_y + 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    if current_sentence_eng:
        cv2.putText(frame, f"ENG: {current_sentence_eng}", (box_x + 15, box_y + 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, f"Buffer: {buffer_info['current']}/{buffer_info['max']}",
                (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    cv2.circle(frame, (120, 55), 8, hand_color, -1)


def main():
    print("="*70)
    print("🎥 FSL DYNAMIC SIGN LANGUAGE RECOGNITION")
    print("   Real-time recognition with Text-to-Speech + Sentence Builder")
    print("="*70)

    initialize_dynamic_model()

    print("\n🔊 Initializing Coqui TTS...")
    tts = CoquiTTS()
    print("✅ TTS ready")

    stabilizer = GestureStabilizer(
        confidence_threshold=0.55,
        stability_window=20,
        min_stable_count=12,
        hold_duration=1.0,
        min_gesture_frames=12,
        max_no_hands_frames=10
    )

    # ✅ NEW: sentence builder (tune pauses here)
    sentence_builder = SentenceBuilder(
        short_pause=0.8,   # between words
        long_pause=2.2     # end of sentence
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n✅ Webcam ready")
    print("\n⌨️  Controls: Q quit | R reset")
    print("="*70)

    fps_counter, fps_start, fps = 0, time.time(), 0

    result = {
        'prediction': 'WAITING',
        'confidence': 0.0,
        'status': 'WAITING',
        'top3_labels': [],
        'top3_confs': [],
        'should_announce': False
    }

    hands_detected = False
    current_raw = ""
    current_eng = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            hands_detected = add_frame_to_buffer(frame)
            buffer_info = get_buffer_info()

            raw_result = predict_dynamic_sign()
            top1_label = raw_result['top1_label']
            top1_conf = raw_result['top1_conf']
            top3_labels = raw_result['top3_labels']
            top3_confs = raw_result['top3_confs']

            result = stabilizer.stabilize(
                top1_label,
                top1_conf,
                top3_labels,
                top3_confs,
                hands_detected=hands_detected
            )

            # ✅ Add tokens when stable
            if result.get('status') == 'STABLE' and result.get('should_announce'):
                token = result['prediction']
                sentence_builder.add_token(token)
                current_raw = " ".join(sentence_builder.tokens)
                current_eng = sentence_builder.expand(current_raw) if current_raw else ""

            # ✅ Finalize sentence on long pause
            finalized = sentence_builder.update_pause(hands_detected)
            if finalized:
                raw_sentence, eng_sentence = finalized
                print(f"\n🧾 RAW: {raw_sentence}")
                print(f"💬 ENG: {eng_sentence}")

                # Speak the expanded sentence (or raw if expansion didn't change)
                speak_text = eng_sentence if eng_sentence else raw_sentence
                tts.speak_async(speak_text)

                current_raw, current_eng = "", ""

            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter, fps_start = 0, time.time()

            draw_ui(frame, result, hands_detected, fps, buffer_info, current_raw, current_eng)
            cv2.imshow('FSL Dynamic Sign Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Exiting...")
                break
            elif key == ord('r'):
                stabilizer.manual_reset()
                reset_buffer()
                tts.stop()
                sentence_builder.finalize()  # clear
                current_raw, current_eng = "", ""
                print("\n🔄 Reset!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        print("\n✅ Done!")


if __name__ == '__main__':
    main()