"""
test_dynamic_webcam.py
Real-time FSL dynamic sign language recognition with TTS + sentence building

- Uses segment-based inference (collect full gesture, resample to 30, predict once)
- Keeps OLD UI style (WAITING / COLLECTING / STABLE / LOCKED) with green highlight
"""

import sys
import cv2
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tts.tts_engine import CoquiTTS

# Segment-based inference
from fsl_dynamic_inference import (
    initialize_dynamic_model,
    update_and_maybe_predict,
    reset_buffer,
    get_model_info
)

from sentence_builder import SentenceBuilder


# -----------------------------
# UI Hold (to mimic old "green stable" behavior)
# -----------------------------
class UIResultHold:
    """
    Holds the last predicted label on screen in green for a short time
    to mimic the previous stabilizer's STABLE/LOCKED feel.
    """
    def __init__(self, stable_hold=0.8, locked_hold=0.8):
        self.stable_hold = stable_hold
        self.locked_hold = locked_hold

        self.last_pred = None
        self.last_conf = 0.0
        self.last_top3_labels = []
        self.last_top3_confs = []

        self.state = "WAITING"  # WAITING / COLLECTING / STABLE / LOCKED
        self.state_since = time.time()

    def update(self, seg_result: dict) -> dict:
        """
        Convert segment inference output to old UI-style result dict:
          prediction, confidence, status, top3_labels, top3_confs, should_announce
        """
        now = time.time()

        # Default result
        out = {
            "prediction": "WAITING",
            "confidence": 0.0,
            "status": "WAITING",
            "top3_labels": [],
            "top3_confs": [],
            "should_announce": False
        }

        is_ready = bool(seg_result.get("is_ready", False))
        top1_label = seg_result.get("top1_label", "WAITING")
        top1_conf = float(seg_result.get("top1_conf", 0.0))
        top3_labels = seg_result.get("top3_labels", [])
        top3_confs = seg_result.get("top3_confs", [])

        dbg = seg_result.get("debug", {})
        collecting = bool(dbg.get("collecting", False))
        frames_collected = int(dbg.get("frames_collected", 0))

        # If we got a new prediction (gesture ended)
        if is_ready:
            self.last_pred = top1_label
            self.last_conf = top1_conf
            self.last_top3_labels = top3_labels
            self.last_top3_confs = top3_confs

            self.state = "STABLE"
            self.state_since = now

            out.update({
                "prediction": self.last_pred,
                "confidence": self.last_conf,
                "status": "STABLE",
                "top3_labels": self.last_top3_labels,
                "top3_confs": self.last_top3_confs,
                "should_announce": True,   # announce exactly once per gesture prediction
                "buffer": f"{min(frames_collected, 30)}/30"  # optional
            })
            return out

        # If we are currently holding a result
        if self.state in ("STABLE", "LOCKED") and self.last_pred is not None:
            elapsed = now - self.state_since

            if self.state == "STABLE":
                # after stable_hold, go to LOCKED (still green) briefly
                if elapsed >= self.stable_hold:
                    self.state = "LOCKED"
                    self.state_since = now
                    elapsed = 0.0

                out.update({
                    "prediction": self.last_pred,
                    "confidence": self.last_conf,
                    "status": "STABLE",
                    "top3_labels": self.last_top3_labels,
                    "top3_confs": self.last_top3_confs,
                    "should_announce": False
                })
                return out

            if self.state == "LOCKED":
                # after locked_hold, clear back to waiting
                if elapsed >= self.locked_hold:
                    self.last_pred = None
                    self.last_conf = 0.0
                    self.last_top3_labels = []
                    self.last_top3_confs = []
                    self.state = "WAITING"
                    self.state_since = now

                out.update({
                    "prediction": self.last_pred if self.last_pred else "WAITING",
                    "confidence": self.last_conf if self.last_pred else 0.0,
                    "status": "LOCKED" if self.last_pred else "WAITING",
                    "top3_labels": self.last_top3_labels,
                    "top3_confs": self.last_top3_confs,
                    "should_announce": False
                })
                return out

        # Not holding any previous result → show collecting/waiting
        if collecting:
            out.update({
                "prediction": "COLLECTING...",
                "confidence": 0.0,
                "status": "COLLECTING",
                "buffer": f"{min(frames_collected, 30)}/30",
                "top3_labels": top3_labels,
                "top3_confs": top3_confs,
                "should_announce": False
            })
        else:
            out.update({
                "prediction": "WAITING",
                "confidence": 0.0,
                "status": "WAITING",
                "top3_labels": [],
                "top3_confs": [],
                "should_announce": False
            })

        return out

    def manual_reset(self):
        self.last_pred = None
        self.last_conf = 0.0
        self.last_top3_labels = []
        self.last_top3_confs = []
        self.state = "WAITING"
        self.state_since = time.time()


# -----------------------------
# OLD UI Drawer (kept style)
# -----------------------------
def draw_ui(frame, result, hands_detected, fps, current_sentence_raw="", current_sentence_eng=""):
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

    # Progress bar (same look as your old code)
    if 'buffer' in result and status == "COLLECTING":
        try:
            cur, total = map(int, result['buffer'].split('/'))
        except Exception:
            cur, total = 0, 30
        progress = min(cur / max(total, 1), 1.0)
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

    # Sentence display
    if current_sentence_raw:
        cv2.putText(frame, f"RAW: {current_sentence_raw}", (box_x + 15, box_y + 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    if current_sentence_eng:
        cv2.putText(frame, f"ENG: {current_sentence_eng}", (box_x + 15, box_y + 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    cv2.circle(frame, (120, 55), 8, hand_color, -1)


def main():
    print("=" * 70)
    print("🎥 FSL DYNAMIC SIGN LANGUAGE RECOGNITION")
    print("   Segment-based inference + OLD UI (green stable highlight)")
    print("=" * 70)

    initialize_dynamic_model()
    info = get_model_info()
    print(f"📌 Model ready: {info}")

    print("\n🔊 Initializing Coqui TTS...")
    tts = CoquiTTS()
    print("✅ TTS ready")

    # Sentence builder
    sentence_builder = SentenceBuilder(
        short_pause=0.8,
        long_pause=2.2
    )

    # ✅ UI hold to show STABLE/LOCKED like before
    ui_hold = UIResultHold(stable_hold=0.9, locked_hold=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n✅ Webcam ready")
    print("\n⌨️  Controls: Q quit | R reset")
    print("=" * 70)

    fps_counter, fps_start, fps = 0, time.time(), 0

    # Old UI format
    ui_result = {
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

            # Segment inference result (dict)
            seg_result = update_and_maybe_predict(frame)

            # Better hands_detected: when collecting, we know hands were recently present
            dbg = seg_result.get("debug", {})
            hands_detected = bool(dbg.get("collecting", False))

            # Convert to old UI result and hold green stable for a short time
            ui_result = ui_hold.update(seg_result)

            # Add token ONLY when we got a new gesture prediction
            if ui_result.get("status") == "STABLE" and ui_result.get("should_announce"):
                token = ui_result["prediction"]
                conf = ui_result.get("confidence", 0.0)

                if conf >= 0.40 and token not in ["Too short / ignored", "WAITING", "COLLECTING..."]:
                    sentence_builder.add_token(token)
                    current_raw = " ".join(sentence_builder.tokens)
                    current_eng = sentence_builder.expand(current_raw) if current_raw else ""

            # Finalize sentence on long pause
            finalized = sentence_builder.update_pause(hands_detected)
            if finalized:
                raw_sentence, eng_sentence = finalized
                print(f"\n🧾 RAW: {raw_sentence}")
                print(f"💬 ENG: {eng_sentence}")

                speak_text = eng_sentence if eng_sentence else raw_sentence
                tts.speak_async(speak_text)

                current_raw, current_eng = "", ""

            # FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter, fps_start = 0, time.time()

            draw_ui(frame, ui_result, hands_detected, fps, current_raw, current_eng)
            cv2.imshow('FSL Dynamic Sign Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Exiting...")
                break
            elif key == ord('r'):
                reset_buffer()
                ui_hold.manual_reset()
                tts.stop()
                sentence_builder.finalize()
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