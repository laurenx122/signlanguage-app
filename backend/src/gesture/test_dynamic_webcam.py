"""
test_dynamic_webcam.py
=======================
FSL Dynamic Sign Language Recognition — FULL TIMING & LOGGING VERSION

Every time measurement in this file:
─────────────────────────────────────────────────────────────────────
 T1  gesture_collection_start_ms   → when hand first detected (collection begins)
 T2  gesture_collection_end_ms     → when hand disappears (gesture ended)
 T3  gesture_inference_ms          → T3 - T2 : how long model took to predict
 T4  word_display_delay_ms         → T4 - T2 : from gesture end → word shown on screen
 T5  tts_dispatch_ms               → how long speak_async() call itself took
 T6  tts_audio_start_ms            → T6 - sentence_finalized : sentence end → audio begins
 T7  stt_latency_ms                → speech end → text displayed on screen
 T8  sos_response_ms               → button press → "Help me!" audio starts
─────────────────────────────────────────────────────────────────────

All values are printed to terminal AND saved to:
  logs/session_YYYYMMDD_HHMMSS.csv
  logs/session_YYYYMMDD_HHMMSS_summary.json
"""

import sys
import cv2
import time
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from tts.tts_engine import CoquiTTS

from fsl_dynamic_inference import (
    initialize_dynamic_model,
    update_and_maybe_predict,
    reset_buffer,
    get_model_info
)

from sentence_builder import SentenceBuilder
from backend.session_logger import SessionLogger


# ══════════════════════════════════════════════════════════════════════════════
# TimingTracker — tracks all timestamps for one gesture-to-audio pipeline
# ══════════════════════════════════════════════════════════════════════════════
class TimingTracker:
    """
    Holds all timing checkpoints for the current gesture/sentence cycle.
    Resets after each full cycle (gesture → word displayed → TTS played).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # ── Gesture collection ──
        self.t_collection_start   = None   # hand first detected
        self.t_collection_end     = None   # gesture segment complete
        self.frames_collected     = 0

        # ── Model inference ──
        self.t_inference_start    = None
        self.t_inference_end      = None

        # ── Word / text display ──
        self.t_word_display       = None   # when word appears on screen

        # ── Sentence / TTS ──
        self.t_sentence_finalized = None   # long-pause triggered
        self.t_tts_dispatch_start = None   # speak_async() called
        self.t_tts_dispatch_end   = None   # speak_async() returned
        self.t_tts_audio_start    = None   # audio actually begins (if measurable)

        # ── STT ──
        self.t_stt_speech_end     = None   # microphone input ends
        self.t_stt_text_display   = None   # text shown on screen

        # ── SOS ──
        self.t_sos_press          = None
        self.t_sos_audio_start    = None

    # ── Computed delays (all in ms) ──────────────────────────────────────────

    @property
    def collection_duration_ms(self):
        """How long the gesture lasted (collection window)."""
        if self.t_collection_start and self.t_collection_end:
            return (self.t_collection_end - self.t_collection_start) * 1000
        return None

    @property
    def inference_time_ms(self):
        """Pure model inference time (gesture end → prediction ready)."""
        if self.t_inference_start and self.t_inference_end:
            return (self.t_inference_end - self.t_inference_start) * 1000
        return None

    @property
    def word_display_delay_ms(self):
        """Gesture end → word appears on screen (includes inference)."""
        if self.t_collection_end and self.t_word_display:
            return (self.t_word_display - self.t_collection_end) * 1000
        return None

    @property
    def tts_dispatch_ms(self):
        """How long speak_async() call itself takes."""
        if self.t_tts_dispatch_start and self.t_tts_dispatch_end:
            return (self.t_tts_dispatch_end - self.t_tts_dispatch_start) * 1000
        return None

    @property
    def tts_total_latency_ms(self):
        """Sentence finalized → TTS dispatch complete."""
        if self.t_sentence_finalized and self.t_tts_dispatch_end:
            return (self.t_tts_dispatch_end - self.t_sentence_finalized) * 1000
        return None

    @property
    def stt_latency_ms(self):
        """Speech ends → text shown on screen."""
        if self.t_stt_speech_end and self.t_stt_text_display:
            return (self.t_stt_text_display - self.t_stt_speech_end) * 1000
        return None

    @property
    def sos_response_ms(self):
        """Button press → audio output starts."""
        if self.t_sos_press and self.t_sos_audio_start:
            return (self.t_sos_audio_start - self.t_sos_press) * 1000
        return None

    def print_gesture_timing(self, label: str):
        """Print a clean timing breakdown after each gesture."""
        print(f"\n{'─'*55}")
        print(f"  ⏱  TIMING BREAKDOWN — [{label}]")
        print(f"{'─'*55}")
        if self.collection_duration_ms is not None:
            print(f"  T1→T2  Collection duration   : {self.collection_duration_ms:>8.1f} ms  ({self.frames_collected} frames)")
        if self.inference_time_ms is not None:
            print(f"  T2→T3  Model inference        : {self.inference_time_ms:>8.1f} ms")
        if self.word_display_delay_ms is not None:
            print(f"  T2→T4  Word display delay     : {self.word_display_delay_ms:>8.1f} ms")
        print(f"{'─'*55}\n")

    def print_tts_timing(self, text: str):
        """Print TTS timing breakdown."""
        print(f"\n{'─'*55}")
        print(f"  🔊 TTS TIMING — \"{text}\"")
        print(f"{'─'*55}")
        if self.tts_dispatch_ms is not None:
            print(f"  T5     TTS dispatch (call)    : {self.tts_dispatch_ms:>8.1f} ms")
        if self.tts_total_latency_ms is not None:
            print(f"  T4→T6  Sentence→audio start   : {self.tts_total_latency_ms:>8.1f} ms")
        print(f"{'─'*55}\n")

    def print_stt_timing(self, transcript: str):
        """Print STT timing breakdown."""
        print(f"\n{'─'*55}")
        print(f"  🎙️  STT TIMING — \"{transcript}\"")
        print(f"{'─'*55}")
        if self.stt_latency_ms is not None:
            print(f"  T7     Speech end→text display : {self.stt_latency_ms:>8.1f} ms")
        print(f"{'─'*55}\n")

    def print_sos_timing(self):
        """Print SOS timing breakdown."""
        print(f"\n{'─'*55}")
        print(f"  🆘 SOS TIMING")
        print(f"{'─'*55}")
        if self.sos_response_ms is not None:
            print(f"  T8     Button→audio start     : {self.sos_response_ms:>8.1f} ms")
        print(f"{'─'*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# UIResultHold — same logic, now also sets timing checkpoints
# ══════════════════════════════════════════════════════════════════════════════
class UIResultHold:
    def __init__(self, stable_hold=0.8, locked_hold=0.8):
        self.stable_hold = stable_hold
        self.locked_hold = locked_hold
        self.last_pred = None
        self.last_conf = 0.0
        self.last_top3_labels = []
        self.last_top3_confs = []
        self.state = "WAITING"
        self.state_since = time.time()

    def update(self, seg_result: dict) -> dict:
        now = time.time()
        out = {
            "prediction": "WAITING", "confidence": 0.0, "status": "WAITING",
            "top3_labels": [], "top3_confs": [], "should_announce": False
        }

        is_ready         = bool(seg_result.get("is_ready", False))
        top1_label       = seg_result.get("top1_label", "WAITING")
        top1_conf        = float(seg_result.get("top1_conf", 0.0))
        top3_labels      = seg_result.get("top3_labels", [])
        top3_confs       = seg_result.get("top3_confs", [])
        dbg              = seg_result.get("debug", {})
        collecting       = bool(dbg.get("collecting", False))
        frames_collected = int(dbg.get("frames_collected", 0))

        if is_ready:
            self.last_pred = top1_label
            self.last_conf = top1_conf
            self.last_top3_labels = top3_labels
            self.last_top3_confs = top3_confs
            self.state = "STABLE"
            self.state_since = now
            out.update({
                "prediction": self.last_pred, "confidence": self.last_conf,
                "status": "STABLE", "top3_labels": self.last_top3_labels,
                "top3_confs": self.last_top3_confs, "should_announce": True,
                "buffer": f"{min(frames_collected, 30)}/30"
            })
            return out

        if self.state in ("STABLE", "LOCKED") and self.last_pred is not None:
            elapsed = now - self.state_since
            if self.state == "STABLE":
                if elapsed >= self.stable_hold:
                    self.state = "LOCKED"
                    self.state_since = now
                    elapsed = 0.0
                out.update({
                    "prediction": self.last_pred, "confidence": self.last_conf,
                    "status": "STABLE", "top3_labels": self.last_top3_labels,
                    "top3_confs": self.last_top3_confs, "should_announce": False
                })
                return out
            if self.state == "LOCKED":
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
                    "top3_confs": self.last_top3_confs, "should_announce": False
                })
                return out

        if collecting:
            out.update({
                "prediction": "COLLECTING...", "confidence": 0.0,
                "status": "COLLECTING", "buffer": f"{min(frames_collected, 30)}/30",
                "top3_labels": top3_labels, "top3_confs": top3_confs,
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


# ══════════════════════════════════════════════════════════════════════════════
# UI Drawer — now shows live timing info on screen
# ══════════════════════════════════════════════════════════════════════════════
def draw_ui(frame, result, hands_detected, fps,
            current_sentence_raw="", current_sentence_eng="",
            timing: TimingTracker = None):

    h, w, _ = frame.shape
    prediction   = result['prediction']
    confidence   = result['confidence']
    status       = result['status']
    top3_labels  = result.get('top3_labels', [])
    top3_confs   = result.get('top3_confs', [])

    status_colors = {
        'WAITING':    (180, 180, 180),
        'COLLECTING': (0, 165, 255),
        'STABLE':     (0, 255, 0),
        'LOCKED':     (0, 255, 0),
        'COMPLETE':   (100, 200, 100),
        'RESET':      (0, 200, 255)
    }
    color = status_colors.get(status, (255, 255, 255))

    # ── Collection progress bar ──
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

    # ── Main prediction box ──
    box_x, box_y, box_w, box_h = 20, h - 280, 700, 250
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)

    cv2.putText(frame, prediction,
                (box_x + 15, box_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"{confidence:.1%}",
                (box_x + 15, box_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
    cv2.putText(frame, status,
                (box_x + 520, box_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── Top-3 ──
    if top3_labels:
        cv2.putText(frame, "Top-3:", (box_x + 15, box_y + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y_off = 124
        for i, (label, conf) in enumerate(zip(top3_labels[:3], top3_confs[:3])):
            cv2.putText(frame, f"{i+1}. {label} ({conf:.1%})",
                        (box_x + 25, box_y + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            y_off += 18

    # ── Live timing panel (shown on screen) ──
    if timing is not None:
        ty = box_y + 165
        cv2.putText(frame, "── TIMING ──",
                    (box_x + 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 200, 255), 1)
        ty += 16

        def tline(label, value_ms):
            nonlocal ty
            val_str = f"{value_ms:.1f} ms" if value_ms is not None else "---"
            cv2.putText(frame, f"{label}: {val_str}",
                        (box_x + 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 220, 255), 1)
            ty += 15

        # Show live collection timer if collecting
        if timing.t_collection_start and status == "COLLECTING":
            live_ms = (time.monotonic() - timing.t_collection_start) * 1000
            tline("Collecting", live_ms)
        else:
            tline("Collection dur", timing.collection_duration_ms)

        tline("Inference     ", timing.inference_time_ms)
        tline("Word display  ", timing.word_display_delay_ms)
        tline("TTS dispatch  ", timing.tts_dispatch_ms)
        tline("TTS total lat ", timing.tts_total_latency_ms)

    # ── Sentence lines ──
    if current_sentence_raw:
        cv2.putText(frame, f"RAW: {current_sentence_raw}",
                    (box_x + 15, box_y + 232),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)
    if current_sentence_eng:
        cv2.putText(frame, f"ENG: {current_sentence_eng}",
                    (box_x + 15, box_y + 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

    # ── FPS + hand indicator ──
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    cv2.circle(frame, (130, 55), 8, hand_color, -1)

    # ── Top-right: datetime ──
    dt_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, dt_str, (w - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("🎥 FSL DYNAMIC SIGN LANGUAGE RECOGNITION")
    print("   Full Timing + Session Logging Edition")
    print("=" * 70)
    print("""
  Timing checkpoints measured every cycle:
    T1 → T2   Collection duration   (frames in gesture)
    T2 → T3   Model inference time  (gesture end → prediction)
    T2 → T4   Word display delay    (gesture end → word on screen)
    T5        TTS dispatch time     (speak_async call duration)
    T4 → T6   TTS total latency     (sentence ready → audio start)
    T7        STT latency           (speech end → text on screen)
    T8        SOS response time     (button press → audio start)
""")

    initialize_dynamic_model()
    info = get_model_info()
    print(f"📌 Model: {info}")

    print("🔊 Initializing TTS...")
    tts = CoquiTTS()
    print("✅ TTS ready\n")

    sentence_builder = SentenceBuilder(short_pause=0.8, long_pause=2.2)
    ui_hold  = UIResultHold(stable_hold=0.9, locked_hold=0.7)
    timing   = TimingTracker()

    logger = SessionLogger(
        session_label="FullTest_Run1",
        log_dir="logs"
    )
    logger.start_session()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        logger.end_session()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("✅ Webcam ready")
    print("⌨️  Controls: Q=quit  R=reset  S=SOS")
    print("=" * 70)

    fps_counter, fps_start, fps = 0, time.time(), 0
    _was_collecting = False
    hands_detected  = False
    current_raw     = ""
    current_eng     = ""

    ui_result = {
        'prediction': 'WAITING', 'confidence': 0.0, 'status': 'WAITING',
        'top3_labels': [], 'top3_confs': [], 'should_announce': False
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # ────────────────────────────────────────────────────────────────
            # STEP 1: Run inference
            # ────────────────────────────────────────────────────────────────
            timing.t_inference_start = time.monotonic()
            seg_result = update_and_maybe_predict(frame)
            timing.t_inference_end = time.monotonic()

            dbg              = seg_result.get("debug", {})
            collecting       = bool(dbg.get("collecting", False))
            frames_collected = int(dbg.get("frames_collected", 0))

            # ────────────────────────────────────────────────────────────────
            # STEP 2: Collection start / end checkpoints  (T1, T2)
            # ────────────────────────────────────────────────────────────────
            if collecting and not _was_collecting:
                # T1 — hand just appeared, collection started
                timing.t_collection_start = time.monotonic()
                timing.frames_collected   = 0
                print(f"  [T1] Collection started  @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

            if not collecting and _was_collecting:
                # T2 — hand just left, gesture segment complete
                timing.t_collection_end = time.monotonic()
                timing.frames_collected = frames_collected
                print(f"  [T2] Collection ended    @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                      f"  ({frames_collected} frames, {timing.collection_duration_ms:.1f} ms)")

            _was_collecting  = collecting
            hands_detected   = collecting
            timing.frames_collected = frames_collected

            # ────────────────────────────────────────────────────────────────
            # STEP 3: Update UI state
            # ────────────────────────────────────────────────────────────────
            ui_result = ui_hold.update(seg_result)

            # ────────────────────────────────────────────────────────────────
            # STEP 4: New gesture prediction → word display  (T3, T4)
            # ────────────────────────────────────────────────────────────────
            if ui_result.get("status") == "STABLE" and ui_result.get("should_announce"):
                token = ui_result["prediction"]
                conf  = ui_result.get("confidence", 0.0)

                if conf >= 0.40 and token not in ["Too short / ignored", "WAITING", "COLLECTING..."]:

                    # T3 — inference complete (already captured above)
                    infer_ms = timing.inference_time_ms
                    print(f"  [T3] Inference done      @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                          f"  ({infer_ms:.1f} ms)")

                    # T4 — word now appears on screen
                    timing.t_word_display = time.monotonic()
                    print(f"  [T4] Word displayed      @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                          f"  delay={timing.word_display_delay_ms:.1f} ms  word=[{token}]")

                    # Add to sentence
                    sentence_builder.add_token(token)
                    current_raw = " ".join(sentence_builder.tokens)
                    current_eng = sentence_builder.expand(current_raw) if current_raw else ""

                    # Print + log gesture timing
                    timing.print_gesture_timing(token)

                    logger.log_gesture(
                        predicted_label   = token,
                        confidence        = conf,
                        frames_collected  = frames_collected,
                        inference_time_ms = infer_ms or 0.0,
                        ground_truth      = None,    # ← fill in during Strategy 1 testing
                        notes             = (
                            f"collection_ms={timing.collection_duration_ms:.1f}|"
                            f"word_display_delay_ms={timing.word_display_delay_ms:.1f}"
                        )
                    )

            # ────────────────────────────────────────────────────────────────
            # STEP 5: Sentence finalized → TTS  (T5, T6)
            # ────────────────────────────────────────────────────────────────
            finalized = sentence_builder.update_pause(hands_detected)
            if finalized:
                raw_sentence, eng_sentence = finalized
                speak_text = eng_sentence if eng_sentence else raw_sentence

                print(f"\n🧾 RAW: {raw_sentence}")
                print(f"💬 ENG: {eng_sentence}")

                # T_sentence_finalized — sentence ready for TTS
                timing.t_sentence_finalized = time.monotonic()
                print(f"  [T4→] Sentence finalized @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                # T5 — speak_async() dispatch
                timing.t_tts_dispatch_start = time.monotonic()
                tts.speak_async(speak_text)
                timing.t_tts_dispatch_end = time.monotonic()

                # T6 — audio start approximated as dispatch end
                # (speak_async is non-blocking; audio queued immediately after)
                timing.t_tts_audio_start = timing.t_tts_dispatch_end

                print(f"  [T5] TTS dispatch        @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                      f"  dispatch={timing.tts_dispatch_ms:.1f} ms")
                print(f"  [T6] TTS audio start ~   @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                      f"  total_lat={timing.tts_total_latency_ms:.1f} ms")

                timing.print_tts_timing(speak_text)

                logger.log_tts(
                    text           = speak_text,
                    tts_latency_ms = timing.tts_dispatch_ms or 0.0,
                    notes          = f"tts_total_latency_ms={timing.tts_total_latency_ms:.1f}"
                )

                current_raw, current_eng = "", ""

                # Reset timing for next cycle
                timing.reset()

            # ────────────────────────────────────────────────────────────────
            # STEP 6: FPS counter
            # ────────────────────────────────────────────────────────────────
            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter, fps_start = 0, time.time()

            # ────────────────────────────────────────────────────────────────
            # STEP 7: Draw frame with live timing overlay
            # ────────────────────────────────────────────────────────────────
            draw_ui(frame, ui_result, hands_detected, fps,
                    current_raw, current_eng, timing=timing)
            cv2.imshow('FSL Dynamic Sign Recognition', frame)

            # ────────────────────────────────────────────────────────────────
            # Key controls
            # ────────────────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n👋 Exiting...")
                break

            elif key == ord('r'):
                reset_buffer()
                ui_hold.manual_reset()
                tts.stop()
                sentence_builder.finalize()
                timing.reset()
                current_raw, current_eng = "", ""
                _was_collecting = False
                print("\n🔄 Reset!")

            # ── SOS: press S key  (T8)
            elif key == ord('s'):
                timing.t_sos_press = time.monotonic()
                print(f"\n  [T8] SOS button press    @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

                tts.speak_async("Help me!")

                timing.t_sos_audio_start = time.monotonic()
                sos_ms = timing.sos_response_ms

                print(f"  [T8] SOS audio start     @ {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                      f"  response={sos_ms:.1f} ms")
                timing.print_sos_timing()

                current_state = "active" if collecting else "idle"
                logger.log_sos(
                    response_time_ms = sos_ms or 0.0,
                    state            = current_state,
                    success          = True,
                )

            # ── STT: press T key to manually log an STT result  (T7)
            # Replace this block with your actual STT callback in production.
            elif key == ord('t'):
                print("\n  [T7] STT — enter transcript in terminal:")
                # In production: t_stt_speech_end is set when mic input stops.
                # Here we simulate it for testing.
                timing.t_stt_speech_end   = time.monotonic()
                transcript                = input("  Transcript : ").strip()
                reference                 = input("  Reference  : ").strip()
                environment               = input("  Env (quiet/noisy) : ").strip() or "quiet"
                timing.t_stt_text_display = time.monotonic()

                timing.print_stt_timing(transcript)

                logger.log_stt(
                    transcript     = transcript,
                    stt_latency_ms = timing.stt_latency_ms or 0.0,
                    reference      = reference or None,
                    environment    = environment,
                )

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        logger.end_session()
        print("\n✅ Done!")


if __name__ == '__main__':
    main()