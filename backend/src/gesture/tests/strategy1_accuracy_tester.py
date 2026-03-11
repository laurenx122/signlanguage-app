"""
stt_logger_wrapper.py
======================
Wraps your existing STT engine to automatically log every recognition
event (transcript, latency, WER) via SessionLogger.

Usage (in your STT/speech flow):
    from stt_logger_wrapper import LoggedSTT

    stt = LoggedSTT(your_stt_engine, logger=logger)
    transcript = stt.recognize(audio_chunk, reference="Good morning", environment="quiet")
"""

import time


class LoggedSTT:
    """
    Thin wrapper around any STT engine that auto-logs each recognition.

    Args:
        stt_engine : Your existing STT object (must have a .recognize() or .transcribe() method)
        logger     : SessionLogger instance
        recognize_method : name of the method on stt_engine to call (default "recognize")
    """

    def __init__(self, stt_engine, logger, recognize_method: str = "recognize"):
        self._engine = stt_engine
        self._logger = logger
        self._method = recognize_method

    def recognize(
        self,
        audio,
        reference: str = None,
        environment: str = "quiet",
        notes: str = ""
    ) -> str:
        """
        Call STT engine, measure latency, log result.

        Args:
            audio       : Audio data / file path — passed directly to your engine
            reference   : Ground truth text (for WER). Leave None if unknown.
            environment : "quiet" or "noisy"
            notes       : Extra notes for the log

        Returns:
            transcript (str)
        """
        t_start = time.monotonic()

        # Call the actual STT engine
        fn = getattr(self._engine, self._method)
        transcript = fn(audio)

        stt_latency_ms = (time.monotonic() - t_start) * 1000

        # Log it
        self._logger.log_stt(
            transcript     = transcript,
            stt_latency_ms = stt_latency_ms,
            reference      = reference,
            environment    = environment,
            notes          = notes,
        )

        return transcript


# ──────────────────────────────────────────────────────────────────────────────
"""
strategy1_accuracy_tester.py
=============================
Script for Testing Strategy 1 — Gesture Recognition Accuracy.

This script lets you run gestures one at a time, input the GROUND TRUTH label,
and automatically computes per-session accuracy + logs everything.

How to use:
  1. Run this script alongside your webcam feed.
  2. For each gesture you perform, enter the correct label in the terminal.
  3. The logger captures: prediction, ground truth, is_correct, confidence,
     frames, latency, timestamp.
  4. At the end, a summary JSON is saved with overall accuracy.
"""


def run_strategy1_accuracy_test(
    test_label: str = "Strategy1_Accuracy_Run1",
    num_trials: int = 30,
):
    """
    Interactive accuracy testing session.

    Runs the FSL webcam loop. After each gesture prediction, prompts
    the tester in the terminal to confirm the ground truth label.

    Args:
        test_label  : Label for this test session
        num_trials  : How many gesture predictions to capture before auto-ending
    """
    import sys
    import cv2
    import time
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from tts.tts_engine import CoquiTTS
    from fsl_dynamic_inference import (
        initialize_dynamic_model, update_and_maybe_predict,
        reset_buffer, get_model_info
    )
    from sentence_builder import SentenceBuilder
    from backend.session_logger import SessionLogger

    # Keep original UIResultHold import
    # (assumes test_dynamic_webcam_logged.py is in same dir)
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "webcam_logged",
        os.path.join(os.path.dirname(__file__), "test_dynamic_webcam_logged.py")
    )
    mod = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(mod)
    UIResultHold = mod.UIResultHold
    draw_ui = mod.draw_ui

    print("=" * 70)
    print("🧪 STRATEGY 1: GESTURE RECOGNITION ACCURACY TEST")
    print(f"   Session : {test_label}")
    print(f"   Trials  : {num_trials}")
    print("=" * 70)
    print("   After each gesture prediction appears,")
    print("   TYPE the correct label in the terminal and press ENTER.")
    print("   Type 'skip' to skip a prediction.")
    print("=" * 70)

    initialize_dynamic_model()
    logger = SessionLogger(session_label=test_label, log_dir="logs")
    logger.start_session()

    tts = CoquiTTS()
    sentence_builder = SentenceBuilder(short_pause=0.8, long_pause=2.2)
    ui_hold = UIResultHold(stable_hold=0.9, locked_hold=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        logger.end_session()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_counter, fps_start, fps = 0, time.time(), 0
    _segment_start_time = None
    _was_collecting = False
    hands_detected = False
    current_raw, current_eng = "", ""
    trial_count = 0

    ui_result = {
        'prediction': 'WAITING', 'confidence': 0.0, 'status': 'WAITING',
        'top3_labels': [], 'top3_confs': [], 'should_announce': False
    }

    try:
        while trial_count < num_trials:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            t_before = time.monotonic()
            seg_result = update_and_maybe_predict(frame)
            t_after = time.monotonic()

            dbg = seg_result.get("debug", {})
            collecting = bool(dbg.get("collecting", False))
            frames_collected = int(dbg.get("frames_collected", 0))

            if collecting and not _was_collecting:
                _segment_start_time = time.monotonic()
            _was_collecting = collecting
            hands_detected = collecting

            ui_result = ui_hold.update(seg_result)

            if ui_result.get("status") == "STABLE" and ui_result.get("should_announce"):
                token = ui_result["prediction"]
                conf  = ui_result.get("confidence", 0.0)

                if conf >= 0.40 and token not in ["Too short / ignored", "WAITING", "COLLECTING..."]:
                    # ── Prompt tester for ground truth ──
                    print(f"\n🤟 Prediction #{trial_count + 1}: [{token}] ({conf:.1%})")
                    ground_truth = input("   ✏️  Enter correct label (or 'skip'): ").strip()

                    if ground_truth.lower() == "skip":
                        print("   ⏭️  Skipped.")
                    else:
                        if _segment_start_time is not None:
                            latency_ms = (t_after - _segment_start_time) * 1000
                        else:
                            latency_ms = (t_after - t_before) * 1000

                        logger.log_gesture(
                            predicted_label   = token,
                            confidence        = conf,
                            frames_collected  = frames_collected,
                            inference_time_ms = latency_ms,
                            ground_truth      = ground_truth.upper(),
                        )

                        trial_count += 1
                        print(f"   📝 Logged ({trial_count}/{num_trials})")
                        _segment_start_time = None

                    sentence_builder.add_token(token)
                    current_raw = " ".join(sentence_builder.tokens)
                    current_eng = sentence_builder.expand(current_raw) if current_raw else ""

            finalized = sentence_builder.update_pause(hands_detected)
            if finalized:
                raw_sentence, eng_sentence = finalized
                speak_text = eng_sentence if eng_sentence else raw_sentence
                t_tts = time.monotonic()
                tts.speak_async(speak_text)
                tts_ms = (time.monotonic() - t_tts) * 1000
                logger.log_tts(speak_text, tts_ms)
                current_raw, current_eng = "", ""

            fps_counter += 1
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start)
                fps_counter, fps_start = 0, time.time()

            # Show trial count on frame
            cv2.putText(frame, f"Trial: {trial_count}/{num_trials}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            draw_ui(frame, ui_result, hands_detected, fps, current_raw, current_eng)
            cv2.imshow('Strategy 1 — Accuracy Test', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Exiting early...")
                break
            elif key == ord('r'):
                reset_buffer()
                ui_hold.manual_reset()
                tts.stop()
                sentence_builder.finalize()
                current_raw, current_eng = "", ""
                _segment_start_time = None

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()
        logger.end_session()
        print("\n✅ Strategy 1 test complete!")


if __name__ == "__main__":
    run_strategy1_accuracy_test(
        test_label="Strategy1_Accuracy_Run1",
        num_trials=30,   # ← change to however many you need
    )