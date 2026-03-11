"""
session_logger.py
=================
Comprehensive testing session logger for FSL Bidirectional Communication System.

Records:
  - Gesture recognition events (prediction, confidence, frames, latency)
  - TTS events (text spoken, timestamp, latency)
  - STT events (transcript, WER score, latency)
  - SOS trigger events
  - Full session summary (accuracy rate, avg latency, frame stats)

Output:
  - logs/session_YYYYMMDD_HHMMSS.csv     (per-event log)
  - logs/session_YYYYMMDD_HHMMSS_summary.json  (summary stats)
"""

import csv
import json
import time
import os
from datetime import datetime
from pathlib import Path
from threading import Lock

# ──────────────────────────────────────────────────────────────────────────────
# Optional: WER calculation (install with: pip install jiwer)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from jiwer import wer as compute_wer
    WER_AVAILABLE = True
except ImportError:
    WER_AVAILABLE = False
    print("⚠️  jiwer not installed. WER calculation disabled. Run: pip install jiwer")


# ──────────────────────────────────────────────────────────────────────────────
# SessionLogger
# ──────────────────────────────────────────────────────────────────────────────
class SessionLogger:
    """
    Drop-in logger for FSL testing sessions.

    Usage:
        logger = SessionLogger(session_label="Test_Run_1")
        logger.start_session()

        # On each gesture prediction:
        logger.log_gesture(
            predicted_label="HELLO",
            confidence=0.97,
            frames_collected=28,
            inference_time_ms=42.3,
            ground_truth="HELLO"   # optional, for accuracy tracking
        )

        # On TTS output:
        logger.log_tts(text="Hello", tts_latency_ms=310.5)

        # On STT result:
        logger.log_stt(
            transcript="Good morning",
            reference="Good morning",   # optional, for WER
            stt_latency_ms=520.0,
            environment="quiet"         # "quiet" or "noisy"
        )

        # On SOS press:
        logger.log_sos(response_time_ms=85.2, state="idle")

        logger.end_session()
    """

    # Event type constants
    EVENT_GESTURE = "GESTURE"
    EVENT_TTS     = "TTS"
    EVENT_STT     = "STT"
    EVENT_SOS     = "SOS"
    EVENT_SESSION = "SESSION"

    CSV_FIELDS = [
        "event_id",
        "event_type",
        "timestamp",
        "datetime",

        # Gesture fields
        "predicted_label",
        "ground_truth",
        "is_correct",
        "confidence",
        "frames_collected",
        "inference_time_ms",

        # TTS fields
        "tts_text",
        "tts_latency_ms",

        # STT fields
        "stt_transcript",
        "stt_reference",
        "stt_wer",
        "stt_latency_ms",
        "stt_environment",

        # SOS fields
        "sos_state",
        "sos_response_time_ms",
        "sos_success",

        # Notes
        "notes",
    ]

    def __init__(self, session_label: str = "", log_dir: str = "logs"):
        self.session_label = session_label or "session"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique session ID from timestamp
        self._session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = self.log_dir / f"session_{self._session_ts}.csv"
        self._summary_path = self.log_dir / f"session_{self._session_ts}_summary.json"

        self._lock = Lock()
        self._event_counter = 0

        # Session-level aggregates
        self._session_start: float = 0.0
        self._session_end: float = 0.0

        # Gesture stats
        self._gesture_events: list = []

        # TTS stats
        self._tts_latencies: list = []

        # STT stats
        self._stt_events: list = []

        # SOS stats
        self._sos_events: list = []

        self._csv_file = None
        self._csv_writer = None

    # ──────────────────────────────────────────────────────────────────────
    # Session lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def start_session(self):
        """Call once at the beginning of a test session."""
        self._session_start = time.monotonic()
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        print(f"\n{'='*60}")
        print(f"📋 SESSION LOGGER STARTED")
        print(f"   Label    : {self.session_label}")
        print(f"   CSV Log  : {self._csv_path}")
        print(f"   Summary  : {self._summary_path}")
        print(f"{'='*60}\n")

        self._write_row({
            "event_type": self.EVENT_SESSION,
            "notes": f"SESSION_START label={self.session_label}"
        })

    def end_session(self):
        """Call once at the end. Saves summary JSON and closes CSV."""
        self._session_end = time.monotonic()
        total_duration = self._session_end - self._session_start

        self._write_row({
            "event_type": self.EVENT_SESSION,
            "notes": f"SESSION_END duration={total_duration:.2f}s"
        })

        if self._csv_file:
            self._csv_file.close()

        summary = self._build_summary(total_duration)
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._print_summary(summary)
        return summary

    # ──────────────────────────────────────────────────────────────────────
    # Logging methods
    # ──────────────────────────────────────────────────────────────────────

    def log_gesture(
        self,
        predicted_label: str,
        confidence: float,
        frames_collected: int,
        inference_time_ms: float,
        ground_truth: str = None,
        notes: str = ""
    ):
        """
        Log a single gesture recognition event.

        Args:
            predicted_label   : What the model predicted (e.g. "HELLO")
            confidence        : Model confidence 0.0–1.0
            frames_collected  : How many frames were in the gesture segment
            inference_time_ms : Time from segment end → prediction ready (ms)
            ground_truth      : The correct label if known (for accuracy tracking)
            notes             : Any extra notes
        """
        is_correct = None
        if ground_truth is not None:
            is_correct = (predicted_label.strip().upper() == ground_truth.strip().upper())

        row = {
            "event_type":       self.EVENT_GESTURE,
            "predicted_label":  predicted_label,
            "ground_truth":     ground_truth or "",
            "is_correct":       "" if is_correct is None else str(is_correct),
            "confidence":       f"{confidence:.4f}",
            "frames_collected": frames_collected,
            "inference_time_ms": f"{inference_time_ms:.2f}",
            "notes":            notes,
        }
        self._write_row(row)

        self._gesture_events.append({
            "predicted":        predicted_label,
            "ground_truth":     ground_truth,
            "is_correct":       is_correct,
            "confidence":       confidence,
            "frames":           frames_collected,
            "inference_ms":     inference_time_ms,
        })

        correct_str = ""
        if is_correct is not None:
            correct_str = " ✅" if is_correct else " ❌"
        print(f"[GESTURE] {predicted_label}{correct_str} | conf={confidence:.1%} | "
              f"frames={frames_collected} | latency={inference_time_ms:.1f}ms")

    def log_tts(
        self,
        text: str,
        tts_latency_ms: float,
        notes: str = ""
    ):
        """
        Log a TTS output event.

        Args:
            text            : Text that was spoken
            tts_latency_ms  : Time from TTS call → audio started (ms)
            notes           : Any extra notes
        """
        row = {
            "event_type":    self.EVENT_TTS,
            "tts_text":      text,
            "tts_latency_ms": f"{tts_latency_ms:.2f}",
            "notes":         notes,
        }
        self._write_row(row)
        self._tts_latencies.append(tts_latency_ms)

        print(f"[TTS] \"{text}\" | latency={tts_latency_ms:.1f}ms")

    def log_stt(
        self,
        transcript: str,
        stt_latency_ms: float,
        reference: str = None,
        environment: str = "quiet",
        notes: str = ""
    ):
        """
        Log a Speech-to-Text result.

        Args:
            transcript      : What the STT engine returned
            stt_latency_ms  : Time from speech end → text displayed (ms)
            reference       : Ground truth script (for WER calculation)
            environment     : "quiet" or "noisy"
            notes           : Any extra notes
        """
        wer_score = None
        if reference and WER_AVAILABLE:
            try:
                wer_score = compute_wer(reference.lower(), transcript.lower())
            except Exception:
                wer_score = None

        row = {
            "event_type":     self.EVENT_STT,
            "stt_transcript": transcript,
            "stt_reference":  reference or "",
            "stt_wer":        f"{wer_score:.4f}" if wer_score is not None else "",
            "stt_latency_ms": f"{stt_latency_ms:.2f}",
            "stt_environment": environment,
            "notes":          notes,
        }
        self._write_row(row)

        self._stt_events.append({
            "transcript":   transcript,
            "reference":    reference,
            "wer":          wer_score,
            "latency_ms":   stt_latency_ms,
            "environment":  environment,
        })

        wer_str = f" | WER={wer_score:.2%}" if wer_score is not None else ""
        print(f"[STT] \"{transcript}\" | env={environment} | latency={stt_latency_ms:.1f}ms{wer_str}")

    def log_sos(
        self,
        response_time_ms: float,
        state: str = "idle",
        success: bool = True,
        notes: str = ""
    ):
        """
        Log an SOS button press event.

        Args:
            response_time_ms : Time from button press → audio output starts (ms)
            state            : "idle" or "active" (was system busy?)
            success          : Did it successfully trigger?
            notes            : Any extra notes
        """
        row = {
            "event_type":         self.EVENT_SOS,
            "sos_state":          state,
            "sos_response_time_ms": f"{response_time_ms:.2f}",
            "sos_success":        str(success),
            "notes":              notes,
        }
        self._write_row(row)

        self._sos_events.append({
            "response_ms": response_time_ms,
            "state":       state,
            "success":     success,
        })

        status = "✅ PASS" if success else "❌ FAIL"
        print(f"[SOS] {status} | state={state} | response={response_time_ms:.1f}ms")

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _write_row(self, data: dict):
        """Write a single event row to the CSV."""
        with self._lock:
            self._event_counter += 1
            now = time.time()
            base = {
                "event_id":  self._event_counter,
                "timestamp": f"{now:.4f}",
                "datetime":  datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            }
            # Fill all missing fields with empty string
            row = {field: "" for field in self.CSV_FIELDS}
            row.update(base)
            row.update(data)

            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def _safe_avg(self, lst):
        return sum(lst) / len(lst) if lst else 0.0

    def _safe_median(self, lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

    def _safe_p95(self, lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    def _build_summary(self, total_duration: float) -> dict:
        """Build the full summary dictionary."""

        # ── Gesture stats ──
        total_gestures = len(self._gesture_events)
        gestures_with_truth = [e for e in self._gesture_events if e["is_correct"] is not None]
        correct_gestures = [e for e in gestures_with_truth if e["is_correct"]]
        accuracy = len(correct_gestures) / len(gestures_with_truth) if gestures_with_truth else None

        inference_times = [e["inference_ms"] for e in self._gesture_events]
        confidences     = [e["confidence"]   for e in self._gesture_events]
        frames_list     = [e["frames"]       for e in self._gesture_events]

        gesture_summary = {
            "total_predictions":    total_gestures,
            "evaluated_with_truth": len(gestures_with_truth),
            "correct":              len(correct_gestures),
            "accuracy_percent":     round(accuracy * 100, 2) if accuracy is not None else "N/A",
            "inference_latency_ms": {
                "mean":   round(self._safe_avg(inference_times), 2),
                "median": round(self._safe_median(inference_times), 2),
                "p95":    round(self._safe_p95(inference_times), 2),
                "min":    round(min(inference_times), 2) if inference_times else 0,
                "max":    round(max(inference_times), 2) if inference_times else 0,
            },
            "confidence": {
                "mean":   round(self._safe_avg(confidences), 4),
                "min":    round(min(confidences), 4) if confidences else 0,
                "max":    round(max(confidences), 4) if confidences else 0,
            },
            "frames_per_gesture": {
                "mean":   round(self._safe_avg(frames_list), 1),
                "min":    min(frames_list) if frames_list else 0,
                "max":    max(frames_list) if frames_list else 0,
            },
        }

        # ── TTS stats ──
        tts_summary = {
            "total_tts_events": len(self._tts_latencies),
            "latency_ms": {
                "mean":   round(self._safe_avg(self._tts_latencies), 2),
                "median": round(self._safe_median(self._tts_latencies), 2),
                "p95":    round(self._safe_p95(self._tts_latencies), 2),
            }
        }

        # ── STT stats ──
        quiet_events = [e for e in self._stt_events if e["environment"] == "quiet"]
        noisy_events = [e for e in self._stt_events if e["environment"] == "noisy"]

        def wer_stats(events):
            wers = [e["wer"] for e in events if e["wer"] is not None]
            lats = [e["latency_ms"] for e in events]
            return {
                "count": len(events),
                "avg_wer":       round(self._safe_avg(wers), 4) if wers else "N/A",
                "avg_latency_ms": round(self._safe_avg(lats), 2) if lats else "N/A",
            }

        stt_summary = {
            "total_stt_events": len(self._stt_events),
            "quiet":  wer_stats(quiet_events),
            "noisy":  wer_stats(noisy_events),
        }

        # ── SOS stats ──
        sos_pass = [e for e in self._sos_events if e["success"]]
        sos_idle   = [e for e in self._sos_events if e["state"] == "idle"]
        sos_active = [e for e in self._sos_events if e["state"] == "active"]

        sos_times = [e["response_ms"] for e in self._sos_events]
        sos_summary = {
            "total_trials":   len(self._sos_events),
            "passed":         len(sos_pass),
            "success_rate_percent": round(len(sos_pass) / len(self._sos_events) * 100, 2) if self._sos_events else "N/A",
            "response_time_ms": {
                "mean":   round(self._safe_avg(sos_times), 2),
                "median": round(self._safe_median(sos_times), 2),
                "p95":    round(self._safe_p95(sos_times), 2),
            },
            "idle_trials":   len(sos_idle),
            "active_trials": len(sos_active),
        }

        return {
            "session_info": {
                "label":          self.session_label,
                "session_id":     self._session_ts,
                "csv_log":        str(self._csv_path),
                "duration_sec":   round(total_duration, 2),
                "started_at":     datetime.fromtimestamp(
                    time.time() - total_duration
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "ended_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "gesture_recognition": gesture_summary,
            "text_to_speech":      tts_summary,
            "speech_to_text":      stt_summary,
            "sos_feature":         sos_summary,
        }

    def _print_summary(self, summary: dict):
        g = summary["gesture_recognition"]
        t = summary["text_to_speech"]
        s = summary["speech_to_text"]
        o = summary["sos_feature"]

        print(f"\n{'='*60}")
        print(f"📊 SESSION SUMMARY — {summary['session_info']['label']}")
        print(f"{'='*60}")
        print(f"  Duration : {summary['session_info']['duration_sec']}s")
        print(f"\n  🤟 Gesture Recognition")
        print(f"     Total predictions : {g['total_predictions']}")
        print(f"     Accuracy          : {g['accuracy_percent']}%")
        print(f"     Avg inference     : {g['inference_latency_ms']['mean']}ms")
        print(f"     P95 inference     : {g['inference_latency_ms']['p95']}ms")
        print(f"     Avg confidence    : {g['confidence']['mean']:.1%}")
        print(f"     Avg frames/gesture: {g['frames_per_gesture']['mean']}")
        print(f"\n  🔊 TTS")
        print(f"     Events     : {t['total_tts_events']}")
        print(f"     Avg latency: {t['latency_ms']['mean']}ms")
        print(f"\n  🎙️  STT")
        print(f"     Events     : {s['total_stt_events']}")
        if s['quiet']['count'] > 0:
            print(f"     Quiet WER  : {s['quiet']['avg_wer']}")
        if s['noisy']['count'] > 0:
            print(f"     Noisy WER  : {s['noisy']['avg_wer']}")
        print(f"\n  🆘 SOS")
        print(f"     Trials       : {o['total_trials']}")
        print(f"     Success rate : {o['success_rate_percent']}%")
        print(f"     Avg response : {o['response_time_ms']['mean']}ms")
        print(f"\n  📁 Saved to:")
        print(f"     {self._csv_path}")
        print(f"     {self._summary_path}")
        print(f"{'='*60}\n")