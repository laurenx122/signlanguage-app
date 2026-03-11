"""
ws_fsl_server.py
=================
FSL Static Gesture Recognition — WebSocket Server with Full Session Logging.

Every connected mobile/frontend session is automatically tracked:
  - Per-frame predictions (label, confidence, committed letter, latency)
  - TTS trigger events (word spoken, timing)
  - Session summary saved to logs/ on disconnect

Timing measured per frame:
  T1  frame_received_ms     — when frame arrives at server
  T2  inference_ms          — pure model prediction time
  T3  response_sent_ms      — when result is sent back to client
  T_total  round_trip_ms    — T1 → T3 (full server processing time)
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.gesture.fsl_static_inference import (
    initialize_fsl_model,
    predict_fsl_static,
)

_BACKEND_ROOT = Path(__file__).parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from session_logger import SessionLogger

router = APIRouter()


def _strip_data_url(frame_b64: str) -> str:
    if frame_b64 and frame_b64.lower().startswith("data:") and "," in frame_b64:
        return frame_b64.split(",", 1)[1]
    return frame_b64


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket endpoint — /ws/fsl-simple
# ══════════════════════════════════════════════════════════════════════════════
@router.websocket("/ws/fsl-simple")
async def fsl_simple_endpoint(websocket: WebSocket):
    await websocket.accept()

    client_id = f"{websocket.client.host}:{websocket.client.port}"
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n📱 Client connected: {client_id}")

    # ── Initialize model ──────────────────────────────────────────────────────
    try:
        initialize_fsl_model()
    except Exception as e:
        await websocket.send_json({"error": str(e), "prediction": "ERROR"})
        await websocket.close()
        return

    # ── Start session logger for this connection ──────────────────────────────
    logger = SessionLogger(
        session_label=f"WebSocket_Session_{session_ts}_{websocket.client.host}",
        log_dir="logs"
    )
    logger.start_session()

    # ── Per-session accumulators (for timing summary) ─────────────────────────
    frame_count          = 0
    frame_latencies_ms   = []      # server processing time per frame
    inference_latencies  = []      # pure model inference per frame
    last_committed_time  = None    # for tracking inter-word timing
    word_intervals_ms    = []      # time between consecutive committed words

    try:
        while True:

            # ── T1: Frame arrives ─────────────────────────────────────────────
            t1_received = time.monotonic()
            frame_b64   = await websocket.receive_text()
            frame_b64   = _strip_data_url(frame_b64)
            frame_count += 1

            # ── T2: Model inference ───────────────────────────────────────────
            t2_infer_start = time.monotonic()
            result         = predict_fsl_static(frame_b64, confidence_threshold=0.65)
            t2_infer_end   = time.monotonic()

            inference_ms = (t2_infer_end - t2_infer_start) * 1000

            # ── T3: Send result back to client ────────────────────────────────
            await websocket.send_json(result)
            t3_sent = time.monotonic()

            total_server_ms = (t3_sent - t1_received) * 1000

            # Accumulate frame-level timing
            frame_latencies_ms.append(total_server_ms)
            inference_latencies.append(inference_ms)

            # ── Log committed letter (when a letter is actually accepted) ─────
            committed = result.get("committed_letter")
            if committed:
                now_ts = time.monotonic()

                # Time since last committed letter
                if last_committed_time is not None:
                    interval_ms = (now_ts - last_committed_time) * 1000
                    word_intervals_ms.append(interval_ms)

                last_committed_time = now_ts

                logger.log_gesture(
                    predicted_label   = committed,
                    confidence        = result.get("confidence", 0.0),
                    frames_collected  = frame_count,   # cumulative frames this session
                    inference_time_ms = inference_ms,
                    ground_truth      = None,          # set during Strategy 1 testing
                    notes=(
                        f"server_total_ms={total_server_ms:.2f}|"
                        f"frame_no={frame_count}|"
                        f"client={client_id}"
                    )
                )

            # ── Log TTS trigger (word spoken) ─────────────────────────────────
            if result.get("should_speak") and result.get("letters_to_speak"):
                word = "".join(result["letters_to_speak"])
                # TTS was triggered inside predict_fsl_static already.
                # We log the dispatch timing as the inference→response window.
                logger.log_tts(
                    text           = word,
                    tts_latency_ms = total_server_ms,   # server-side processing until TTS triggered
                    notes          = f"letters={''.join(result['letters_to_speak'])}|frame_no={frame_count}"
                )

            # ── Print per-frame timing (every 30 frames to avoid spam) ────────
            if frame_count % 30 == 0:
                avg_inf = sum(inference_latencies[-30:]) / 30
                avg_srv = sum(frame_latencies_ms[-30:]) / 30
                print(
                    f"  [FRAME {frame_count:>5}]  "
                    f"inference={avg_inf:.1f}ms (avg30)  "
                    f"server_total={avg_srv:.1f}ms (avg30)  "
                    f"pred={result.get('prediction','?')}  "
                    f"conf={result.get('confidence', 0):.1%}"
                )

    except WebSocketDisconnect:
        print(f"\n🔌 Client disconnected: {client_id}")

    except Exception as e:
        print(f"❌ WebSocket error [{client_id}]: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

    finally:
        # ── End session — saves CSV + JSON summary ────────────────────────────
        summary = logger.end_session()

        # Print extra timing stats not in the standard summary
        if frame_latencies_ms:
            avg_inf = sum(inference_latencies) / len(inference_latencies)
            avg_srv = sum(frame_latencies_ms) / len(frame_latencies_ms)
            p95_srv = sorted(frame_latencies_ms)[int(len(frame_latencies_ms) * 0.95)]
            print(f"\n  📡 Frame-level Stats ({frame_count} frames total)")
            print(f"     Avg inference      : {avg_inf:.2f} ms")
            print(f"     Avg server total   : {avg_srv:.2f} ms")
            print(f"     P95 server total   : {p95_srv:.2f} ms")
            if word_intervals_ms:
                avg_interval = sum(word_intervals_ms) / len(word_intervals_ms)
                print(f"     Avg letter interval: {avg_interval:.2f} ms")