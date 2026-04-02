"""
ws_fsl_dynamic_server.py
========================
Dynamic FSL Gesture Recognition — WebSocket Server for Raspberry Pi 5

Pi-ready improvements incorporated from ws_fsl_server.py:
- safer data URL stripping before frame decode
- consistent timing measurement:
    T1 = frame received
    T2 = inference done
    T3 = response sent
- total server latency includes websocket send time
- per-frame inference timing accumulation
- rolling timing printouts
- safer disconnect/error handling
- full session cleanup/reset on disconnect
"""

import sys
import base64
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# ── sys.path fix FIRST — before any local imports ─────────────────────────
_BACKEND_ROOT = Path(__file__).parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ── Local imports AFTER path fix ──────────────────────────────────────────
from src.gesture.sentence_builder import SentenceBuilder
from src.gesture.fsl_dynamic_inference import (
    initialize_dynamic_model,
    update_and_maybe_predict,
    reset_buffer,
    get_model_info,
)
from src.tts.tts_engine import speak
from session_logger import SessionLogger

router = APIRouter()


def _strip_data_url(frame_b64: str) -> str:
    """Strip browser data URL prefix if present."""
    if frame_b64 and frame_b64.lower().startswith("data:") and "," in frame_b64:
        return frame_b64.split(",", 1)[1]
    return frame_b64


def _decode_frame(frame_b64: str):
    """Decode base64 image into OpenCV frame."""
    frame_b64 = _strip_data_url(frame_b64)
    try:
        img_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"❌ Frame decode error: {e}")
        return None


@router.websocket("/ws/fsl-dynamic")
async def fsl_dynamic_endpoint(websocket: WebSocket):
    await websocket.accept()

    client_id = f"{websocket.client.host}:{websocket.client.port}"
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n📱 [DYNAMIC] Client connected: {client_id}")

    # ── Initialize model ───────────────────────────────────────────────────
    try:
        initialize_dynamic_model()
        print(f"📌 Model: {get_model_info()}")
    except Exception as e:
        await websocket.send_json({"error": str(e), "prediction": "ERROR"})
        await websocket.close()
        return

    # ── Start session logger Uncomment Later ───────────────────────────────────────────────
    logger = SessionLogger(
        session_label=f"Dynamic_FSL_{session_ts}_{websocket.client.host}",
        log_dir=str(_BACKEND_ROOT / "logs")
    )
    logger.start_session()

    # ── Per-session state ──────────────────────────────────────────────────
    builder = SentenceBuilder()
    frame_count = 0

    frame_latencies_ms = []      # full server time per frame
    inference_latencies = []     # pure model/update time per frame
    gesture_intervals = []       # time between recognized gestures
    last_gesture_time = None
    last_printed_status = None

    print("\n" + "=" * 55)
    print("  🔍 LIVE STATUS LOG (prints only on change)")
    print("  ⚫ = no hand   🟠 = hand seen   🟡 = collecting")
    print("  ✅ = gesture predicted   💬 = sentence ready")
    print("=" * 55 + "\n")

    try:
        while True:
            # ── T1: frame received ─────────────────────────────────────────
            t1_received = time.monotonic()
            frame_b64 = await websocket.receive_text()
            frame_count += 1

            if frame_count <= 5 or frame_count % 30 == 0:
                print(f"📥 Frame received from {client_id} | frame #{frame_count} | size={len(frame_b64)}")

            frame = _decode_frame(frame_b64)
            if frame is None:
                error_payload = {
                    "is_ready": False,
                    "top1_label": "DECODE_ERROR",
                    "top1_conf": 0.0,
                    "sentence_raw": None,
                    "sentence_english": None,
                    "debug": {
                        "frame_no": frame_count,
                        "hands_detected": False,
                        "collecting": False,
                        "frames_collected": 0,
                        "consec_hand": 0,
                        "consec_nohand": 0,
                        "status": "DECODE_ERROR"
                    }
                }
                await websocket.send_json(error_payload)
                continue

            if frame is not None and (frame_count <= 5 or frame_count % 30 == 0):
                print(f"🖼 Decoded frame #{frame_count} | shape={frame.shape}")

            # ── T2: inference/update ───────────────────────────────────────
            t2_infer_start = time.monotonic()
            result = update_and_maybe_predict(frame)
            t2_infer_end = time.monotonic()

            inference_ms = (t2_infer_end - t2_infer_start) * 1000
            inference_latencies.append(inference_ms)

            dbg = result.get("debug", {})
            collecting = bool(dbg.get("collecting", False))
            frames_in_seg = int(dbg.get("frames_collected", 0))
            consec_hand = int(dbg.get("consec_hand", 0))
            consec_nohand = int(dbg.get("consec_nohand", 0))
            is_ready = bool(result.get("is_ready", False))
            top1_label = result.get("top1_label", "Waiting...")
            hands_now = consec_hand > 0

            if frame_count <= 5 or frame_count % 30 == 0:
                print(
                    f"🧠 Result #{frame_count} | "
                    f"label={result.get('top1_label')} | "
                    f"ready={result.get('is_ready')} | "
                    f"debug={result.get('debug', {})}"
                )

            # ── Status key for terminal logging ────────────────────────────
            if is_ready:
                status_key = f"READY:{top1_label}"
            elif top1_label == "Too short / ignored":
                status_key = "TOO_SHORT"
            elif collecting:
                status_key = f"COLLECTING:{(frames_in_seg // 3) * 3}"
            elif hands_now:
                status_key = f"HAND:{consec_hand}"
            else:
                status_key = f"NOHAND:{(consec_nohand // 3) * 3}"

            if status_key != last_printed_status:
                ts = datetime.now().strftime("%H:%M:%S")
                if is_ready:
                    conf = result.get("top1_conf", 0.0)
                    print(f"\n  ✅ [{ts}] PREDICTED → {top1_label} ({conf:.0%})")
                elif top1_label == "Too short / ignored":
                    print(f"  ❌ [{ts}] TOO SHORT — {frames_in_seg} frames")
                elif collecting:
                    print(f"  🟡 [{ts}] COLLECTING... {frames_in_seg} frames")
                elif hands_now:
                    print(f"  🟠 [{ts}] HAND DETECTED — {consec_hand}/2 needed")
                else:
                    print(f"  ⚫ [{ts}] Waiting... (frame #{frame_count})")
                last_printed_status = status_key

            # ── Build response (sentence fields default None) ──────────────
            enriched = {
                **result,
                "sentence_raw": None,
                "sentence_english": None,
                "debug": {
                    "frame_no": frame_count,
                    "hands_detected": hands_now,
                    "collecting": collecting,
                    "frames_collected": frames_in_seg,
                    "consec_hand": consec_hand,
                    "consec_nohand": consec_nohand,
                    "status": status_key,
                    "inference_ms": round(inference_ms, 1),
                    "frame_size": f"{frame.shape[1]}x{frame.shape[0]}",
                }
            }

            # ── Feed pause into sentence builder every frame ────────────────
            sentence_result = builder.update_pause(hands_now)

            # ── On completed gesture, log + feed token ─────────────────────
            if is_ready:
                label = result.get("top1_label", "UNKNOWN")
                conf = result.get("top1_conf", 0.0)

                now_ts = time.monotonic()
                if last_gesture_time is not None:
                    gesture_intervals.append((now_ts - last_gesture_time) * 1000)
                last_gesture_time = now_ts

                #Uncomment later
                logger.log_gesture(
                    predicted_label=label,
                    confidence=conf,
                    frames_collected=frames_in_seg,
                    inference_time_ms=inference_ms,
                    ground_truth=None,
                    notes=(
                        f"server_frame_pending|"
                        f"frame_no={frame_count}|"
                        f"client={client_id}"
                    )
                )

                # add_token may finalize immediately if max tokens reached
                token_result = builder.add_token(label)
                if token_result:
                    sentence_result = token_result

            # ── Finalize sentence and speak on Pi ───────────────────────────
            if sentence_result:
                raw, english = sentence_result
                enriched["sentence_raw"] = raw
                enriched["sentence_english"] = english

                builder.reset()

                ts = datetime.now().strftime("%H:%M:%S")
                print(f"\n  💬 [{ts}] SENTENCE FINALIZED")
                print(f"       Signs   : {raw}")
                print(f"       English : \"{english}\"")
                print(f"       → Speaking through Pi speaker...\n")

                try:
                    speak(english)
                    #Uncomment Later
                    logger.log_tts(
                        text=english,
                        tts_latency_ms=inference_ms,
                        notes=f"sentence_raw={raw}|frame_no={frame_count}"
                    )
                except Exception as e:
                    print(f"❌ TTS error: {e}")

            # ── T3: send response ───────────────────────────────────────────
            await websocket.send_json(enriched)
            t3_sent = time.monotonic()

            total_server_ms = (t3_sent - t1_received) * 1000
            frame_latencies_ms.append(total_server_ms)

            # add final timing after actual send
            enriched["debug"]["server_total_ms"] = round(total_server_ms, 1)

            # ── Rolling print every 30 frames ───────────────────────────────
            if frame_count % 30 == 0:
                avg_inf = sum(inference_latencies[-30:]) / min(30, len(inference_latencies))
                avg_srv = sum(frame_latencies_ms[-30:]) / min(30, len(frame_latencies_ms))
                print(
                    f"  [FRAME {frame_count:>5}]  "
                    f"inference={avg_inf:.1f}ms (avg30)  "
                    f"server_total={avg_srv:.1f}ms (avg30)  "
                    f"pred={result.get('top1_label', '?')}  "
                    f"ready={result.get('is_ready', False)}"
                )

    except WebSocketDisconnect:
        print(f"\n🔌 Disconnected: {client_id} | frames={frame_count} | gestures={len(gesture_intervals) + (1 if last_gesture_time else 0)}")

    except Exception as e:
        print(f"❌ Error [{client_id}]: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close()
        except Exception:
            pass

    finally:
        # ── Cleanup/reset ──────────────────────────────────────────────────
        reset_buffer()
        builder.reset()
        #Uncomment Later
        logger.end_session() 

        if frame_latencies_ms:
            avg_inf = sum(inference_latencies) / len(inference_latencies) if inference_latencies else 0.0
            avg_srv = sum(frame_latencies_ms) / len(frame_latencies_ms)
            p95_srv = sorted(frame_latencies_ms)[int(len(frame_latencies_ms) * 0.95)]
            print(f"\n  📡 Frame-level Stats ({frame_count} frames total)")
            print(f"     Avg inference      : {avg_inf:.2f} ms")
            print(f"     Avg server total   : {avg_srv:.2f} ms")
            print(f"     P95 server total   : {p95_srv:.2f} ms")
            if gesture_intervals:
                avg_interval = sum(gesture_intervals) / len(gesture_intervals)
                print(f"     Avg gesture interval: {avg_interval:.2f} ms")