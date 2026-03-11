"""
stt_http.py
============
Speech-to-Text HTTP endpoint — original structure preserved, logging added.

Changes from original:
  [LOG 1] SessionLogger initialized at module level (one logger per server run)
  [LOG 2] T1 timestamp recorded when file is received
  [LOG 3] T2/T3 timestamps recorded around Whisper transcription
  [LOG 4] log_stt() called with transcript, latency, environment, and WER
"""

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import time
from datetime import datetime

from src.stt.whisper_engine import WhisperEngine   # original import — unchanged
from session_logger import SessionLogger            # [LOG 1]

router = APIRouter()
engine = WhisperEngine(model_size="small.en")      # original — unchanged

# [LOG 1] One shared logger for all /stt requests in this server process.
#         Saves to logs/stt_session_YYYYMMDD_HHMMSS.csv
_stt_logger = SessionLogger(
    session_label=f"STT_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    log_dir="logs"
)
_stt_logger.start_session()


@router.post("/stt")
async def stt(
    file: UploadFile = File(...),
    reference: str   = None,        # optional: ground truth script for WER calculation
    environment: str = "quiet",     # optional: "quiet" or "noisy" — for Strategy 2
):
    print("📥 STT received file:", file.filename, file.content_type)
    ext = os.path.splitext(file.filename)[1] or ".m4a"

    # [LOG 2] T1 — file received, about to start transcription
    t1 = time.monotonic()
    t1_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"  [T1] File received     @ {t1_str}  env={environment}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # [LOG 3] T2 → T3 — wrap Whisper call to measure pure engine time
        t2 = time.monotonic()
        text = engine.transcribe_file(tmp_path) or ""   # original call — unchanged
        t3 = time.monotonic()

        engine_ms = (t3 - t2) * 1000    # pure Whisper transcription time (ms)
        total_ms  = (t3 - t1) * 1000    # file received → transcript ready (ms)

        t3_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"  [T2] Whisper done      @ {t3_str}  engine={engine_ms:.1f}ms")
        print(f"  [T3] Response ready    @ {t3_str}  total={total_ms:.1f}ms")
        print(f"  ── STT Result ──────────────────────────────")
        print(f"     Transcript  : \"{text}\"")
        print(f"     Environment : {environment}")
        print(f"     Engine time : {engine_ms:.1f} ms   ← Whisper only")
        print(f"     Total time  : {total_ms:.1f} ms   ← file received → text ready")
        if reference:
            print(f"     Reference   : \"{reference}\"")
        print(f"  ────────────────────────────────────────────\n")

        # [LOG 4] Log the STT event (WER auto-computed if reference is provided)
        _stt_logger.log_stt(
            transcript     = text,
            stt_latency_ms = total_ms,
            reference      = reference,     # None if not sent → WER skipped
            environment    = environment,
            notes          = f"engine_ms={engine_ms:.2f}|file={file.filename}"
        )

        # Original response + bonus latency fields for the frontend
        return JSONResponse(content={
            "text":       text,             # original field — unchanged
            "latency_ms": round(total_ms, 2),
            "engine_ms":  round(engine_ms, 2),
        })

    finally:
        try:
            os.remove(tmp_path)             # original cleanup — unchanged
        except Exception:
            pass