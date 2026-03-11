"""
main.py
========
FastAPI application entry point — with session lifecycle management.

Startup  → initializes the shared session logger for the server process
Shutdown → saves final summary JSON

Also adds:
  POST /sos/trigger   — logs SOS button presses from the frontend
  GET  /session/summary — returns live session stats
"""

from contextlib import asynccontextmanager
from datetime import datetime
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.gesture.ws_fsl_server import router as fsl_router
from src.stt.stt_http import router as stt_router
from session_logger import SessionLogger


# ══════════════════════════════════════════════════════════════════════════════
# Global server-level session logger
# Tracks SOS events and any server-wide events
# ══════════════════════════════════════════════════════════════════════════════
_server_logger: SessionLogger | None = None


def get_server_logger() -> SessionLogger:
    return _server_logger


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — startup and shutdown
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _server_logger

    # ── STARTUP ───────────────────────────────────────────────────────────────
    startup_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _server_logger = SessionLogger(
        session_label=f"Server_Session_{startup_ts}",
        log_dir="logs"
    )
    _server_logger.start_session()

    print("="*60)
    print("🚀 FSL Communication System — Server Started")
    print(f"   Session ID : {startup_ts}")
    print(f"   Log dir    : logs/")
    print("="*60)

    yield   # server runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    print("\n🛑 Server shutting down — saving session summary...")
    if _server_logger:
        _server_logger.end_session()
    print("✅ Session saved. Goodbye!")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="FSL Bidirectional Communication System",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fsl_router)
app.include_router(stt_router)


# ══════════════════════════════════════════════════════════════════════════════
# POST /sos/trigger
# Called by the frontend when the SOS button is pressed
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/sos/trigger")
async def sos_trigger(request: Request):
    """
    Log an SOS button press.

    Body (JSON):
    {
      "state":            "idle" | "active",
      "response_time_ms": 85.2,        # measured on the frontend (button→audio)
      "success":          true,
      "client_id":        "mobile_001"
    }
    """
    logger = get_server_logger()

    # ── T8: Server receives SOS event ─────────────────────────────────────────
    t8 = time.monotonic()
    t8_dt = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    state            = body.get("state", "unknown")
    response_time_ms = float(body.get("response_time_ms", 0.0))
    success          = bool(body.get("success", True))
    client_id        = body.get("client_id", "unknown")

    server_receive_ms = (time.monotonic() - t8) * 1000

    print(f"\n  [SOS T8] Received @ {t8_dt}")
    print(f"  {'─'*50}")
    print(f"  🆘 SOS EVENT")
    print(f"     State            : {state}")
    print(f"     Frontend response: {response_time_ms:.1f} ms  (button → audio)")
    print(f"     Server receive   : {server_receive_ms:.1f} ms")
    print(f"     Client           : {client_id}")
    print(f"     Result           : {'✅ PASS' if success else '❌ FAIL'}")
    print(f"  {'─'*50}\n")

    if logger:
        logger.log_sos(
            response_time_ms = response_time_ms,
            state            = state,
            success          = success,
            notes            = f"client_id={client_id}|server_receive_ms={server_receive_ms:.2f}"
        )

    return JSONResponse(content={
        "logged":            True,
        "state":             state,
        "response_time_ms":  response_time_ms,
        "success":           success,
    })


# ══════════════════════════════════════════════════════════════════════════════
# GET /session/summary
# Returns live stats for the current server session
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/session/summary")
async def session_summary():
    """Returns real-time session stats."""
    logger = get_server_logger()
    if not logger:
        return JSONResponse(content={"error": "No active session"})

    g_events = logger._gesture_events
    t_events = logger._tts_latencies
    s_events = logger._stt_events
    o_events = logger._sos_events

    def avg(lst): return round(sum(lst)/len(lst), 2) if lst else 0

    return JSONResponse(content={
        "session_label": logger.session_label,
        "gesture": {
            "total":        len(g_events),
            "avg_conf":     avg([e["confidence"] for e in g_events]),
            "avg_infer_ms": avg([e["inference_ms"] for e in g_events]),
        },
        "tts": {
            "total":        len(t_events),
            "avg_latency":  avg(t_events),
        },
        "stt": {
            "total":        len(s_events),
            "quiet_count":  len([e for e in s_events if e["environment"] == "quiet"]),
            "noisy_count":  len([e for e in s_events if e["environment"] == "noisy"]),
        },
        "sos": {
            "total":        len(o_events),
            "passed":       len([e for e in o_events if e["success"]]),
            "avg_response": avg([e["response_ms"] for e in o_events]),
        },
        "csv_path":     str(logger._csv_path),
        "summary_path": str(logger._summary_path),
    })