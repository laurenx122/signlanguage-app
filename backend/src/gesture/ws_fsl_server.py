# ws_fsl_server.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.gesture.fsl_static_inference import (
    initialize_fsl_model,
    predict_fsl_static,
)

router = APIRouter()


def _strip_data_url(frame_b64: str) -> str:
    if frame_b64 and frame_b64.lower().startswith("data:") and "," in frame_b64:
        return frame_b64.split(",", 1)[1]
    return frame_b64


@router.websocket("/ws/fsl-simple")
async def fsl_simple_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📱 Mobile connected to /ws/fsl-simple")

    try:
        initialize_fsl_model()
    except Exception as e:
        await websocket.send_json({"error": str(e), "prediction": "ERROR"})
        await websocket.close()
        return

    try:
        while True:
            frame_b64 = await websocket.receive_text()
            frame_b64 = _strip_data_url(frame_b64)

            # ✅ ONLY call predict_fsl_static ONCE per received frame
            result = predict_fsl_static(frame_b64, confidence_threshold=0.65)

            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("🔌 Mobile disconnected from /ws/fsl-simple")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.close()