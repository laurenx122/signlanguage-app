#ws_fsl_server.py
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from collections import deque
import json

from src.gesture.fsl_static_inference import (
    initialize_fsl_model,
    predict_fsl_static,
    predict_fsl_batch
)

router = APIRouter()

# -------------------------------
# FSL Static WebSocket
# -------------------------------
# src/ws_fsl_server.py - Update the simple endpoint
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

    frame_buffer = deque(maxlen=5)

    try:
        while True:
            frame_b64 = await websocket.receive_text()
            frame_buffer.append(frame_b64)

            if len(frame_buffer) >= 3:
                result = predict_fsl_batch(list(frame_buffer), confidence_threshold=0.6)
            else:
                result = predict_fsl_static(frame_b64, confidence_threshold=0.6)

            # Send full JSON response instead of just prediction text
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("🔌 Mobile disconnected from /ws/fsl-simple")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.close()