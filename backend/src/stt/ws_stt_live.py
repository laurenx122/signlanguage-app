#ws_stt_live.py
import asyncio
import contextlib
import json
import os
import tempfile
import wave

import numpy as np
import subprocess
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

from src.audio.shared_mic import shared_mic

router = APIRouter()

INPUT_SAMPLE_RATE = 48000
OUTPUT_SAMPLE_RATE = 16000

_model = None


def get_model():
    global _model
    if _model is None:
        print("🔊 Loading faster-whisper base.en on cpu")
        _model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return _model


from scipy.signal import resample_poly

def downsample_48k_to_16k(samples: np.ndarray) -> np.ndarray:
    return resample_poly(samples, 1, 3).astype(np.float32)


def save_wav(samples: np.ndarray, path: str):
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(OUTPUT_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def transcribe_samples(samples: np.ndarray) -> str:
    model = get_model()

    mono_16k = downsample_48k_to_16k(samples)

    # 🔥 Direct numpy input (NO FILE)
    segments, _ = model.transcribe(
        mono_16k,
        language="en",
        vad_filter=False,  # 🔥 disable VAD (faster)
        condition_on_previous_text=False,
        beam_size=1,
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


@router.websocket("/ws/stt-live")
async def stt_live_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎤 Client connected to /ws/stt-live")

    try:
        get_model()
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to load STT model: {str(e)}"
        })
        await websocket.close()
        return

    is_listening = False
    recorded_chunks = []
    stop_event = asyncio.Event()

    async def receiver():
        nonlocal is_listening, recorded_chunks

        while not stop_event.is_set():
            raw = await websocket.receive_text()
            data = json.loads(raw)

            action = data.get("action")

            if action == "start":
                print("🎙 STT start received")
                is_listening = True
                recorded_chunks = []
                shared_mic.drain_chunks()

                await websocket.send_json({
                    "type": "status",
                    "message": "Listening started"
                })

            elif action == "stop":
                print("🛑 STT stop received")
                is_listening = False

                audio = (
                    np.concatenate(recorded_chunks)
                    if recorded_chunks
                    else np.array([], dtype=np.float32)
                )

                print(f"🧪 Recorded chunks: {len(recorded_chunks)}")
                print(f"🧪 Audio samples: {len(audio)}")
                if len(audio) > 0:
                    print(f"🧪 Audio max amplitude: {float(np.max(np.abs(audio)))}")

                text = ""
                if len(audio) > 0:
                    try:
                        text = await asyncio.to_thread(transcribe_samples, audio)
                        print(f"🧪 Transcript text: {text}")
                    except Exception as e:
                        print(f"❌ STT transcription error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"STT transcription error: {str(e)}"
                        })
                        continue

                await websocket.send_json({
                    "type": "transcript",
                    "text": text if text else "…",
                })

    async def sender():
        nonlocal is_listening, recorded_chunks

        while not stop_event.is_set():
            level = shared_mic.get_level()
            chunks = shared_mic.drain_chunks()

            if is_listening and chunks:
                recorded_chunks.extend(chunks)

            await websocket.send_json({
                "type": "level",
                "level": level,
                "isRecording": is_listening,
            })

            await asyncio.sleep(0.1)

    receiver_task = asyncio.create_task(receiver())
    sender_task = asyncio.create_task(sender())

    try:
        done, pending = await asyncio.wait(
            [receiver_task, sender_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in done:
            exc = task.exception()
            if exc:
                raise exc

    except WebSocketDisconnect:
        print("🔌 Client disconnected from /ws/stt-live")
    except Exception as e:
        print(f"❌ STT websocket error: {e}")
        with contextlib.suppress(Exception):
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
    finally:
        stop_event.set()
        for task in (receiver_task, sender_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task