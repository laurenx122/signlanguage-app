#stt_http.py
from fastapi import APIRouter, UploadFile, File
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os

from src.stt.whisper_engine import WhisperEngine

router = APIRouter()
engine = WhisperEngine(model_size="small.en")

@router.post("/stt")
async def stt(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = engine.transcribe_file(tmp_path) or ""
        return JSONResponse(content={"text": text})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"text": "", "error": str(e)},
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass