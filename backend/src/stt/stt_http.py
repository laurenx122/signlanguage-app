from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os

from src.stt.whisper_engine import WhisperEngine  # adjust path if needed

router = APIRouter()
engine = WhisperEngine(model_size="small.en")

@router.post("/stt")
async def stt(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".m4a"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = engine.transcribe_file(tmp_path) or ""
        # ✅ always JSON
        return JSONResponse(content={"text": text})
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
