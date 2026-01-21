# whisper_engine.py
import whisper
import torch

class WhisperEngine:
    def __init__(self, model_size="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔊 Loading Whisper model ({model_size}) on {self.device}")
        self.model = whisper.load_model(model_size).to(self.device)

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path, fp16=torch.cuda.is_available())
        return result["text"].strip()
