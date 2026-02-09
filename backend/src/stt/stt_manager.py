# stt_manager.py
from .whisper_engine import WhisperEngine
from .audio_recorder import AudioRecorder


class STTManager:
    def __init__(self):
        self.engine = WhisperEngine(model_size="small.en")
        self.recorder = AudioRecorder()

    def listen_and_transcribe(self):
        audio_file = self.recorder.record()
        text = self.engine.transcribe_file(audio_file)
        return text
