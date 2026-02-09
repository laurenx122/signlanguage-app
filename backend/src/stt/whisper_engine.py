# whisper_engine.py
import torch
from typing import Optional
from faster_whisper import WhisperModel


class WhisperEngine:
    def __init__(self, model_size="small.en"):
        self.device = "cpu"  # Change to "cuda" if using GPU
        print(f"🔊 Loading faster-whisper {model_size} on {self.device}")

        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type="int8"  # Fast + efficient
        )

        torch.set_num_threads(1)

    def transcribe_file(self, audio_path: str) -> Optional[str]:
        """Transcribe speech from WAV file."""
        try:
            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                vad_filter=True,
                condition_on_previous_text=False
            )

            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            return text if text else None

        except Exception as e:
            print(f"Transcription error: {e}")
            return None
