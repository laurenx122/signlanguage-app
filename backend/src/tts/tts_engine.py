# tts_engine.py
import os
import torch
import threading
from TTS.api import TTS
import playsound

# ------------------ ESPEAK SETUP ------------------
ESPEAK_PATH = r"C:\Program Files\eSpeak NG"

# FULL PATHS (no guessing)
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
os.environ["PATH"] += r";C:\Program Files\eSpeak NG"

# --------------------------------------------------


class CoquiTTS:
    def __init__(self, model_name="tts_models/en/vctk/vits"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name=model_name).to(self.device)

        # 🔥 FIX: Choose a speaker automatically
        if hasattr(self.tts, "speakers") and self.tts.speakers:
            self.speaker = self.tts.speakers[0]  # first available voice
            print(f"🔊 Using speaker: {self.speaker}")
        else:
            self.speaker = None

        self.lock = threading.Lock()
        self.is_speaking = False

    def speak_async(self, text):
        if not text.strip():
            return

        def _run():
            with self.lock:
                self.is_speaking = True
                try:
                    import uuid
                    import time

                    # 🔥 UNIQUE FILE NAME EVERY TIME
                    temp_file = f"temp_tts_{uuid.uuid4().hex}.wav"

                    if self.speaker:
                        self.tts.tts_to_file(
                            text=text,
                            speaker=self.speaker,
                            file_path=temp_file
                        )
                    else:
                        self.tts.tts_to_file(
                            text=text,
                            file_path=temp_file
                        )

                    # Play sound (blocking until done)
                    playsound.playsound(temp_file)

                    # Ensure file is released
                    time.sleep(0.1)

                    # Delete file
                    try:
                        os.remove(temp_file)
                    except:
                        pass

                except Exception as e:
                    print("TTS Error:", e)
                finally:
                    self.is_speaking = False

        # 🔥 THREAD MUST BE INSIDE FUNCTION
        threading.Thread(target=_run, daemon=True).start()


    def stop(self):
        pass
