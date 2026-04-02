#tts_engine.py
import os
import torch
import threading
import uuid
import time
import glob
from TTS.api import TTS
import pygame

# ------------------ ESPEAK SETUP ------------------
ESPEAK_PATH = r"C:\Program Files\eSpeak NG"

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
os.environ["PATH"] += r";C:\Program Files\eSpeak NG"
# --------------------------------------------------

pygame.mixer.init()


class CoquiTTS:
    def __init__(self, model_name="tts_models/en/vctk/vits"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name=model_name).to(self.device)

        if hasattr(self.tts, "speakers") and self.tts.speakers:
            self.speaker = self.tts.speakers[0]
            print(f"🔊 Using speaker: {self.speaker}")
        else:
            self.speaker = None

        self.lock = threading.Lock()
        self.is_speaking = False

    def speak_async(self, text: str):
        if not text or not text.strip():
            return

        def _run():
            with self.lock:
                self.is_speaking = True
                temp_file = None
                try:
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

                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()

                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    time.sleep(0.2)

                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"✅ Deleted: {temp_file}")

                except Exception as e:
                    print(f"TTS Error: {e}")
                    if temp_file and os.path.exists(temp_file):
                        try:
                            pygame.mixer.music.stop()
                            pygame.mixer.music.unload()
                            time.sleep(0.2)
                            os.remove(temp_file)
                            print(f"🧹 Cleaned up after error: {temp_file}")
                        except Exception as cleanup_err:
                            print(f"⚠️ Could not delete {temp_file}: {cleanup_err}")
                finally:
                    self.is_speaking = False

        threading.Thread(target=_run, daemon=True).start()

    def stop(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.2)

            for file in glob.glob("temp_tts_*.wav"):
                try:
                    os.remove(file)
                    print(f"🧹 Cleaned up: {file}")
                except Exception as e:
                    print(f"⚠️ Could not delete {file}: {e}")
        except Exception as e:
            print(f"Cleanup error: {e}")


class EmergencyAudio:
    def __init__(self, mp3_name="help_me.mp3"):
        self.mp3_path = os.path.join("/home/sms/fsl_project", mp3_name)

    def play_help_instant(self):
        def _run():
            try:
                if os.path.exists(self.mp3_path):
                    print(f"🔊 Playing audio: {self.mp3_path}")
                    pygame.mixer.music.load(self.mp3_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                else:
                    print(f"❌ Error: File not found at {self.mp3_path}")
            except Exception as e:
                print(f"Audio Error: {e}")

        threading.Thread(target=_run, daemon=True).start()


# Global Coqui instance
_coqui_engine = CoquiTTS()


def speak(text: str):
    _coqui_engine.speak_async(text)


def stop_speaking():
    _coqui_engine.stop()