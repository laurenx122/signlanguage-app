# tts_engine.py
import os
import torch
import threading
import uuid
import time
import glob
from TTS.api import TTS
import pygame  # ← CHANGED FROM playsound

# ------------------ ESPEAK SETUP ------------------
ESPEAK_PATH = r"C:\Program Files\eSpeak NG"

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
os.environ["PATH"] += r";C:\Program Files\eSpeak NG"

# --------------------------------------------------

# Initialize pygame mixer once
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

    def speak_async(self, text):
        if not text.strip():
            return

        def _run():
            with self.lock:
                self.is_speaking = True
                temp_file = None
                try:
                    # Generate unique filename
                    temp_file = f"temp_tts_{uuid.uuid4().hex}.wav"

                    # Generate speech
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

                    # Play with pygame (properly releases file)
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    # Stop and unload the file
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()  # ← KEY: This releases the file
                    
                    # Small delay to ensure file is fully released
                    time.sleep(0.2)

                    # Delete file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"✅ Deleted: {temp_file}")

                except Exception as e:
                    print(f"TTS Error: {e}")
                    # Cleanup on error
                    if temp_file and os.path.exists(temp_file):
                        try:
                            pygame.mixer.music.stop()
                            pygame.mixer.music.unload()
                            time.sleep(0.2)
                            os.remove(temp_file)
                            print(f"🧹 Cleaned up after error: {temp_file}")
                        except Exception as cleanup_err:
                            print(f"⚠️  Could not delete {temp_file}: {cleanup_err}")
                finally:
                    self.is_speaking = False

        threading.Thread(target=_run, daemon=True).start()

    def stop(self):
        """Stop current playback and clean up all temp files"""
        try:
            # Stop current playback
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.2)
            
            # Clean up all temp files
            for file in glob.glob("temp_tts_*.wav"):
                try:
                    os.remove(file)
                    print(f"🧹 Cleaned up: {file}")
                except Exception as e:
                    print(f"⚠️  Could not delete {file}: {e}")
        except Exception as e:
            print(f"Cleanup error: {e}")

#raspberry pi emergency audio class
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
                    # Keep thread alive while music plays
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                else:
                    print(f"❌ Error: File not found at {self.mp3_path}")
            except Exception as e:
                print(f"Audio Error: {e}")

        threading.Thread(target=_run, daemon=True).start()