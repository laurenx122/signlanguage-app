# src/tts/tts_engine.py
import pyttsx3
import threading

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def _speak(text):
    engine.say(text)
    engine.runAndWait()

def speak(text):
    if not text:
        return

    threading.Thread(
        target=_speak,
        args=(text,),
        daemon=True
    ).start()
