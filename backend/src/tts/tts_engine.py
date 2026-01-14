import pyttsx3

engine = pyttsx3.init()

# Optional voice tuning
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def speak(text):
    if not text:
        return
    engine.say(text)
    engine.runAndWait()
