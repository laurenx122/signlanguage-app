from TTS.api import TTS
import sounddevice as sd
import threading

tts = TTS(
    model_name="tts_models/en/ljspeech/tacotron2-DDC",
    progress_bar=False,
    gpu=False
)

def _play_audio(text):
    audio = tts.tts(text)
    sd.play(audio, samplerate=tts.synthesizer.output_sample_rate)
    sd.wait()

def speak(text):
    if not text:
        return

    # Run speech in background thread
    threading.Thread(
        target=_play_audio,
        args=(text,),
        daemon=True
    ).start()
