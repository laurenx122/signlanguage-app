import sounddevice as sd
import numpy as np
import wave
import webrtcvad
from collections import deque


class AudioRecorder:
    def __init__(self, filename="temp.wav", sample_rate=16000):
        self.filename = filename
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(3)

    def record(self):
        print("🎙 Listening... (Start speaking)")

        frame_duration_ms = 30
        chunk_size = int(self.sample_rate * frame_duration_ms / 1000)

        audio_frames = []
        silence_duration = 0
        has_speech_started = False
        silence_threshold = 1.5

        # Buffer to avoid cutting first word
        pre_buffer = deque(maxlen=5)

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
                while True:
                    data, overflow = stream.read(chunk_size)
                    audio_bytes = data.tobytes()

                    is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)

                    pre_buffer.append(audio_bytes)

                    if is_speech:
                        if not has_speech_started:
                            print("🗣 Speech detected! Recording...")
                            has_speech_started = True
                            audio_frames.extend(pre_buffer)
                        silence_duration = 0
                    else:
                        if has_speech_started:
                            silence_duration += (frame_duration_ms / 1000)

                    if has_speech_started:
                        audio_frames.append(audio_bytes)

                    if has_speech_started and silence_duration > silence_threshold:
                        print(f"🛑 Silence detected ({silence_threshold}s). Stopping.")
                        break

        except Exception as e:
            print(f"Error recording: {e}")

        with wave.open(self.filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(audio_frames))

        return self.filename
