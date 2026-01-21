import sounddevice as sd
import numpy as np
import wave
import webrtcvad
import sys

class AudioRecorder:
    def __init__(self, filename="temp.wav", sample_rate=16000):
        self.filename = filename
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode (0-3). 3 is most aggressive at filtering non-speech.

    def record(self):
        print("🎙 Listening... (Start speaking)")
        
        # VAD requires specific frame durations (10, 20, or 30ms)
        frame_duration_ms = 30
        chunk_size = int(self.sample_rate * frame_duration_ms / 1000) # 480 samples for 16kHz
        
        audio_frames = []
        silence_duration = 0
        has_speech_started = False
        
        # Thresholds
        silence_threshold = 1.5  # Seconds of silence to trigger stop
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
                while True:
                    # Read audio chunk
                    data, overflow = stream.read(chunk_size)
                    
                    # Convert to bytes for VAD
                    audio_bytes = data.tobytes()
                    
                    # Check if this chunk contains speech
                    is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)

                    if is_speech:
                        if not has_speech_started:
                            print("🗣 Speech detected! Recording...")
                            has_speech_started = True
                        silence_duration = 0 # Reset silence counter
                    else:
                        if has_speech_started:
                            silence_duration += (frame_duration_ms / 1000)

                    # Store audio if speech has started (or slightly before to catch the start)
                    if has_speech_started:
                        audio_frames.append(audio_bytes)

                    # Stop if silence limit reached
                    if has_speech_started and silence_duration > silence_threshold:
                        print(f"🛑 Silence detected ({silence_threshold}s). Stopping.")
                        break
                        
        except Exception as e:
            print(f"Error recording: {e}")

        # Save to WAV file
        with wave.open(self.filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 2 bytes for 'int16'
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(audio_frames))

        return self.filename