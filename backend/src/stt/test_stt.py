import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from stt.stt_manager import STTManager


def main():
    print("=" * 50)
    print("🎙 Whisper VAD Speech-to-Text")
    print("Press ENTER to start listening.")
    print("Recording stops automatically after silence.")
    print("Ctrl+C to exit")
    print("=" * 50)

    try:
        stt = STTManager()

        while True:
            input("\n▶ Press ENTER to start recording...")
            text = stt.listen_and_transcribe()
            print(f"📝 Transcription: {text}")

    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
