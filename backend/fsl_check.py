# Save as 'fsl_check.py'
import sys

print("--- FSL Environment Health Check ---")
try:
    import numpy as np
    import cv2
    import mediapipe as mp
    import torch
    import whisper
    from TTS.api import TTS
    
    print(f"✅ NumPy {np.__version__}: LOADED")
    print(f"✅ OpenCV {cv2.__version__}: LOADED")
    print(f"✅ MediaPipe {mp.__version__}: LOADED")
    print(f"✅ PyTorch {torch.__version__}: LOADED")
    
    # Updated TTS check for version 0.22.0
    print("⏳ Testing TTS engine initialization...")
    
    # 1. Initialize the TTS API
    tts_instance = TTS() 
    
    # 2. Get model names correctly without using len()
    models = tts_instance.list_models()
    # Check if it's a manager object and get names list if so
    if hasattr(models, 'list_tts_models'):
        model_list = models.list_tts_models()
        print(f"✅ Coqui TTS: ALIVE (Found {len(model_list)} TTS models)")
    else:
        print("✅ Coqui TTS: ALIVE (Model manager ready)")
    
    print("\n🚀 CONCLUSION: Environment is STABLE and ready for Pi 5.")

except Exception as e:
    print(f"\n❌ FAILED: {e}")