from gpiozero import Button
from tts.tts_engine import EmergencyAudio
import time
from signal import pause

audio = EmergencyAudio("help_me.mp3")

button = Button(26, pull_up=True)

press_count = 0
last_press_time = 0
RESET_TIMEOUT = 0.8  # Time window to complete the 3 presses

def handle_press():
    global press_count, last_press_time
    current_time = time.time()
    
    # If the gap between presses is too long, restart the counter
    if current_time - last_press_time > RESET_TIMEOUT:
        press_count = 1
    else:
        press_count += 1
        
    last_press_time = current_time
    print(f"Button Pressed! Count: {press_count}/3")
    
    if press_count == 3:
        print("🚨 Triple Press Detected! Playing 'Help me!'...")
        audio.play_help_instant()
        press_count = 0  # Reset for next time

# Attach the function to the button press event
button.when_pressed = handle_press

print("="*40)
print("Pi 5 Emergency System Active")
print("Press button 3x quickly to speak.")
print("="*40)

pause()
