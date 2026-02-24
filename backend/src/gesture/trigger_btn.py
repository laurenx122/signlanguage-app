import msvcrt
import time
from tts.tts_engine import CoquiTTS

tts = CoquiTTS()
tts.pre_record_help() 

press_count = 0
last_press_time = 0
RESET_TIMEOUT = 0.5  # Seconds allowed between presses

print("System Active. Press '9' THREE TIMES quickly to speak.")
print("Press 'Esc' to stop.")

try:
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            
            if key == b'9':
                current_time = time.time()
                
                # If too much time passed since last press, reset counter to 1
                if current_time - last_press_time > RESET_TIMEOUT:
                    press_count = 1
                else:
                    press_count += 1
                
                last_press_time = current_time
                print(f"Press {press_count}/3")

                # Trigger only on the 3rd press
                if press_count == 3:
                    print("🚨 Triple press detected! Calling for help...")
                    tts.speak_help_instant()
                    press_count = 0  # Reset after speaking
                    
            elif key == b'\x1b':  # ESC
                print("Stopping.")
                break
        
        time.sleep(0.01)

except KeyboardInterrupt:
    pass


# from gpiozero import Button
# from tts.tts_engine import CoquiTTS
# import time
# from signal import pause

# tts = CoquiTTS()
# tts.pre_record_help()

# button = Button(26, pull_up=True)

# press_count = 0
# last_press_time = 0
# RESET_TIMEOUT = 0.5 # Seconds to complete 3 presses

# def handle_press():
#     global press_count, last_press_time
    
#     current_time = time.time()
    
#     if current_time - last_press_time > RESET_TIMEOUT:
#         press_count = 1
#     else:
#         press_count += 1
        
#     last_press_time = current_time
#     print(f"Button Pressed! Count: {press_count}/3")
    
#     if press_count == 3:
#         print("🚨 Emergency Triggered: Speaking 'Help me!'")
#         tts.speak_help_instant()
#         press_count = 0 

# button.when_pressed = handle_press
# pause() 
