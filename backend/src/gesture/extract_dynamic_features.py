"""
extract_dynamic_features.py
Extracts hand landmarks from FSL dynamic videos and saves as numpy sequences.
Includes visualization and data augmentation for training set.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic_final'
SAVE_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic_sequences'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6  # ADD: Better temporal consistency
)

SEQUENCE_LENGTH = 30

def normalize_landmarks(landmarks_array):
    """Normalizes landmarks relative to the wrist (Landmark 0)"""
    coords = landmarks_array.reshape(21, 3)
    wrist = coords[0]
    centered_coords = coords - wrist
    
    max_val = np.max(np.abs(centered_coords))
    if max_val > 0:
        normalized = centered_coords / max_val
    else:
        normalized = centered_coords
        
    return normalized.flatten()

def extract_and_visualize(video_path, class_name):
    """Extract hand landmarks from video with frame sampling"""
    cap = cv2.VideoCapture(str(video_path))
    sequence = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, total_frames // SEQUENCE_LENGTH)

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames)
        success, frame = cap.read()
        if not success: 
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        # Initialize 126 features (63 Left, 63 Right)
        frame_feats = np.zeros(126, dtype=np.float32)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                
                # Draw for visualization
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract and normalize
                raw_lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                norm_lms = normalize_landmarks(raw_lms)
                
                # Place in correct slot
                if label == 'Left':
                    frame_feats[0:63] = norm_lms
                else:
                    frame_feats[63:126] = norm_lms
        
        sequence.append(frame_feats)
            
        # Visualization
        cv2.putText(frame, f"Class: {class_name} | Frame: {i+1}/{SEQUENCE_LENGTH}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("FSL Feature Extraction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
    cap.release()
    
    # FIX: Ensure exactly SEQUENCE_LENGTH frames
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(np.zeros(126, dtype=np.float32))
            
    return np.array(sequence[:SEQUENCE_LENGTH], dtype=np.float32)

def mirror_sequence(sequence):
    """Augmentation: Horizontal flip with hand slot swapping"""
    mirrored = sequence.copy()
    
    # Extract slots
    left_hand = sequence[:, 0:63].copy()
    right_hand = sequence[:, 63:126].copy()
    
    # Flip X-coordinates (index 0)
    left_hand_reshaped = left_hand.reshape(SEQUENCE_LENGTH, 21, 3)
    left_hand_reshaped[:, :, 0] = -left_hand_reshaped[:, :, 0]
    
    right_hand_reshaped = right_hand.reshape(SEQUENCE_LENGTH, 21, 3)
    right_hand_reshaped[:, :, 0] = -right_hand_reshaped[:, :, 0]
    
    # Swap slots (Left ↔ Right)
    mirrored[:, 0:63] = right_hand_reshaped.reshape(SEQUENCE_LENGTH, 63)
    mirrored[:, 63:126] = left_hand_reshaped.reshape(SEQUENCE_LENGTH, 63)
    
    return mirrored

def process_and_save():
    """Process all splits and save feature sequences"""
    
    print("="*60)
    print("🎬 FSL Dynamic Feature Extraction Pipeline")
    print("="*60)
    
    # Track global statistics
    global_stats = {'train': {}, 'val': {}, 'test': {}}
    
    for split in ['train', 'val', 'test']:
        split_path = DATA_DIR / split
        dest_split_path = SAVE_DIR / split
        
        # Validation
        if not split_path.exists():
            print(f"⚠️  {split.upper()} directory not found, skipping...")
            continue
            
        dest_split_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Processing {split.upper()} set...")
        print("-" * 60)
        
        orig_total, aug_total = 0, 0
        
        for class_folder in sorted(split_path.iterdir()):
            if not class_folder.is_dir(): 
                continue
                
            class_name = class_folder.name
            (dest_split_path / class_name).mkdir(parents=True, exist_ok=True)
            
            # Count videos for this class
            video_files = list(class_folder.glob('*.MOV'))
            
            for video_file in video_files:
                # 1. Extract original sequence
                original_seq = extract_and_visualize(video_file, class_name)
                np.save(dest_split_path / class_name / f"{video_file.stem}_orig.npy", original_seq)
                orig_total += 1
                
                # 2. Augment ONLY for training split
                if split == 'train':
                    mirrored_seq = mirror_sequence(original_seq)
                    np.save(dest_split_path / class_name / f"{video_file.stem}_aug.npy", mirrored_seq)
                    aug_total += 1
            
            print(f"✅ {class_name}: {len(video_files)} videos processed")
        
        # Store stats
        global_stats[split] = {
            'original': orig_total,
            'augmented': aug_total,
            'total': orig_total + aug_total
        }
        
        print("-" * 60)
        print(f"📊 {split.upper()} Summary:")
        print(f"   - Original Sequences:  {orig_total}")
        print(f"   - Augmented Sequences: {aug_total}")
        print(f"   - Total Files Saved:   {orig_total + aug_total}")

    # Final Summary
    print("\n" + "="*60)
    print("📊 FINAL EXTRACTION SUMMARY")
    print("="*60)
    for split in ['train', 'val', 'test']:
        stats = global_stats[split]
        if stats:
            print(f"{split.upper():5} | Orig: {stats['original']:4} | Aug: {stats['augmented']:4} | Total: {stats['total']:4}")
    
    grand_total = sum(s['total'] for s in global_stats.values() if s)
    print("="*60)
    print(f"🎯 Grand Total Sequences: {grand_total}")
    print(f"💾 Saved to: {SAVE_DIR}")
    print("="*60)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_and_save()