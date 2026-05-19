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
    min_tracking_confidence=0.6 
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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open: {video_path}")
        return np.zeros((SEQUENCE_LENGTH, 126), dtype=np.float32)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_idxs = np.unique(
        np.linspace(0, max(total_frames - 1, 0), SEQUENCE_LENGTH).astype(int)
    )
    target_set = set(target_idxs.tolist())

    sequence = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break

        if frame_idx in target_set:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            frame_feats = np.zeros(126, dtype=np.float32)

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    raw_lms = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    norm_lms = normalize_landmarks(raw_lms)

                    if label == 'Left':
                        frame_feats[0:63] = norm_lms
                    else:
                        frame_feats[63:126] = norm_lms

            sequence.append(frame_feats)

            cv2.putText(frame, f"Class: {class_name} | Frame: {len(sequence)}/{SEQUENCE_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("FSL Feature Extraction", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(sequence) >= SEQUENCE_LENGTH:
                break

        frame_idx += 1

    cap.release()

    sampled_count = len(sequence)

    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(np.zeros(126, dtype=np.float32))

    if sampled_count == 0:
        print(f"❌ No frames extracted for {video_path.name} (read failed / decode issue)")

    if sampled_count > 0 and sum(np.any(f != 0) for f in sequence[:sampled_count]) == 0:
        print(f"⚠️ {video_path.name}: sampled frames but all features are zeros (no hands detected?)")

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

def add_velocity_features(sequence):
    """
    Append frame-to-frame velocity to each frame's features.
    Input:  (30, 126)
    Output: (30, 252) — position + velocity
    """
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, velocity], axis=1)

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
            # video_files = list(class_folder.glob('*.MOV'))
            video_files = [
                f for f in class_folder.iterdir()
                if f.is_file() and f.suffix.lower() in ('.mov', '.mp4')
            ]

            
            for video_file in video_files:
                # 1. Extract original sequence (30, 126)
                original_seq = extract_and_visualize(video_file, class_name)

                # 2. Mirror BEFORE adding velocity (mirror works on 126 features only)
                if split == 'train':
                    mirrored_seq = mirror_sequence(original_seq)
                    mirrored_seq = add_velocity_features(mirrored_seq)  # (30, 252)
                    np.save(dest_split_path / class_name / f"{video_file.stem}_aug.npy", mirrored_seq)
                    aug_total += 1

                # 3. Add velocity to original and save ONCE
                original_seq = add_velocity_features(original_seq)  # (30, 252)
                np.save(dest_split_path / class_name / f"{video_file.stem}_orig.npy", original_seq)
                orig_total += 1
            
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