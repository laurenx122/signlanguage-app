"""
extract_video_frames.py
Extract landmark sequences from FSL video dataset
For dynamic sign language recognition
"""

import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
VIDEO_DIR = PROJECT_ROOT / 'data' / 'raw' / 'fsl105' / 'clips'
LABELS_CSV = PROJECT_ROOT / 'data' / 'raw' / 'fsl105' / 'labels.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 🎯 TESTING CONFIGURATION
# ============================================
# Set TARGET_CLASSES to test with specific classes only
# None = process all classes
# [0, 1, 2] = process only classes 0, 1, 2
TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6]  # ← Change this! Set to None for all classes

# 🔄 DATA AUGMENTATION
AUGMENT_WITH_MIRRORING = True  # ← Creates both left and right hand versions!
# ============================================

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Sequence parameters
MAX_SEQUENCE_LENGTH = 30  # Maximum frames per video
TARGET_FPS = 10  # Sample every Nth frame to get ~10 FPS


def load_labels():
    """Load labels CSV"""
    df = pd.read_csv(LABELS_CSV)
    label_map = {}
    
    for _, row in df.iterrows():
        folder_id = str(row['id'])
        
        # 🎯 Filter by TARGET_CLASSES if specified
        if TARGET_CLASSES is not None:
            if int(folder_id) not in TARGET_CLASSES:
                continue  # Skip this class
        
        label_map[folder_id] = {
            'label': row['label'],
            'category': row['category']
        }
    
    print(f"📋 Loaded {len(label_map)} categories")
    if TARGET_CLASSES is not None:
        print(f"🎯 Filtering to classes: {TARGET_CLASSES}")
        print(f"   Selected: {[label_map[str(i)]['label'] for i in TARGET_CLASSES if str(i) in label_map]}")
    
    return label_map


def normalize_landmarks(landmarks):
    """Normalize landmarks (wrist-centered, scale-invariant)"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    
    # Center at wrist
    wrist = landmarks[0].copy()
    landmarks_centered = landmarks - wrist
    
    # Scale by hand size
    distances = np.linalg.norm(landmarks_centered, axis=1)
    hand_size = np.max(distances)
    
    if hand_size < 1e-6:
        hand_size = 1.0
    
    landmarks_normalized = landmarks_centered / hand_size
    
    return landmarks_normalized.flatten()


def mirror_sequence_horizontal(sequence):
    """
    Mirror entire sequence horizontally (flip left/right)
    For data augmentation
    """
    mirrored_sequence = []
    
    for frame_landmarks in sequence:
        # Each frame has 126 features (2 hands)
        # Split into two hands
        hand1 = frame_landmarks[:63]
        hand2 = frame_landmarks[63:]
        
        # Mirror each hand's x-coordinates
        mirrored_hand1 = mirror_hand_landmarks(hand1)
        mirrored_hand2 = mirror_hand_landmarks(hand2)
        
        # Concatenate mirrored hands
        mirrored_frame = np.concatenate([mirrored_hand1, mirrored_hand2])
        mirrored_sequence.append(mirrored_frame)
    
    return np.array(mirrored_sequence, dtype=np.float32)


def mirror_hand_landmarks(landmarks):
    """Mirror single hand landmarks (63 features)"""
    if np.all(landmarks == 0):  # Skip if empty (padded) hand
        return landmarks
    
    mirrored = landmarks.copy()
    # Mirror x-coordinates (every 3rd value starting from 0)
    for i in range(0, len(mirrored), 3):
        mirrored[i] = -mirrored[i]  # Flip x in normalized space
    
    return mirrored


def extract_landmarks_sequence(video_path, max_frames=30):
    """Extract landmark sequence from video"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames to get target FPS
    frame_skip = max(1, int(fps / TARGET_FPS))
    
    sequence = []
    frame_count = 0
    
    while len(sequence) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames to reduce FPS
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        frame_count += 1
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Process up to 2 hands
            frame_landmarks = []
            
            for i in range(min(len(results.multi_hand_landmarks), 2)):
                hand_landmarks = results.multi_hand_landmarks[i]
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks = np.array(landmarks, dtype=np.float32)
                
                # Normalize
                landmarks_norm = normalize_landmarks(landmarks)
                frame_landmarks.extend(landmarks_norm)
            
            # Pad to 126 if only one hand
            while len(frame_landmarks) < 126:
                frame_landmarks.extend([0.0] * 63)
            
            sequence.append(frame_landmarks[:126])
    
    cap.release()
    
    if len(sequence) == 0:
        return None
    
    # Pad sequence to max_frames if needed
    while len(sequence) < max_frames:
        sequence.append([0.0] * 126)
    
    return np.array(sequence[:max_frames], dtype=np.float32)


def process_dataset():
    """Process all videos and extract sequences"""
    label_map = load_labels()
    
    all_sequences = []
    all_labels = []
    label_to_idx = {}
    idx_counter = 0
    
    failed_videos = 0
    successful_videos = 0
    
    print("\n🎬 Processing videos...")
    if AUGMENT_WITH_MIRRORING:
        print("🔄 Data augmentation enabled - mirroring all sequences")
    
    for folder_id, info in tqdm(label_map.items(), desc="Categories"):
        label = info['label']
        
        # Create label index if new
        if label not in label_to_idx:
            label_to_idx[label] = idx_counter
            idx_counter += 1
        
        label_idx = label_to_idx[label]
        
        # Find video folder
        video_folder = VIDEO_DIR / folder_id
        
        if not video_folder.exists():
            continue
        
        # Process all videos
        video_files = list(video_folder.glob('*.MOV')) + list(video_folder.glob('*.mp4'))
        
        for video_file in video_files:
            sequence = extract_landmarks_sequence(video_file, MAX_SEQUENCE_LENGTH)
            
            if sequence is not None:
                # Add original sequence
                all_sequences.append(sequence)
                all_labels.append(label_idx)
                successful_videos += 1
                
                # 🔄 Add mirrored version for data augmentation
                if AUGMENT_WITH_MIRRORING:
                    mirrored_sequence = mirror_sequence_horizontal(sequence)
                    all_sequences.append(mirrored_sequence)
                    all_labels.append(label_idx)
            else:
                failed_videos += 1
    
    # Convert to numpy arrays
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Save data
    print("\n💾 Saving processed data...")
    
    np.save(OUTPUT_DIR / 'sequences_X.npy', X)
    np.save(OUTPUT_DIR / 'labels_y.npy', y)
    
    # Save label mapping
    with open(OUTPUT_DIR / 'label_mapping.pkl', 'wb') as f:
        pickle.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': {v: k for k, v in label_to_idx.items()}
        }, f)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ Processing Complete!")
    print("="*60)
    if TARGET_CLASSES is not None:
        print(f"🎯 Processed ONLY classes: {TARGET_CLASSES}")
    if AUGMENT_WITH_MIRRORING:
        print(f"🔄 Data augmentation: ON (dataset doubled with mirroring)")
        print(f"   Original videos: {successful_videos}")
        print(f"   Total sequences (with augmentation): {len(X)}")
    else:
        print(f"Successful videos: {successful_videos}")
        print(f"Total sequences: {len(X)}")
    print(f"Failed videos: {failed_videos}")
    print(f"Sequence shape: {X.shape}")
    print(f"Number of classes: {len(label_to_idx)}")
    print(f"Classes: {list(label_to_idx.keys())}")
    print(f"\nSaved to: {OUTPUT_DIR}")
    if TARGET_CLASSES is not None:
        print(f"\n💡 To process all classes, set TARGET_CLASSES = None")
    if not AUGMENT_WITH_MIRRORING:
        print(f"💡 To double dataset with mirroring, set AUGMENT_WITH_MIRRORING = True")
    print("="*60)
    
    hands.close()


if __name__ == '__main__':
    process_dataset()