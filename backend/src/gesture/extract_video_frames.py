"""
extract_video_frames.py
Extract features for ONLY the 14 specified gestures
Location: D:/SMS/backend/src/gesture/extract_video_frames.py
"""

import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle
from collections import Counter

# Configuration (AUTO-DETECT PROJECT ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # → backend folder

DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'fsl105'
TRAIN_CSV = DATA_DIR / 'train.csv'
LABELS_CSV = DATA_DIR / 'labels.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic_12'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# CRITICAL: Only these 14 gestures will be processed
ALLOWED_GESTURES = [
    'GOOD MORNING',
    'GOOD AFTERNOON',
    'GOOD EVENING',
    'HELLO',
    'HOW ARE YOU',
    'IM FINE',
    'NICE TO MEET YOU',
    'THANK YOU',
    'YOURE WELCOME',
    'SEE YOU TOMORROW',
    'UNDERSTAND',
    'KNOW',
]

# Parameters - will be adjusted based on data analysis
AUGMENT_WITH_MIRRORING = True
MAX_SEQUENCE_LENGTH = None  # Will be determined from data
MIN_FRAMES_WITH_HANDS = 5

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def normalize_landmarks(landmarks):
    """Normalize landmarks (wrist-centered, scale-invariant)"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0].copy()
    landmarks_centered = landmarks - wrist
    distances = np.linalg.norm(landmarks_centered, axis=1)
    hand_size = np.max(distances)
    if hand_size < 1e-6:
        hand_size = 1.0
    return (landmarks_centered / hand_size).flatten()


def mirror_sequence(sequence):
    """Mirror sequence for augmentation"""
    mirrored = sequence.copy()
    mirrored[:, 0::3] *= -1  # Flip X coordinates
    return mirrored


def extract_active_sequence(video_path, max_length=None):
    """
    Extract landmarks from video, ONLY keeping frames where hands are detected
    Returns: (sequence, actual_length) or (None, 0) if insufficient hands
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0

    sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        frame_feats = np.zeros(126, dtype=np.float32)
        hand_detected = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_detected = True
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                
                lm = []
                for landmark in hand_landmarks.landmark:
                    lm.extend([landmark.x, landmark.y, landmark.z])
                
                norm_lm = normalize_landmarks(lm)
                
                if label == 'Left':
                    frame_feats[0:63] = norm_lm
                else:
                    frame_feats[63:126] = norm_lm

        if hand_detected:
            sequence.append(frame_feats)
        
        # Stop if we have enough frames (if max_length specified)
        if max_length and len(sequence) >= max_length:
            break

    cap.release()

    actual_length = len(sequence)
    
    # Discard videos with insufficient hand movement
    if actual_length < MIN_FRAMES_WITH_HANDS:
        return None, 0

    return np.array(sequence, dtype=np.float32), actual_length


def analyze_sequence_lengths(train_df):
    """
    STEP 1: Analyze actual sequence lengths in the dataset
    This helps determine optimal MAX_SEQUENCE_LENGTH
    """
    print("\n" + "="*70)
    print("STEP 1: ANALYZING SEQUENCE LENGTHS")
    print("="*70)
    
    sequence_lengths = []
    per_class_lengths = {gesture: [] for gesture in ALLOWED_GESTURES}
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Analyzing videos"):
        vid_rel_path = row['vid_path'].replace('\\', '/')
        video_path = DATA_DIR / vid_rel_path
        
        if not video_path.exists():
            continue
        
        _, actual_length = extract_active_sequence(video_path, max_length=200)  # Temp max
        
        if actual_length > 0:
            sequence_lengths.append(actual_length)
            per_class_lengths[row['label']].append(actual_length)
    
    if len(sequence_lengths) == 0:
        print("\n❌ ERROR: No valid sequences found!")
        return None
    
    # Calculate statistics
    sequence_lengths = np.array(sequence_lengths)
    
    print(f"\n📊 SEQUENCE LENGTH STATISTICS:")
    print(f"   Total videos analyzed: {len(sequence_lengths)}")
    print(f"   Min length: {sequence_lengths.min()} frames")
    print(f"   Max length: {sequence_lengths.max()} frames")
    print(f"   Mean length: {sequence_lengths.mean():.1f} frames")
    print(f"   Median length: {np.median(sequence_lengths):.1f} frames")
    print(f"   25th percentile: {np.percentile(sequence_lengths, 25):.1f} frames")
    print(f"   75th percentile: {np.percentile(sequence_lengths, 75):.1f} frames")
    print(f"   95th percentile: {np.percentile(sequence_lengths, 95):.1f} frames")
    print(f"   99th percentile: {np.percentile(sequence_lengths, 99):.1f} frames")
    
    # Per-class statistics
    print(f"\n📊 PER-CLASS SEQUENCE LENGTHS:")
    for gesture in ALLOWED_GESTURES:
        if len(per_class_lengths[gesture]) > 0:
            lengths = np.array(per_class_lengths[gesture])
            print(f"   {gesture:20s} - Mean: {lengths.mean():5.1f}, Max: {lengths.max():3.0f}, "
                  f"Samples: {len(lengths):3d}")
        else:
            print(f"   {gesture:20s} - NO DATA FOUND!")
    
    # Recommendation
    p95 = np.percentile(sequence_lengths, 95)
    recommended_max = int(np.ceil(p95 / 10) * 10)  # Round up to nearest 10
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   95% of gestures complete within {p95:.1f} frames")
    print(f"   Recommended MAX_SEQUENCE_LENGTH: {recommended_max}")
    print(f"   This will capture 95% of gestures completely")
    print(f"   {len(sequence_lengths[sequence_lengths > recommended_max])} gestures "
          f"({100 * len(sequence_lengths[sequence_lengths > recommended_max]) / len(sequence_lengths):.1f}%) "
          f"will be truncated")
    
    return recommended_max


def process_filtered_dataset():
    """
    Process ONLY the 14 specified gestures
    """
    global MAX_SEQUENCE_LENGTH
    
    print("="*70)
    print("🎬 FSL DYNAMIC - FILTERED EXTRACTION (14 Gestures Only)")
    print("="*70)
    
    # Load CSVs
    print("\n📂 Loading CSV files...")
    
    if not TRAIN_CSV.exists():
        print(f"❌ Error: train.csv not found at {TRAIN_CSV}")
        return
    
    train_df = pd.read_csv(TRAIN_CSV)
    labels_df = pd.read_csv(LABELS_CSV)
    
    print(f"✅ Loaded train.csv: {len(train_df)} total video entries")
    
    # FILTER for only allowed gestures
    print(f"\n🔍 Filtering for {len(ALLOWED_GESTURES)} allowed gestures...")
    train_df_filtered = train_df[train_df['label'].isin(ALLOWED_GESTURES)].copy()
    
    print(f"✅ Filtered dataset: {len(train_df_filtered)} videos")
    print(f"❌ Excluded: {len(train_df) - len(train_df_filtered)} videos from other gestures")
    
    if len(train_df_filtered) == 0:
        print("\n❌ ERROR: No videos found for the specified gestures!")
        print("Check that your train.csv contains these labels exactly:")
        for gesture in ALLOWED_GESTURES:
            print(f"   - {gesture}")
        return
    
    # Show class distribution
    print(f"\n📊 CLASS DISTRIBUTION (Before Augmentation):")
    class_counts = train_df_filtered['label'].value_counts()
    for gesture in ALLOWED_GESTURES:
        count = class_counts.get(gesture, 0)
        if count > 0:
            print(f"   {gesture:20s} - {count:3d} videos")
        else:
            print(f"   {gesture:20s} - ⚠️  NO VIDEOS FOUND!")
    
    print(f"\n   Min samples: {class_counts.min()}")
    print(f"   Max samples: {class_counts.max()}")
    print(f"   Imbalance ratio: {class_counts.max() / class_counts.min():.2f}:1")
    
    if class_counts.max() / class_counts.min() > 3:
        print(f"\n⚠️  WARNING: High class imbalance detected!")
        print(f"   Consider collecting more data for underrepresented classes")
    
    # STEP 1: Analyze sequence lengths
    recommended_max = analyze_sequence_lengths(train_df_filtered)
    
    if recommended_max is None:
        return
    
    # Ask user or use recommended
    MAX_SEQUENCE_LENGTH = recommended_max
    print(f"\n✅ Using MAX_SEQUENCE_LENGTH = {MAX_SEQUENCE_LENGTH}")
    
    # STEP 2: Extract features with determined max length
    print("\n" + "="*70)
    print("STEP 2: EXTRACTING FEATURES")
    print("="*70)
    
    # Create label mappings (only for our 14 gestures)
    label_to_idx = {name: i for i, name in enumerate(ALLOWED_GESTURES)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}
    num_classes = len(ALLOWED_GESTURES)
    
    print(f"\n📊 Creating model with {num_classes} classes")
    
    all_sequences = []
    all_labels = []
    
    successful = 0
    failed = 0
    skipped_no_hands = 0
    truncated = 0
    
    print(f"\n🎬 Processing {len(train_df_filtered)} videos...")
    
    for idx, row in tqdm(train_df_filtered.iterrows(), total=len(train_df_filtered), 
                         desc="Extracting features"):
        vid_rel_path = row['vid_path'].replace('\\', '/')
        video_path = DATA_DIR / vid_rel_path
        
        if not video_path.exists():
            failed += 1
            if failed <= 5:
                print(f"\n⚠️  Video not found: {video_path}")
            continue
        
        # Extract sequence
        seq, actual_length = extract_active_sequence(video_path, max_length=MAX_SEQUENCE_LENGTH)
        
        if seq is not None:
            label = row['label']
            label_idx = label_to_idx[label]
            
            # Pad to MAX_SEQUENCE_LENGTH
            if len(seq) < MAX_SEQUENCE_LENGTH:
                padding = np.zeros((MAX_SEQUENCE_LENGTH - len(seq), 126), dtype=np.float32)
                seq = np.vstack([seq, padding])
            elif len(seq) > MAX_SEQUENCE_LENGTH:
                seq = seq[:MAX_SEQUENCE_LENGTH]
                truncated += 1
            
            # Add original sequence
            all_sequences.append(seq)
            all_labels.append(label_idx)
            successful += 1
            
            # Add mirrored sequence for augmentation
            if AUGMENT_WITH_MIRRORING:
                all_sequences.append(mirror_sequence(seq))
                all_labels.append(label_idx)
        else:
            skipped_no_hands += 1
    
    # Convert to numpy arrays
    if len(all_sequences) == 0:
        print("\n❌ ERROR: No sequences extracted! Check video paths and files.")
        return
    
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Verify class distribution
    unique_labels_extracted, counts = np.unique(y, return_counts=True)
    
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Total videos processed: {len(train_df_filtered)}")
    print(f"✅ Successfully processed: {successful}")
    print(f"⚠️  Skipped (no hands detected): {skipped_no_hands}")
    print(f"❌ Failed (file not found): {failed}")
    print(f"✂️  Truncated (longer than {MAX_SEQUENCE_LENGTH}): {truncated}")
    
    print(f"\n📊 Final dataset:")
    print(f"   Total sequences (with mirroring): {len(X)}")
    print(f"   Sequences per class (avg): {len(X) / num_classes:.1f}")
    print(f"   Sequence shape: {X.shape}")
    print(f"   Label shape: {y.shape}")
    print(f"   Number of classes: {num_classes}")
    
    # Show final class distribution
    print(f"\n📊 CLASS DISTRIBUTION (After Augmentation):")
    for i in range(len(unique_labels_extracted)):
        label_idx = unique_labels_extracted[i]
        count = counts[i]
        label_name = idx_to_label[label_idx]
        print(f"   {label_name:20s} - {count:4d} sequences")
    
    # Check for class imbalance
    if counts.max() / counts.min() > 2:
        print(f"\n⚠️  WARNING: Class imbalance still present after augmentation!")
        print(f"   Most common class: {counts.max()} samples")
        print(f"   Least common class: {counts.min()} samples")
        print(f"   Ratio: {counts.max() / counts.min():.2f}:1")
        print(f"   Recommendation: Use class-weighted loss during training")
    
    # Save to disk
    print(f"\n💾 Saving to disk...")
    np.save(OUTPUT_DIR / 'sequences_X.npy', X)
    np.save(OUTPUT_DIR / 'labels_y.npy', y)
    
    with open(OUTPUT_DIR / 'label_mapping.pkl', 'wb') as f:
        pickle.dump({
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label
        }, f)
    
    # Save metadata
    metadata = {
        'total_videos': len(train_df_filtered),
        'successful': successful,
        'skipped_no_hands': skipped_no_hands,
        'failed': failed,
        'truncated': truncated,
        'final_sequences': len(X),
        'num_classes': num_classes,
        'augmentation': 'mirroring' if AUGMENT_WITH_MIRRORING else 'none',
        'sequence_length': MAX_SEQUENCE_LENGTH,
        'classes': ALLOWED_GESTURES,
        'class_distribution': {
            idx_to_label[idx]: int(count) 
            for idx, count in zip(unique_labels_extracted, counts)
        }
    }
    
    with open(OUTPUT_DIR / 'extraction_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✅ Files saved:")
    print(f"   📁 {OUTPUT_DIR / 'sequences_X.npy'}")
    print(f"   📁 {OUTPUT_DIR / 'labels_y.npy'}")
    print(f"   📁 {OUTPUT_DIR / 'label_mapping.pkl'}")
    print(f"   📁 {OUTPUT_DIR / 'extraction_metadata.pkl'}")
    
    print(f"\n{'='*70}")
    print(f"🎉 Ready for training! Run: python train_fsl_dynamic_12.py")
    print(f"{'='*70}")


if __name__ == '__main__':
    process_filtered_dataset()