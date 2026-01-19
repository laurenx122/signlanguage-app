"""
Preprocessing_FSL.py
Extracts and normalizes hand landmarks from FSL images
- Detects hands using MediaPipe (21 landmarks per hand)
- Converts left-hand landmarks to right-hand format
- Normalizes landmarks (wrist-centered, scale-invariant)
- Saves features and labels as .npy files
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAIN_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_train'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe configuration
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,  # Detect both hands
    min_detection_confidence=0.5
)


def is_left_hand(hand_landmarks, handedness):
    """
    Determine if detected hand is left hand
    MediaPipe's handedness is from the perspective of the person in the image
    """
    if handedness and handedness.classification:
        # MediaPipe returns 'Left' or 'Right' from person's perspective
        label = handedness.classification[0].label
        return label == 'Left'
    return False


def mirror_landmarks_horizontal(landmarks):
    """
    Mirror landmarks horizontally for left hand -> right hand conversion
    Only mirror x-coordinates, keep y and z the same
    """
    mirrored = landmarks.copy()
    # Mirror x-coordinates (0, 3, 6, 9, ... are x values)
    for i in range(0, len(mirrored), 3):
        mirrored[i] = 1.0 - mirrored[i]  # Flip x-coordinate
    return mirrored


def normalize_landmarks(landmarks):
    """
    Normalize landmarks to be position and scale invariant
    - Centers at wrist (landmark 0)
    - Scales by hand size
    - Uses relative depth
    """
    landmarks = np.array(landmarks).reshape(-1, 3)  # Shape: (21, 3)
    
    # Get wrist position (landmark 0)
    wrist = landmarks[0].copy()
    
    # Center at wrist
    landmarks_centered = landmarks - wrist
    
    # Calculate hand size (max distance from wrist)
    distances = np.linalg.norm(landmarks_centered, axis=1)
    hand_size = np.max(distances)
    
    # Avoid division by zero
    if hand_size < 1e-6:
        hand_size = 1.0
    
    # Scale by hand size
    landmarks_normalized = landmarks_centered / hand_size
    
    # Flatten back to 1D array
    return landmarks_normalized.flatten()


def extract_landmarks_from_image(image_path):
    """
    Extract and normalize landmarks from an image
    Returns normalized landmarks (126 features for 2 hands) or None if no hand detected
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    # Initialize array for 2 hands (126 features total)
    all_landmarks = []
    
    # Process up to 2 hands
    for i in range(min(len(results.multi_hand_landmarks), 2)):
        hand_landmarks = results.multi_hand_landmarks[i]
        handedness = results.multi_handedness[i] if results.multi_handedness else None
        
        # Extract landmarks as flat array [x1, y1, z1, x2, y2, z2, ...]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Convert left hand to right hand format
        if is_left_hand(hand_landmarks, handedness):
            landmarks = mirror_landmarks_horizontal(landmarks)
        
        # Normalize landmarks
        landmarks_normalized = normalize_landmarks(landmarks)
        all_landmarks.extend(landmarks_normalized)
    
    # Pad to 126 features if only one hand detected
    while len(all_landmarks) < 126:
        all_landmarks.extend([0.0] * 63)  # Pad with zeros for missing hand
    
    return np.array(all_landmarks[:126], dtype=np.float32)


def preprocess_dataset():
    """
    Process all images in the training directory
    Extract and normalize landmarks, save to .npy files
    """
    print("🔄 Starting FSL dataset preprocessing...")
    print(f"📁 Processing directory: {TRAIN_DIR}")
    
    # Get all class folders
    class_folders = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    
    if not class_folders:
        raise ValueError(f"No class folders found in {TRAIN_DIR}")
    
    print(f"📊 Found {len(class_folders)} classes: {[c.name for c in class_folders]}")
    
    # Create class to index mapping
    class_to_idx = {class_folder.name: idx for idx, class_folder in enumerate(class_folders)}
    
    all_landmarks = []
    all_labels = []
    failed_count = 0
    
    # Process each class
    for class_folder in class_folders:
        class_name = class_folder.name
        class_idx = class_to_idx[class_name]
        
        # Get all images
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        
        print(f"\n📸 Processing class '{class_name}' ({len(image_files)} images)...")
        
        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            landmarks = extract_landmarks_from_image(img_path)
            
            if landmarks is not None:
                all_landmarks.append(landmarks)
                all_labels.append(class_idx)
            else:
                failed_count += 1
    
    # Convert to numpy arrays
    X = np.array(all_landmarks, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Save to files
    landmarks_file = OUTPUT_DIR / 'fsl_landmarks_X.npy'
    labels_file = OUTPUT_DIR / 'fsl_labels_y.npy'
    class_map_file = OUTPUT_DIR / 'fsl_class_to_idx.npy'
    
    np.save(landmarks_file, X)
    np.save(labels_file, y)
    np.save(class_map_file, class_to_idx)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ Preprocessing Complete!")
    print("="*60)
    print(f"Total samples processed: {len(X)}")
    print(f"Failed detections: {failed_count}")
    print(f"Success rate: {len(X)/(len(X)+failed_count)*100:.1f}%")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"\n💾 Saved files:")
    print(f"  - Landmarks: {landmarks_file}")
    print(f"  - Labels: {labels_file}")
    print(f"  - Class mapping: {class_map_file}")
    print("="*60)
    
    hands.close()


if __name__ == '__main__':
    preprocess_dataset()