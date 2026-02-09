"""
TrainTestSplit_Dynamic_FSL.py
Randomly samples a fixed number of .MOV videos per FSL dynamic sign.
Destination: data/processed/fsl_dynamic_final
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / 'data' / 'raw' / 'fsl_dynamic' / 'clips'
BASE_DEST = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic_final'
TRAIN_DIR = BASE_DEST / 'train'
VAL_DIR = BASE_DEST / 'val'
TEST_DIR = BASE_DEST / 'test'

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def sample_dynamic_dataset():
    """Sample videos for training, validation, and testing sets"""
    
    # 1. Validation & Setup
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        return

    # Create all directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all class folders
    class_folders = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()])
    if not class_folders:
        print("❌ No class folders found in source directory.")
        return

    print(f"📁 Source directory (Videos): {SOURCE_DIR}")
    print(f"📁 Target Base: {BASE_DEST}")
    
    # 2. Scanning for files and calculating min_samples
    class_video_counts = {}
    total_videos_in_source = 0
    video_extensions = ['*.MOV', '*.mov', '*.mp4', '*.avi']

    for folder in class_folders:
        videos = []
        for ext in video_extensions:
            videos.extend(list(folder.glob(ext)))
        
        # REMOVE DUPLICATES (Windows case-insensitivity fix)
        videos = list(set(videos)) 
        
        count = len(videos)
        class_video_counts[folder.name] = count
        total_videos_in_source += count

    if not class_video_counts or total_videos_in_source == 0:
        print("❌ No videos found!")
        return

    min_samples = min(class_video_counts.values())
    
    # 3. Split Planning (70 / 15 / 15)
    train_count = int(min_samples * 0.7)
    val_count = int(min_samples * 0.15)
    test_count = min_samples - train_count - val_count

    total_to_be_used = min_samples * len(class_folders)  # FIX: Include val_count
    usage_pct = (total_to_be_used / total_videos_in_source) * 100
    
    print(f"\n📊 Found {len(class_folders)} dynamic classes")
    print(f"Total raw videos across all classes: {total_videos_in_source}")
    print(f"🎯 Undersampling Strategy: Using {min_samples} videos per class.")
    print(f"📈 Split Plan: {train_count} Train / {val_count} Val / {test_count} Test per class.")
    print(f"🧪 Total data to be used: {total_to_be_used} / {total_videos_in_source} ({usage_pct:.1f}%)")
    print("-" * 60)
    
    # FIX: Track val in stats
    stats = defaultdict(lambda: {'total': 0, 'train': 0, 'val': 0, 'test': 0})
    
    # 4. Processing Split
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Collect unique files for this specific folder
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(class_folder.glob(ext)))
        
        # CRITICAL: deduplicate here too so the split math is correct
        video_files = list(set(video_files)) 
        
        random.shuffle(video_files)
        balanced_selection = video_files[:min_samples]

        train_vids = balanced_selection[:train_count]
        val_vids = balanced_selection[train_count:train_count + val_count]
        test_vids = balanced_selection[train_count + val_count:]

        # Create directories
        train_class_dir = TRAIN_DIR / class_name
        val_class_dir = VAL_DIR / class_name
        test_class_dir = TEST_DIR / class_name
        
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for vid in train_vids:
            shutil.copy2(vid, train_class_dir / vid.name)
        for vid in val_vids:
            shutil.copy2(vid, val_class_dir / vid.name)
        for vid in test_vids:
            shutil.copy2(vid, test_class_dir / vid.name)
        
        # FIX: Track all splits
        stats[class_name]['total'] = len(video_files)
        stats[class_name]['train'] = len(train_vids)
        stats[class_name]['val'] = len(val_vids)
        stats[class_name]['test'] = len(test_vids)
        
        print(f"✅ {class_name}: {len(train_vids)} train / {len(val_vids)} val / {len(test_vids)} test")

    # 5. Summary
    print("\n" + "="*60)
    print("📊 Final Balanced DYNAMIC Dataset Split Summary")
    print("="*60)
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    
    print(f"Total training videos:   {total_train}")
    print(f"Total validation videos: {total_val}")
    print(f"Total test videos:       {total_test}")
    print(f"Grand Total Used:        {total_train + total_val + total_test}")
    print("="*60)
    
    # 6. Data Distribution Check
    print("\n📊 Split Distribution:")
    print(f"Train: {total_train}/{total_to_be_used} ({100*total_train/total_to_be_used:.1f}%)")
    print(f"Val:   {total_val}/{total_to_be_used} ({100*total_val/total_to_be_used:.1f}%)")
    print(f"Test:  {total_test}/{total_to_be_used} ({100*total_test/total_to_be_used:.1f}%)")

if __name__ == '__main__':
    sample_dynamic_dataset()