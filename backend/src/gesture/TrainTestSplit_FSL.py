"""
TrainTestSplit_FSL.py
Randomly samples a fixed number of images per FSL letter to create a balanced dataset
Prevents data leakage and overfitting
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_DIR = PROJECT_ROOT / 'data' / 'raw' / 'fsl_static'  # Your original dataset
TRAIN_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_train'
TEST_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_test'

# Number of images to sample per class
SAMPLES_PER_CLASS_TRAIN = 150  # Adjust based on your dataset size
SAMPLES_PER_CLASS_TEST = 30

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def sample_dataset():
    """Sample images for training and testing sets"""
    
    # Create output directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Source directory: {SOURCE_DIR}")
    print(f"📁 Train directory: {TRAIN_DIR}")
    print(f"📁 Test directory: {TEST_DIR}")
    print(f"🎯 Samples per class - Train: {SAMPLES_PER_CLASS_TRAIN}, Test: {SAMPLES_PER_CLASS_TEST}")
    
    # Get all class folders
    class_folders = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()])
    
    if not class_folders:
        raise ValueError(f"No class folders found in {SOURCE_DIR}")
    
    print(f"\n📊 Found {len(class_folders)} classes")
    
    stats = defaultdict(lambda: {'total': 0, 'train': 0, 'test': 0})
    
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Get all images
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        
        if not image_files:
            print(f"⚠️  No images found in {class_name}")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        total_needed = SAMPLES_PER_CLASS_TRAIN + SAMPLES_PER_CLASS_TEST
        
        if len(image_files) < total_needed:
            print(f"⚠️  {class_name}: Only {len(image_files)} images available (need {total_needed})")
            train_count = int(len(image_files) * 0.8)
            test_count = len(image_files) - train_count
        else:
            train_count = SAMPLES_PER_CLASS_TRAIN
            test_count = SAMPLES_PER_CLASS_TEST
        
        # Split into train and test
        train_images = image_files[:train_count]
        test_images = image_files[train_count:train_count + test_count]
        
        # Create class directories
        train_class_dir = TRAIN_DIR / class_name
        test_class_dir = TEST_DIR / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy train images
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        
        # Copy test images
        for img in test_images:
            shutil.copy2(img, test_class_dir / img.name)
        
        stats[class_name]['total'] = len(image_files)
        stats[class_name]['train'] = len(train_images)
        stats[class_name]['test'] = len(test_images)
        
        print(f"✅ {class_name}: {len(train_images)} train, {len(test_images)} test (from {len(image_files)} total)")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Dataset Split Summary")
    print("="*60)
    total_train = sum(s['train'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    print(f"Total training samples: {total_train}")
    print(f"Total test samples: {total_test}")
    print(f"Total samples: {total_train + total_test}")
    print("="*60)


if __name__ == '__main__':
    sample_dataset()