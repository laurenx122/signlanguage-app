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
    # print(f"🎯 Samples per class - Train: {SAMPLES_PER_CLASS_TRAIN}, Test: {SAMPLES_PER_CLASS_TEST}")
    
    # Get all class folders
    class_folders = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()])
    
    if not class_folders:
        raise ValueError(f"No class folders found in {SOURCE_DIR}")
    class_image_counts = {}
    total_images_in_source = 0
    print(f"\n📊 ORIGINAL: Found {len(class_folders)} classes")

    for folder in class_folders:
        images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
        count = len(images)
        class_image_counts[folder.name] = count
        total_images_in_source += count

    # Determine the "Smallest Class" logic
    min_samples = min(class_image_counts.values())
    max_samples = max(class_image_counts.values())

    print(f"\n📊 Found {len(class_folders)} classes")
    print(f"Total raw images across all classes: {total_images_in_source}")

    if min_samples == max_samples:
        print(f"⚖️  Everything is perfectly equal! Every class has {min_samples} images.")
    else:
        smallest_class = min(class_image_counts, key=class_image_counts.get)
        print(f"⚠️  Data is unequal. Smallest class is '{smallest_class}' with {min_samples} images.")

    # Calculate splits based on the minority class
    train_count = int(min_samples * 0.8)
    test_count = min_samples - train_count
    total_to_be_used = (train_count + test_count) * len(class_folders)

    print(f"🎯 Undersampling Strategy: Using {min_samples} images per class.")
    print(f"📈 Split Plan: {train_count} Train / {test_count} Test per class.")
    print(f"🧪 Total data to be used: {total_to_be_used} / {total_images_in_source} ({(total_to_be_used/total_images_in_source)*100:.1f}%)")
    print("-" * 60)
    
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
        
        # Slice the list to the minority class size
        balanced_selection = image_files[:min_samples]

        # total_needed = SAMPLES_PER_CLASS_TRAIN + SAMPLES_PER_CLASS_TEST

        # if len(image_files) < total_needed:
        #     print(f"⚠️  {class_name}: Only {len(image_files)} images available (need {total_needed})")
        #     train_count = int(len(image_files) * 0.8)
        #     test_count = len(image_files) - train_count
        # else:
        #     train_count = SAMPLES_PER_CLASS_TRAIN
        #     test_count = SAMPLES_PER_CLASS_TEST
        
        # # Split into train and test
        # train_images = image_files[:train_count]
        # test_images = image_files[train_count:train_count + test_count]
        
        # Split the balanced selection into train and test
        train_images = balanced_selection[:train_count]
        test_images = balanced_selection[train_count:train_count + test_count]
        
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
    print("📊 Final Balanced Dataset Split Summary")
    print("="*60)
    total_train = sum(s['train'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    print(f"Total training samples: {total_train}")
    print(f"Total test samples: {total_test}")
    print(f"Total samples: {total_train + total_test}")
    print(f"Grand Total Used:       {total_train + total_test}")
    print(f"Remaining Data (unused): {total_images_in_source - (total_train + total_test)}")
    print("="*60)


if __name__ == '__main__':
    sample_dataset()