"""
TrainTestSplit_Dynamic_FSL.py
Randomly samples a fixed number of .MOV videos per FSL dynamic sign.
Only processes folders whose ID exists in labels.csv.
Destination: data/processed/fsl_dynamic_final
"""

import os
import csv
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

# ← Path to your labels CSV (id, label, category)
LABELS_CSV = PROJECT_ROOT / 'data' / 'labels.csv'

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_labels_csv(csv_path):
    """
    Load labels CSV and return a dict of {str(id): label}.
    Supports formats:
      - With header:    id,label,category
      - Without header: 0,GOOD MORNING,GREETING
    Strips BOM, whitespace, and quotes from all values.
    """
    label_map = {}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        sample = f.read(1024)
        f.seek(0)

        # Auto-detect header
        sniffer = csv.Sniffer()
        has_header = sniffer.has_header(sample)

        reader = csv.reader(f)
        if has_header:
            next(reader)  # skip header row

        for row in reader:
            if len(row) < 2:
                continue
            id_str = row[0].strip().strip('"').strip("'")
            label = row[1].strip().strip('"').strip("'")
            if id_str.isdigit():
                label_map[id_str] = label

    return label_map


def sample_dynamic_dataset():
    """Sample videos for training, validation, and testing sets"""

    # 1. Validation & Setup
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        return

    if not LABELS_CSV.exists():
        print(f"❌ Labels CSV not found: {LABELS_CSV}")
        print(f"   Please place your labels CSV at: {LABELS_CSV}")
        return

    # Load CSV labels
    label_map = load_labels_csv(LABELS_CSV)
    if not label_map:
        print("❌ No valid labels found in CSV.")
        return

    print(f"📋 Loaded {len(label_map)} labels from CSV")
    print(f"   IDs: {sorted(label_map.keys(), key=lambda x: int(x))}")

    # Create output directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Find matching folders in raw directory
    all_raw_folders = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    all_raw_ids = {d.name for d in all_raw_folders}

    # Only keep folders whose name (ID) is in our CSV
    matched_folders = [d for d in all_raw_folders if d.name in label_map]
    skipped_ids = all_raw_ids - label_map.keys()

    print(f"\n📁 Raw folders found:   {len(all_raw_folders)}")
    print(f"✅ Matched to CSV:      {len(matched_folders)}")

    if skipped_ids:
        print(f"⏭️  Skipped (not in CSV): {sorted(skipped_ids, key=lambda x: int(x) if x.isdigit() else x)}")

    # Check for IDs in CSV but missing from raw folder
    missing_ids = label_map.keys() - all_raw_ids
    if missing_ids:
        print(f"⚠️  In CSV but no raw folder: {sorted(missing_ids, key=lambda x: int(x))}")
        print(f"   Labels: {[label_map[i] for i in sorted(missing_ids, key=lambda x: int(x))]}")

    if not matched_folders:
        print("❌ No matching folders to process.")
        return

    matched_folders = sorted(matched_folders, key=lambda d: int(d.name))

    # 3. Scan video counts per matched folder
    video_extensions = ['*.MOV', '*.mov', '*.mp4', '*.avi']
    class_video_counts = {}
    total_videos_in_source = 0

    for folder in matched_folders:
        videos = []
        for ext in video_extensions:
            videos.extend(list(folder.glob(ext)))
        videos = list(set(videos))  # deduplicate
        class_video_counts[folder.name] = (folder, videos)
        total_videos_in_source += len(videos)

    # 4. Split planning (70 / 15 / 15)
    min_samples = min(len(v) for _, v in class_video_counts.values())

    train_count = int(min_samples * 0.7)
    val_count = int(min_samples * 0.15)
    test_count = min_samples - train_count - val_count

    total_to_be_used = min_samples * len(matched_folders)
    usage_pct = (total_to_be_used / total_videos_in_source) * 100 if total_videos_in_source > 0 else 0

    print(f"\n📊 {len(matched_folders)} classes will be processed")
    print(f"   Total raw videos:   {total_videos_in_source}")
    print(f"   Min videos/class:   {min_samples}")
    print(f"   Split plan:         {train_count} Train / {val_count} Val / {test_count} Test per class")
    print(f"   Total to be used:   {total_to_be_used} / {total_videos_in_source} ({usage_pct:.1f}%)")
    print("-" * 60)

    # 5. Process each matched folder
    stats = defaultdict(lambda: {'label': '', 'total': 0, 'train': 0, 'val': 0, 'test': 0})

    for folder_id_str, (class_folder, video_files) in class_video_counts.items():
        label = label_map[folder_id_str]

        random.shuffle(video_files)
        balanced = video_files[:min_samples]

        train_vids = balanced[:train_count]
        val_vids = balanced[train_count:train_count + val_count]
        test_vids = balanced[train_count + val_count:]

        # Use folder ID as subfolder name (keeps consistency with your model)
        train_class_dir = TRAIN_DIR / folder_id_str
        val_class_dir = VAL_DIR / folder_id_str
        test_class_dir = TEST_DIR / folder_id_str

        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        for vid in train_vids:
            shutil.copy2(vid, train_class_dir / vid.name)
        for vid in val_vids:
            shutil.copy2(vid, val_class_dir / vid.name)
        for vid in test_vids:
            shutil.copy2(vid, test_class_dir / vid.name)

        stats[folder_id_str]['label'] = label
        stats[folder_id_str]['total'] = len(video_files)
        stats[folder_id_str]['train'] = len(train_vids)
        stats[folder_id_str]['val'] = len(val_vids)
        stats[folder_id_str]['test'] = len(test_vids)

        print(f"✅ [{folder_id_str:>3}] {label:<20} "
              f"{len(train_vids)} train / {len(val_vids)} val / {len(test_vids)} test  "
              f"(of {len(video_files)} raw)")

    # 6. Summary
    print("\n" + "=" * 60)
    print("📊 Final Balanced DYNAMIC Dataset Split Summary")
    print("=" * 60)

    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    grand_total = total_train + total_val + total_test

    print(f"Total training videos:   {total_train}")
    print(f"Total validation videos: {total_val}")
    print(f"Total test videos:       {total_test}")
    print(f"Grand Total Used:        {grand_total}")
    print("=" * 60)

    print("\n📊 Split Distribution:")
    print(f"Train: {total_train}/{grand_total} ({100 * total_train / grand_total:.1f}%)")
    print(f"Val:   {total_val}/{grand_total} ({100 * total_val / grand_total:.1f}%)")
    print(f"Test:  {total_test}/{grand_total} ({100 * total_test / grand_total:.1f}%)")

    print("\n📋 Per-class breakdown:")
    print(f"{'ID':<5} {'Label':<22} {'Total Raw':<12} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-" * 65)
    for id_str in sorted(stats.keys(), key=lambda x: int(x)):
        s = stats[id_str]
        print(f"{id_str:<5} {s['label']:<22} {s['total']:<12} {s['train']:<8} {s['val']:<8} {s['test']:<8}")


if __name__ == '__main__':
    sample_dynamic_dataset()