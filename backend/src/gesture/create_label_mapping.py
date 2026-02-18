"""
create_label_mapping.py
Creates a label mapping file from your labels.csv
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
LABELS_CSV = PROJECT_ROOT / "data" / "raw" / "fsl_dynamic" / "labels.csv"
OUTPUT_FILE = PROJECT_ROOT / "models" / "lstm_dynamic_final" / "label_mapping.json"

# Read CSV and create mapping
label_mapping = {}

# with open(LABELS_CSV, 'r', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     header = next(reader)  # Skip header
    
#     print(f"📋 CSV Headers: {header}")
#     print("\n📝 Creating label mapping...")
    
#     for row in reader:
#         if len(row) >= 2:
#             folder_id = row[0].strip()
#             label_name = row[1].strip()
#             label_mapping[folder_id] = label_name
#             print(f"   {folder_id} → {label_name}")
with open(LABELS_CSV, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)

    print("\n📝 Creating label mapping...")
    for row in reader:
        if len(row) >= 2:
            folder_id = row[0].strip()
            label_name = row[1].strip()
            label_mapping[folder_id] = label_name
            print(f"   {folder_id} → {label_name}")


# Save as JSON
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(label_mapping, f, indent=2, ensure_ascii=False)

print(f"\n✅ Label mapping saved to: {OUTPUT_FILE}")
print(f"📊 Total labels: {len(label_mapping)}")