import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
# Make sure your training script is named train_dynamic_fsl_pro.py
from train_dynamic_fsl import ImprovedLSTMModel, FSLSequenceDataset

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'lstm_dynamic_final' / 'best_model.pth'
TUNING_CSV = PROJECT_ROOT / 'models' / 'lstm_dynamic_final' / 'tuning_results.csv'

now = datetime.now()
FOLDER_TS = now.strftime("%Y-%m-%d_%I-%M-%p")
EVAL_DIR = PROJECT_ROOT / 'runs_dynamic' / f'eval_{FOLDER_TS}'
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_dynamic_model():
    # 1. Load Model Weights
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # 2. Load Class Names from JSON (since they aren't in the .pth file)
    import json
    label_map_path = PROJECT_ROOT / 'models' / 'lstm_dynamic_final' / 'label_mapping.json'
    with open(label_map_path, 'r') as f:
        label_dict = json.load(f)
        # Sort by value (0, 1, 2...) to match model output indices
        classes = [k for k, v in sorted(label_dict.items(), key=lambda item: item[1])]
    
    tuning_df = pd.read_csv(TUNING_CSV) if TUNING_CSV.exists() else None
    
    # 3. Setup Dataset and Model
    test_ds = FSLSequenceDataset(split='test')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = ImprovedLSTMModel(input_size=126, num_classes=len(classes))
    
    # CHANGE THIS LINE: Load checkpoint directly as it is the state_dict
    model.load_state_dict(checkpoint) 
    
    model.eval()

    # 3. Inference
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(target.numpy())
            y_pred.extend(preds.numpy())

    # 4. Generate Reports
    # Report 1: Project Summary Text
    # Find the actual F1 column name (handles case sensitivity)
    
    summary_text = f"""
============================================================
📊 FSL DYNAMIC PROJECT SUMMARY & EVALUATION REPORT
============================================================
📅 Date: {now.strftime("%Y/%m/%d %I:%M:%S %p")}
📂 Run Folder: {EVAL_DIR.name}

[HYPERPARAMETER TUNING RESULTS]
✅ Best Tuning F1-Score: {tuning_df['Val_F1'].max() if tuning_df is not None else 'N/A'}
✅ Optimal Learning Rate: {tuning_df.loc[tuning_df['Val_F1'].idxmax(), 'LR'] if tuning_df is not None else 'N/A'}
✅ Optimal Dropout: {tuning_df.loc[tuning_df['Val_F1'].idxmax(), 'Dropout'] if tuning_df is not None else 'N/A'}

[DATASET INFO]
✅ Classes Detected: {len(classes)}
✅ Features: 126 Landmarks (2 Hands)
✅ Evaluation Samples: {len(test_ds)}

[FINAL TEST PERFORMANCE]
🎯 Final Test Accuracy: {np.mean(np.array(y_true) == np.array(y_pred))*100:.2f}%
💾 Model Path: {MODEL_PATH}
============================================================
"""
    with open(EVAL_DIR / 'project_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=classes))

    # Report 2: Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Dynamic FSL Confusion Matrix (Test Set)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.savefig(EVAL_DIR / 'confusion_matrix.png', dpi=300)
    plt.close()

    # Report 3: Tuning Heatmap (Visualizing the Grid Search)
    if tuning_df is not None:
        plt.figure(figsize=(10, 6))
        pivot_table = tuning_df.pivot(index='LR', columns='Dropout', values='Val_F1')
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu')
        plt.title('Hyperparameter Tuning Results (F1-Score)')
        plt.savefig(EVAL_DIR / 'tuning_heatmap.png')
        plt.close()

    print(summary_text)
    print(f"✅ Success! Dynamic results saved to: {EVAL_DIR}")

if __name__ == "__main__":
    evaluate_dynamic_model()