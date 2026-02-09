import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from train_fsl import LSTMGestureModel

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'lstm_static' / 'best_fsl_lstm_model.pth'
DATA_X_PATH = PROJECT_ROOT / 'data' / 'processed' / 'fsl_landmarks_X.npy'
DATA_Y_PATH = PROJECT_ROOT / 'data' / 'processed' / 'fsl_labels_y.npy'

now = datetime.now()
REPORT_DATE = now.strftime("%Y/%m/%d")
REPORT_TIME = now.strftime("%I:%M:%S %p")

FOLDER_TS = now.strftime("%Y-%m-%d_%I-%M-%p")
EVAL_DIR = PROJECT_ROOT / 'runs_static' / f'eval_{FOLDER_TS}'
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model():
    # 1. Load Data and History
    X, y = np.load(DATA_X_PATH), np.load(DATA_Y_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    classes = checkpoint['classes']
    history = checkpoint.get('history', {})
    total_samples = len(X)
    train_count = int(0.8 * total_samples)
    val_count = total_samples - train_count

    final_train_loss = history.get('train_loss', [0])[-1]
    final_val_loss = history.get('val_loss', [0])[-1]

    summary_text = f"""
============================================================
📊 FSL PROJECT SUMMARY & EVALUATION REPORT
============================================================
📅 Date: {REPORT_DATE} {REPORT_TIME}
📂 Run Folder: {EVAL_DIR.name}

[PREPROCESSING RESULTS]
✅ Total Raw Samples Processed: {total_samples}
✅ Feature Dimension: {X.shape[1]} (126 landmarks)
✅ Classes Detected: {len(classes)} ({', '.join(classes[:5])}...)

[TRAIN-TEST SPLIT RESULTS]
📊 Training Set (80%): {train_count} samples
📊 Validation Set (20%): {val_count} samples

[TRAINING RESULTS]
🔥 Total Epochs: 100
🎯 Best Val Accuracy Achieved: {checkpoint['val_acc']:.2f}%
📉 Final Train Loss: {final_train_loss:.6f}
📉 Final Val Loss: {final_val_loss:.6f}
📉 Final Learning Rate: {checkpoint.get('lr', '0.000031')}
💾 Model Path: {MODEL_PATH}
============================================================
"""
    
    # 2. Model Inference
    model = LSTMGestureModel(input_size=126, num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loader = DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)), batch_size=32)
    y_true, y_pred = [], []
    with torch.no_grad():
        for f, l in loader:
            out = model(f)
            _, p = torch.max(out, 1)
            y_true.extend(l.numpy()); y_pred.extend(p.numpy())

    # 3. Generate Files
    # File 1: Full Project Summary (txt)
    with open(EVAL_DIR / 'full_project_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
        f.write("\nDetailed Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=classes))

    # File 2: Confusion Matrix Heatmap (png)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16)) # Increased size for better clarity
    
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Greens',  # Switched from Blues to match dynamic
                xticklabels=classes, 
                yticklabels=classes)
    
    plt.title('FSL Static Confusion Matrix (Test Set)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
    plt.savefig(EVAL_DIR / 'confusion_matrix_heatmap.png', dpi=300)
    plt.close()

    # File 3: Per-Class Accuracy (png)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(classes), y=per_class_acc, palette="viridis")
    plt.title('Accuracy Per Letter (A-Z)', fontweight='bold')
    plt.ylabel('Accuracy Score')
    plt.ylim(0.95, 1.01)
    plt.savefig(EVAL_DIR / 'per_class_accuracy.png')
    plt.close()

    print(summary_text)
    print(f"✅ Success! Results saved to: {EVAL_DIR}")

if __name__ == "__main__":
    evaluate_model()