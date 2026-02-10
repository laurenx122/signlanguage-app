import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from train_dynamic_fsl import ImprovedLSTMModel, FSLSequenceDataset
import warnings

# Suppress the sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'lstm_dynamic_final' / 'best_model.pth'
TUNING_CSV = PROJECT_ROOT / 'models' / 'lstm_dynamic_final' / 'tuning_results.csv'
TRAINING_LOG = PROJECT_ROOT / 'models' / 'lstm_dynamic_final' / 'training_history.csv'

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

    # Load training log to get epoch information
    training_log_df = pd.read_csv(TRAINING_LOG) if TRAINING_LOG.exists() else None
    
    # 3. Setup Dataset and Model
    test_ds = FSLSequenceDataset(split='test')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = ImprovedLSTMModel(input_size=126, num_classes=len(classes))
    
    # Load checkpoint directly as it is the state_dict
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
    # Extract epoch information
    epoch_info = ""
    best_epoch_num = None
    best_val_f1 = None
    
    if training_log_df is not None and len(training_log_df) > 0:
        total_epochs = len(training_log_df)
        first_epoch = training_log_df.iloc[0]
        middle_epoch = training_log_df.iloc[total_epochs // 2]
        last_epoch = training_log_df.iloc[-1]
        
        # Find best epoch (highest val_f1)
        best_idx = training_log_df['val_f1'].idxmax()
        best_epoch = training_log_df.iloc[best_idx]
        best_epoch_num = int(best_epoch['epoch'])
        best_val_f1 = best_epoch['val_f1']
        
        epoch_info = f"""
        [TRAINING PROGRESS]
        ✅ Total Epochs Trained: {total_epochs}
        🏆 Best Model Saved at Epoch: {best_epoch_num} (Val F1: {best_val_f1:.4f} = {best_val_f1*100:.2f}%)

        📈 First Epoch (1):
        • Train Loss: {first_epoch['train_loss']:.4f}
        • Val Loss: {first_epoch['val_loss']:.4f}
        • Val F1: {first_epoch['val_f1']:.4f} ({first_epoch['val_f1']*100:.2f}%)

        📊 Middle Epoch ({int(middle_epoch['epoch'])}):
        • Train Loss: {middle_epoch['train_loss']:.4f}
        • Val Loss: {middle_epoch['val_loss']:.4f}
        • Val F1: {middle_epoch['val_f1']:.4f} ({middle_epoch['val_f1']*100:.2f}%)

        🎯 Final Epoch ({int(last_epoch['epoch'])}):
        • Train Loss: {last_epoch['train_loss']:.4f}
        • Val Loss: {last_epoch['val_loss']:.4f}
        • Val F1: {last_epoch['val_f1']:.4f} ({last_epoch['val_f1']*100:.2f}%)
        """
    
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

{epoch_info}

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
        f.write(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

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

    # Report 4: Training Progress Curves
    if training_log_df is not None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Loss curves
        ax1 = axes[0]
        epochs = training_log_df['epoch']
        ax1.plot(epochs, training_log_df['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax1.plot(epochs, training_log_df['val_loss'], 'r-', linewidth=2, label='Val Loss')
        
        # Mark best epoch
        if best_epoch_num is not None:
            ax1.axvline(x=best_epoch_num, color='green', linestyle='--', linewidth=2, 
                       label=f'Best Model (Epoch {best_epoch_num})')
            ax1.scatter([best_epoch_num], [training_log_df.loc[training_log_df['epoch']==best_epoch_num, 'val_loss'].values[0]], 
                       color='green', s=200, zorder=5, marker='*')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 Score curve
        ax2 = axes[1]
        ax2.plot(epochs, training_log_df['val_f1'] * 100, 'g-', linewidth=2, label='Val F1 Score')
        
        # Mark best epoch
        if best_epoch_num is not None:
            ax2.axvline(x=best_epoch_num, color='green', linestyle='--', linewidth=2,
                       label=f'Best Model (Epoch {best_epoch_num})')
            ax2.scatter([best_epoch_num], [best_val_f1 * 100], 
                       color='green', s=200, zorder=5, marker='*')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('F1 Score (%)', fontsize=12)
        ax2.set_title('Validation F1 Score Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(EVAL_DIR / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Report 5: Training Summary Statistics Table
        summary_stats = pd.DataFrame({
            'Metric': ['Initial Val F1', 'Best Val F1', 'Final Val F1', 'Improvement'],
            'Value': [
                f"{training_log_df.iloc[0]['val_f1']*100:.2f}%",
                f"{best_val_f1*100:.2f}%" if best_val_f1 else "N/A",
                f"{training_log_df.iloc[-1]['val_f1']*100:.2f}%",
                f"+{(best_val_f1 - training_log_df.iloc[0]['val_f1'])*100:.2f}%" if best_val_f1 else "N/A"
            ]
        })
        summary_stats.to_csv(EVAL_DIR / 'training_summary.csv', index=False)

    print(summary_text)
    print(f"✅ Success! Dynamic results saved to: {EVAL_DIR}")

if __name__ == "__main__":
    evaluate_dynamic_model()