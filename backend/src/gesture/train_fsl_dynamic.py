"""
train_fsl_dynamic.py
Enhanced LSTM training with Train/Test split and detailed TP/FP/TN/FN metrics
Location: D:/SMS/backend/src/gesture/train_fsl_dynamic_12_enhanced.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import json

# Configuration
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2] 
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic_12'
MODEL_DIR = PROJECT_ROOT / 'models' / 'lstm_dynamic_14'

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Metrics tracking
SAVE_METRICS_EVERY = 5


class TemporalAugmentation:
    @staticmethod
    def time_warp(sequence, warp_factor=0.2):
        seq_len = len(sequence)
        warp = 1.0 + warp_factor * np.sin(np.linspace(0, 2*np.pi, seq_len))
        cumsum = np.cumsum(warp)
        cumsum = (cumsum - cumsum[0]) / (cumsum[-1] - cumsum[0]) * (seq_len - 1)
        
        warped_seq = np.zeros_like(sequence)
        for i in range(seq_len):
            idx = int(cumsum[i])
            alpha = cumsum[i] - idx
            if idx < seq_len - 1:
                warped_seq[i] = (1 - alpha) * sequence[idx] + alpha * sequence[idx + 1]
            else:
                warped_seq[i] = sequence[-1]
        return warped_seq
    
    @staticmethod
    def speed_variation(sequence, speed_range=(0.7, 1.3)):
        seq_len = len(sequence)
        speed = np.random.uniform(*speed_range)
        new_len = int(seq_len * speed)
        
        old_indices = np.linspace(0, seq_len - 1, new_len)
        new_sequence = np.zeros((new_len, sequence.shape[1]))
        
        for i in range(new_len):
            idx = int(old_indices[i])
            alpha = old_indices[i] - idx
            if idx < seq_len - 1:
                new_sequence[i] = (1 - alpha) * sequence[idx] + alpha * sequence[idx + 1]
            else:
                new_sequence[i] = sequence[-1]
        
        if new_len < seq_len:
            padding = np.zeros((seq_len - new_len, sequence.shape[1]))
            return np.vstack([new_sequence, padding])
        else:
            return new_sequence[:seq_len]
    
    @staticmethod
    def add_jitter(sequence, sigma=0.005):
        noise = np.random.normal(0, sigma, sequence.shape)
        mask = (sequence != 0).astype(np.float32)
        return sequence + noise * mask
    
    @staticmethod
    def random_crop(sequence, crop_ratio=0.9):
        seq_len = len(sequence)
        crop_len = int(seq_len * crop_ratio)
        start_idx = np.random.randint(0, seq_len - crop_len + 1)
        
        cropped = sequence[start_idx:start_idx + crop_len]
        padding = np.zeros((seq_len - crop_len, sequence.shape[1]))
        return np.vstack([cropped, padding])


class ImprovedSignDataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.aug = TemporalAugmentation()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        
        if self.augment:
            if np.random.random() > 0.6:
                sequence = self.aug.speed_variation(sequence)
            
            if np.random.random() > 0.5:
                sequence = self.aug.time_warp(sequence)
            
            if np.random.random() > 0.7:
                sequence = self.aug.random_crop(sequence)
            
            sequence = self.aug.add_jitter(sequence)
        
        return (
            torch.FloatTensor(sequence),
            torch.LongTensor([self.labels[idx]])[0]
        )


class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_layers=3, 
                 num_classes=14, dropout=0.4):
        super(ImprovedLSTMModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, 256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn_conv = nn.BatchNorm1d(256)
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_t = x.transpose(1, 2)
        x_t = torch.relu(self.conv1(x_t))
        x_t = torch.relu(self.conv2(x_t))
        x_t = self.bn_conv(x_t)
        x_t = x_t.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x_t)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = torch.mean(attn_out, dim=1)
        
        out = self.fc1(pooled)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out


def calculate_class_weights(labels, num_classes):
    """Calculate class weights for imbalanced dataset"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    return torch.FloatTensor(class_weights)


def calculate_tp_fp_tn_fn(y_true, y_pred, num_classes):
    """
    Calculate TP, FP, TN, FN for each class
    Returns dictionary with per-class metrics
    """
    metrics_per_class = {}
    
    for class_idx in range(num_classes):
        # Binary classification for this class
        true_positive = np.sum((y_true == class_idx) & (y_pred == class_idx))
        false_positive = np.sum((y_true != class_idx) & (y_pred == class_idx))
        true_negative = np.sum((y_true != class_idx) & (y_pred != class_idx))
        false_negative = np.sum((y_true == class_idx) & (y_pred != class_idx))
        
        # Calculate metrics
        total = len(y_true)
        accuracy = (true_positive + true_negative) / total if total > 0 else 0
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_per_class[class_idx] = {
            'TP': int(true_positive),
            'FP': int(false_positive),
            'TN': int(true_negative),
            'FN': int(false_negative),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1': f1
        }
    
    return metrics_per_class


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(dataloader, desc='Training'):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(dataloader), 100.*correct/total


def validate_with_comprehensive_metrics(model, dataloader, criterion, device, idx_to_label, 
                                        epoch=0, split_name='Validation'):
    """
    Enhanced validation with comprehensive metrics including TP/FP/TN/FN
    """
    model.eval()
    running_loss = 0.0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc=f'{split_name}'):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Calculate metrics
    metrics = {}
    
    # 1. Top-1 Accuracy
    top1_accuracy = 100.0 * np.mean(all_predictions == all_labels)
    metrics['top1_accuracy'] = top1_accuracy
    
    # 2. Top-3 Accuracy
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:]
    top3_correct = np.array([label in top3_preds[i] for i, label in enumerate(all_labels)])
    top3_accuracy = 100.0 * np.mean(top3_correct)
    metrics['top3_accuracy'] = top3_accuracy
    
    # 3. Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    metrics['per_class'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }
    
    # 4. TP/FP/TN/FN metrics
    num_classes = len(idx_to_label)
    tp_fp_tn_fn_metrics = calculate_tp_fp_tn_fn(all_labels, all_predictions, num_classes)
    metrics['tp_fp_tn_fn'] = tp_fp_tn_fn_metrics
    
    # 5. Macro averages
    metrics['macro_precision'] = np.mean(precision)
    metrics['macro_recall'] = np.mean(recall)
    metrics['macro_f1'] = np.mean(f1)
    
    # 6. Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    metrics['confusion_matrix'] = cm
    
    # 7. Identify worst performing classes
    worst_classes_idx = np.argsort(f1)[:3]
    metrics['worst_classes'] = [
        (idx_to_label[idx], f1[idx], support[idx]) 
        for idx in worst_classes_idx if support[idx] > 0
    ]
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"{split_name.upper()} METRICS - Epoch {epoch}")
    print(f"{'='*80}")
    print(f"Loss: {running_loss/len(dataloader):.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    
    # Print detailed per-class table
    print(f"\n📊 PER-CLASS PERFORMANCE:")
    print(f"{'Class':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'Samples':>7}")
    print("-" * 80)
    for i, class_name in idx_to_label.items():
        tp_metrics = tp_fp_tn_fn_metrics[i]
        print(f"{class_name:<22} "
              f"{precision[i]:>6.3f} {recall[i]:>6.3f} {f1[i]:>6.3f} "
              f"{tp_metrics['TP']:>5d} {tp_metrics['FP']:>5d} "
              f"{tp_metrics['TN']:>5d} {tp_metrics['FN']:>5d} "
              f"{int(support[i]):>7d}")
    
    if len(metrics['worst_classes']) > 0:
        print(f"\n⚠️  WORST PERFORMING CLASSES:")
        for class_name, f1_score, samples in metrics['worst_classes']:
            print(f"   {class_name:<22} - F1: {f1_score:.3f} ({int(samples)} samples)")
    
    return running_loss/len(dataloader), metrics


def save_confusion_matrix_table(cm, idx_to_label, epoch, save_path_csv, save_path_png):
    """
    Save confusion matrix as both CSV table and visualization
    """
    # Create DataFrame for CSV
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    
    # Absolute values
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'True Label'
    cm_df.to_csv(save_path_csv)
    print(f"   📊 Confusion matrix (CSV) saved: {save_path_csv}")
    
    # Normalized visualization
    plt.figure(figsize=(14, 12))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, 
                cbar_kws={'label': 'Percentage'},
                square=True)
    
    plt.title(f'Confusion Matrix - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Confusion matrix (PNG) saved: {save_path_png}")


def save_tp_fp_tn_fn_table(tp_fp_tn_fn_metrics, idx_to_label, epoch, save_path):
    """
    Save TP/FP/TN/FN metrics as CSV table
    """
    rows = []
    for class_idx, metrics in tp_fp_tn_fn_metrics.items():
        row = {
            'Class': idx_to_label[class_idx],
            'True_Positive': metrics['TP'],
            'False_Positive': metrics['FP'],
            'True_Negative': metrics['TN'],
            'False_Negative': metrics['FN'],
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'Specificity': f"{metrics['Specificity']:.4f}",
            'F1_Score': f"{metrics['F1']:.4f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"   📊 TP/FP/TN/FN table saved: {save_path}")


def save_metrics_plot(history, save_path):
    """Save training history plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    if 'test_loss' in history and len(history['test_loss']) > 0:
        axes[0, 0].axhline(y=history['test_loss'][-1], color='g', linestyle='--', 
                          label=f"Test Loss ({history['test_loss'][-1]:.4f})", linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-1 Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_top1_acc'], 'r-', label='Val Top-1 Acc', linewidth=2)
    if 'test_top1_acc' in history and len(history['test_top1_acc']) > 0:
        axes[0, 1].axhline(y=history['test_top1_acc'][-1], color='g', linestyle='--',
                          label=f"Test Top-1 ({history['test_top1_acc'][-1]:.2f}%)", linewidth=2)
    axes[0, 1].set_title('Top-1 Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-3 Accuracy
    axes[1, 0].plot(epochs, history['val_top3_acc'], 'g-', label='Val Top-3 Acc', linewidth=2)
    if 'test_top3_acc' in history and len(history['test_top3_acc']) > 0:
        axes[1, 0].axhline(y=history['test_top3_acc'][-1], color='orange', linestyle='--',
                          label=f"Test Top-3 ({history['test_top3_acc'][-1]:.2f}%)", linewidth=2)
    axes[1, 0].set_title('Top-3 Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1-Score
    axes[1, 1].plot(epochs, history['val_f1'], 'purple', label='Val F1', linewidth=2)
    if 'test_f1' in history and len(history['test_f1']) > 0:
        axes[1, 1].axhline(y=history['test_f1'][-1], color='brown', linestyle='--',
                          label=f"Test F1 ({history['test_f1'][-1]:.4f})", linewidth=2)
    axes[1, 1].set_title('Macro F1-Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Training history saved: {save_path}")


def main():
    print("="*80)
    print("🚀 FSL DYNAMIC LSTM TRAINING - 14 Gestures")
    print("   With Train/Test Split & Comprehensive Metrics (TP/FP/TN/FN)")
    print("="*80)
    print(f"Using device: {DEVICE}")
    print(f"\n📊 Data Split: Train {TRAIN_RATIO:.0%} | Val {VAL_RATIO:.0%} | Test {TEST_RATIO:.0%}")
    
    # Load data
    print("\n📂 Loading data...")
    X = np.load(DATA_DIR / 'sequences_X.npy')
    y = np.load(DATA_DIR / 'labels_y.npy')
    
    with open(DATA_DIR / 'label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
    
    idx_to_label = label_mapping['idx_to_label']
    num_classes = len(label_mapping['label_to_idx'])
    
    print(f"✅ Loaded {len(X)} sequences")
    print(f"📊 Shape: {X.shape}")
    print(f"📊 Classes: {num_classes}")
    
    # Three-way split: Train, Validation, Test
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )
    
    # Second split: train and validation from remaining data
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Train:      {len(X_train):4d} sequences ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val):4d} sequences ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:       {len(X_test):4d} sequences ({len(X_test)/len(X)*100:.1f}%)")
    
    # Show class distribution across splits
    print(f"\n📊 Class distribution across splits:")
    print(f"{'Class':<22} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("-" * 80)
    for idx in range(num_classes):
        train_count = np.sum(y_train == idx)
        val_count = np.sum(y_val == idx)
        test_count = np.sum(y_test == idx)
        total_count = train_count + val_count + test_count
        print(f"{idx_to_label[idx]:<22} {train_count:>7d} {val_count:>7d} "
              f"{test_count:>7d} {total_count:>7d}")
    
    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(y_train, num_classes)
    print(f"\n⚖️  Class weights (calculated from training set):")
    for idx in range(num_classes):
        print(f"   {idx_to_label[idx]:<22} - weight: {class_weights[idx]:.3f}")
    
    # Create datasets
    train_dataset = ImprovedSignDataset(X_train, y_train, augment=True)
    val_dataset = ImprovedSignDataset(X_val, y_val, augment=False)
    test_dataset = ImprovedSignDataset(X_test, y_test, augment=False)
    
    print(f"\n📊 Dataset sizes:")
    print(f"   Train:      {len(train_dataset)} (Augmentation: ON)")
    print(f"   Validation: {len(val_dataset)} (Augmentation: OFF)")
    print(f"   Test:       {len(test_dataset)} (Augmentation: OFF)")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    model = ImprovedLSTMModel(
        input_size=126,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"\n🏗️  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_top1_acc': [],
        'val_top3_acc': [],
        'val_f1': [],
        'test_loss': [],
        'test_top1_acc': [],
        'test_top3_acc': [],
        'test_f1': []
    }
    
    best_val_f1 = 0.0
    best_val_top1 = 0.0
    patience = 20
    patience_counter = 0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*80}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics = validate_with_comprehensive_metrics(
            model, val_loader, criterion, DEVICE, idx_to_label, epoch+1, 'Validation'
        )
        
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_top1_acc'].append(val_metrics['top1_accuracy'])
        history['val_top3_acc'].append(val_metrics['top3_accuracy'])
        history['val_f1'].append(val_metrics['macro_f1'])
        
        print(f"\n📊 Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Top-1: {val_metrics['top1_accuracy']:.2f}% | "
              f"Val Top-3: {val_metrics['top3_accuracy']:.2f}% | Val F1: {val_metrics['macro_f1']:.4f}")
        
        # Save best model based on F1-score
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_val_top1 = val_metrics['top1_accuracy']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['macro_f1'],
                'val_top1_acc': val_metrics['top1_accuracy'],
                'val_top3_acc': val_metrics['top3_accuracy'],
                'label_mapping': label_mapping,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'sequence_length': X.shape[1],
                'input_size': 126,
                'model_type': 'ImprovedLSTM',
                'train_ratio': TRAIN_RATIO,
                'val_ratio': VAL_RATIO,
                'test_ratio': TEST_RATIO
            }
            
            torch.save(checkpoint, MODEL_DIR / 'best_model.pth')
            print(f"\n✓ Best model saved (F1: {val_metrics['macro_f1']:.4f}, "
                  f"Top-1: {val_metrics['top1_accuracy']:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        # Save detailed metrics periodically
        if (epoch + 1) % SAVE_METRICS_EVERY == 0:
            print(f"\n💾 Saving detailed metrics...")
            
            # Confusion matrix (both CSV and PNG)
            save_confusion_matrix_table(
                val_metrics['confusion_matrix'],
                idx_to_label,
                epoch + 1,
                MODEL_DIR / f'confusion_matrix_epoch_{epoch+1}.csv',
                MODEL_DIR / f'confusion_matrix_epoch_{epoch+1}.png'
            )
            
            # TP/FP/TN/FN table
            save_tp_fp_tn_fn_table(
                val_metrics['tp_fp_tn_fn'],
                idx_to_label,
                epoch + 1,
                MODEL_DIR / f'tp_fp_tn_fn_epoch_{epoch+1}.csv'
            )
            
            # Training history plot
            save_metrics_plot(
                history,
                MODEL_DIR / f'training_history_epoch_{epoch+1}.png'
            )
    
    # ============================================================================
    # FINAL TEST SET EVALUATION
    # ============================================================================
    print(f"\n{'='*80}")
    print("🧪 FINAL TEST SET EVALUATION")
    print(f"{'='*80}")
    
    # Load best model
    checkpoint = torch.load(
        MODEL_DIR / 'best_model.pth',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = validate_with_comprehensive_metrics(
        model, test_loader, criterion, DEVICE, idx_to_label, 
        epoch=checkpoint['epoch']+1, split_name='Test'
    )
    
    # Update history with test results
    history['test_loss'].append(test_loss)
    history['test_top1_acc'].append(test_metrics['top1_accuracy'])
    history['test_top3_acc'].append(test_metrics['top3_accuracy'])
    history['test_f1'].append(test_metrics['macro_f1'])
    
    # Save final test metrics
    print(f"\n💾 Saving final test metrics...")
    
    save_confusion_matrix_table(
        test_metrics['confusion_matrix'],
        idx_to_label,
        'FINAL_TEST',
        MODEL_DIR / 'confusion_matrix_test_final.csv',
        MODEL_DIR / 'confusion_matrix_test_final.png'
    )
    
    save_tp_fp_tn_fn_table(
        test_metrics['tp_fp_tn_fn'],
        idx_to_label,
        'FINAL_TEST',
        MODEL_DIR / 'tp_fp_tn_fn_test_final.csv'
    )
    
    # Save complete metrics report
    final_report = {
        'validation': {
            'top1_accuracy': best_val_top1,
            'f1_score': best_val_f1
        },
        'test': {
            'loss': test_loss,
            'top1_accuracy': test_metrics['top1_accuracy'],
            'top3_accuracy': test_metrics['top3_accuracy'],
            'macro_precision': test_metrics['macro_precision'],
            'macro_recall': test_metrics['macro_recall'],
            'macro_f1': test_metrics['macro_f1']
        },
        'per_class_test': {}
    }
    
    for idx in range(num_classes):
        class_name = idx_to_label[idx]
        final_report['per_class_test'][class_name] = test_metrics['tp_fp_tn_fn'][idx]
    
    with open(MODEL_DIR / 'final_test_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"   📊 Final test report saved: {MODEL_DIR / 'final_test_report.json'}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 BEST VALIDATION RESULTS:")
    print(f"   Top-1 Accuracy: {best_val_top1:.2f}%")
    print(f"   F1-Score:       {best_val_f1:.4f}")
    
    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"   Loss:           {test_loss:.4f}")
    print(f"   Top-1 Accuracy: {test_metrics['top1_accuracy']:.2f}%")
    print(f"   Top-3 Accuracy: {test_metrics['top3_accuracy']:.2f}%")
    print(f"   Macro F1-Score: {test_metrics['macro_f1']:.4f}")
    print(f"   Macro Precision:{test_metrics['macro_precision']:.4f}")
    print(f"   Macro Recall:   {test_metrics['macro_recall']:.4f}")
    
    # Save final visualizations
    print(f"\n💾 Saving final visualizations...")
    save_metrics_plot(history, MODEL_DIR / 'final_training_history.png')
    
    # Save training history
    with open(MODEL_DIR / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\n📁 All files saved in: {MODEL_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()