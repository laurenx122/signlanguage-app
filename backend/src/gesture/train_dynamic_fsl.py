"""
train_dynamic_fsl.py
Trains LSTM model for FSL dynamic sign recognition with hyperparameter tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic_sequences'
MODEL_DIR = PROJECT_ROOT / 'models' / 'lstm_dynamic_final'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 126  # 2 hands * 21 landmarks * 3 coords
SEQ_LEN = 30

print(f"🖥️  Using device: {DEVICE}")

# --- Early Stopping ---
class EarlyStopping:
    """Stop training when validation loss stops improving"""
    def __init__(self, patience=10, min_delta=0.001, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# --- Model Architecture ---
class ImprovedLSTMModel(nn.Module):
    """LSTM with Conv1D preprocessing, bidirectional layers, and attention"""
    def __init__(self, input_size=126, hidden_size=256, num_layers=3, num_classes=44, dropout=0.4):
        super(ImprovedLSTMModel, self).__init__()
        
        # Conv1D for local feature extraction
        self.conv1 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm1d(256)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            256, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=8, batch_first=True
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Conv preprocessing: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn_conv(self.conv1(x)))
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling
        pooled = torch.mean(attn_out, dim=1)
        
        return self.fc(pooled)

# --- Dataset ---
class FSLSequenceDataset(Dataset):
    """Load preprocessed .npy sequences"""
    def __init__(self, split='train'):
        self.samples, self.labels = [], []
        split_dir = DATA_DIR / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Get sorted class names
        self.classes = sorted(
            [d.name for d in split_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x) if x.isdigit() else x
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load all samples
        for cls_name in self.classes:
            class_dir = split_dir / cls_name
            for npy_file in class_dir.glob("*.npy"):
                self.samples.append(npy_file)
                self.labels.append(self.class_to_idx[cls_name])
        
        print(f"📂 Loaded {len(self.samples)} samples from {split} split")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx]).astype(np.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return torch.from_numpy(data), label

# --- Training Functions ---
def evaluate_model(model, dataloader, criterion):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            all_preds.extend(output.argmax(1).cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    return avg_loss, precision, recall, f1, all_preds, all_labels

def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for data, target in dataloader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_with_tuning():
    """Main training pipeline with hyperparameter tuning"""
    
    print("="*60)
    print("🎯 FSL Dynamic Sign Language - LSTM Training Pipeline")
    print("="*60)
    
    # Load datasets
    train_ds = FSLSequenceDataset('train')
    val_ds = FSLSequenceDataset('val')
    test_ds = FSLSequenceDataset('test')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=0)
    
    num_classes = len(train_ds.classes)
    print(f"📊 Dataset: {num_classes} classes")
    print(f"   Train: {len(train_ds)} samples")
    print(f"   Val:   {len(val_ds)} samples")
    print(f"   Test:  {len(test_ds)} samples")
    
    # === PHASE 1: HYPERPARAMETER TUNING ===
    print("\n" + "="*60)
    print("🔎 PHASE 1: Hyperparameter Tuning")
    print("="*60)
    
    lr_options = [0.001, 0.0005]
    dropout_options = [0.3, 0.5]
    tuning_results = []
    
    for lr in lr_options:
        for dropout in dropout_options:
            print(f"\n🧪 Testing: LR={lr}, Dropout={dropout}")
            
            model = ImprovedLSTMModel(
                num_classes=num_classes, 
                dropout=dropout
            ).to(DEVICE)
            
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Quick tuning run (15 epochs)
            for epoch in range(1, 16):
                train_loss = train_epoch(model, train_loader, optimizer, criterion)
                
                if epoch % 5 == 0:
                    val_loss, _, _, val_f1, _, _ = evaluate_model(model, val_loader, criterion)
                    print(f"   Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            # Final validation evaluation
            val_loss, val_prec, val_rec, val_f1, _, _ = evaluate_model(model, val_loader, criterion)
            
            tuning_results.append({
                "LR": lr,
                "Dropout": dropout,
                "Val_Loss": val_loss,
                "Val_Precision": val_prec,
                "Val_Recall": val_rec,
                "Val_F1": val_f1
            })
            
            print(f"   ✅ Final Val F1: {val_f1:.4f}")
    
    # Save tuning results
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(MODEL_DIR / 'tuning_results.csv', index=False)
    print(f"\n📊 Tuning results saved to: {MODEL_DIR / 'tuning_results.csv'}")
    
    # Find best configuration
    best_config = max(tuning_results, key=lambda x: x['Val_F1'])
    print("\n" + "="*60)
    print("🏆 BEST CONFIGURATION:")
    print(f"   LR:      {best_config['LR']}")
    print(f"   Dropout: {best_config['Dropout']}")
    print(f"   Val F1:  {best_config['Val_F1']:.4f}")
    print("="*60)
    
    # === PHASE 2: FINAL TRAINING ===
    print("\n" + "="*60)
    print("🚀 PHASE 2: Final Training with Best Config")
    print("="*60)
    
    final_model = ImprovedLSTMModel(
        num_classes=num_classes,
        dropout=best_config['Dropout']
    ).to(DEVICE)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=best_config['LR'])
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=10,
        path=MODEL_DIR / 'best_model.pth'
    )
    
    training_history = []
    
    for epoch in range(1, 101):
        # Train
        train_loss = train_epoch(final_model, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_prec, val_rec, val_f1, _, _ = evaluate_model(final_model, val_loader, criterion)
        
        # Record history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1
        })
        
        # Print progress
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        # Early stopping check
        early_stopping(val_loss, final_model)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            break
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(MODEL_DIR / 'training_history.csv', index=False)
    
    # === PHASE 3: FINAL EVALUATION ===
    print("\n" + "="*60)
    print("📊 PHASE 3: Final Model Evaluation")
    print("="*60)
    
    # Load best model
    final_model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth'))
    
    # Evaluate on test set
    test_loss, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate_model(
        final_model, test_loader, criterion
    )
    
    print(f"\n🎯 TEST SET RESULTS:")
    print(f"   Loss:      {test_loss:.4f}")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall:    {test_rec:.4f}")
    print(f"   F1 Score:  {test_f1:.4f}")
    
    # Save final model with metadata
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'classes': train_ds.classes,
        'class_to_idx': train_ds.class_to_idx,
        'num_classes': num_classes,
        'best_config': best_config,
        'test_metrics': {
            'loss': test_loss,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1
        }
    }, MODEL_DIR / 'final_model_complete.pth')
    
    # Save metadata separately
    metadata = {
        'classes': train_ds.classes,
        'class_to_idx': train_ds.class_to_idx,
        'num_classes': num_classes,
        'best_hyperparameters': best_config,
        'test_performance': {
            'loss': float(test_loss),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1': float(test_f1)
        }
    }
    
    with open(MODEL_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print(f"📁 Models saved to: {MODEL_DIR}")
    print("   - best_model.pth (weights only)")
    print("   - final_model_complete.pth (weights + metadata)")
    print("   - model_metadata.json")
    print("   - tuning_results.csv")
    print("   - training_history.csv")
    print("="*60)

if __name__ == "__main__":
    train_with_tuning()