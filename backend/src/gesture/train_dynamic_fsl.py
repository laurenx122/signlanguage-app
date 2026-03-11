"""
train_dynamic_fsl.py
Trains LSTM model for FSL dynamic sign recognition with hyperparameter tuning.

CHANGES FROM ORIGINAL:
  [CHANGE 1] Tuning epochs bumped from 15 → 30 (better convergence visibility)
  [CHANGE 2] Added ReduceLROnPlateau scheduler during main training (soft fine-tune)
  [CHANGE 3] Added Phase 3: Fine-tune pass after early stopping (10x lower LR, patience=5)
  [CHANGE 4] Scheduler state saved/loaded correctly across fine-tune phase
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
INPUT_SIZE = 252  # 126 position + 126 velocity (after add_velocity_features)
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

    # =========================================================
    # [CHANGE 3] Added reset() so fine-tune phase gets a fresh
    #            early stopping counter without creating a new object.
    #            Belongs to: EarlyStopping class
    # =========================================================
    def reset(self, new_patience=None):
        """Reset counter and best_loss for a second fine-tune pass."""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        if new_patience is not None:
            self.patience = new_patience

# --- Model Architecture ---
class ImprovedLSTMModel(nn.Module):
    """LSTM with Conv1D preprocessing, bidirectional layers, and attention"""
    def __init__(self, input_size=252, hidden_size=256, num_layers=3, num_classes=44, dropout=0.4):
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
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state from both directions
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        pooled = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Attention over all timesteps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_pooled = torch.mean(attn_out, dim=1)
        
        # Combine last hidden state + attention pooling
        pooled = pooled + attn_pooled

        return self.fc(pooled)

# --- Dataset ---
class FSLSequenceDataset(Dataset):
    """Load preprocessed .npy sequences"""
    def __init__(self, split='train'):
        self.split = split
        self.samples, self.labels = [], []
        split_dir = DATA_DIR / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        self.classes = sorted(
            [d.name for d in split_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x) if x.isdigit() else x
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
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
        
        if self.split == 'train':
            max_shift = 5
            shift = np.random.randint(-max_shift, max_shift)
            feat_dim = data.shape[1]
            if shift > 0:
                data = np.concatenate([np.zeros((shift, feat_dim)), data[:-shift]], axis=0)
            elif shift < 0:
                data = np.concatenate([data[-shift:], np.zeros((-shift, feat_dim))], axis=0)
        
        return torch.from_numpy(data).float(), torch.tensor(self.labels[idx], dtype=torch.long)

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
    
    # =========================================================
    # PHASE 1: HYPERPARAMETER TUNING
    # [CHANGE 1] Bumped tuning epochs from 15 → 30
    #            Reason: Heavy architecture (BiLSTM + attention) needs
    #            more epochs to show true generalization differences
    #            between hyperparameter combos. 15 was too short and
    #            could pick the wrong "best" config.
    # =========================================================
    print("\n" + "="*60)
    print("🔎 PHASE 1: Hyperparameter Tuning")
    print("="*60)
    
    lr_options = [0.001, 0.0005]
    dropout_options = [0.3, 0.5]
    tuning_results = []

    TUNING_EPOCHS = 30  # [CHANGE 1] was: 15

    for lr in lr_options:
        for dropout in dropout_options:
            print(f"\n🧪 Testing: LR={lr}, Dropout={dropout}")
            
            model = ImprovedLSTMModel(
                input_size=INPUT_SIZE,
                num_classes=num_classes, 
                dropout=dropout
            ).to(DEVICE)
            
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # [CHANGE 1] Loop now runs to TUNING_EPOCHS (30) instead of 16
            for epoch in range(1, TUNING_EPOCHS + 1):
                train_loss = train_epoch(model, train_loader, optimizer, criterion)
                
                if epoch % 10 == 0:  # [CHANGE 1] print every 10 instead of 5 to keep output clean
                    val_loss, _, _, val_f1, _, _ = evaluate_model(model, val_loader, criterion)
                    print(f"   Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
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
    
    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(MODEL_DIR / 'tuning_results.csv', index=False)
    print(f"\n📊 Tuning results saved to: {MODEL_DIR / 'tuning_results.csv'}")
    
    best_config = max(tuning_results, key=lambda x: x['Val_F1'])
    print("\n" + "="*60)
    print("🏆 BEST CONFIGURATION:")
    print(f"   LR:      {best_config['LR']}")
    print(f"   Dropout: {best_config['Dropout']}")
    print(f"   Val F1:  {best_config['Val_F1']:.4f}")
    print("="*60)
    
    # =========================================================
    # PHASE 2: MAIN TRAINING WITH ReduceLROnPlateau
    # [CHANGE 2] Added ReduceLROnPlateau scheduler
    #            Reason: When val loss stalls, LR is automatically
    #            halved (factor=0.5). This acts like a built-in
    #            soft fine-tune — the model keeps refining instead
    #            of just sitting at a plateau waiting for early stop.
    #            patience=5 means it waits 5 epochs of no improvement
    #            before reducing. min_lr=1e-6 prevents LR going to zero.
    # =========================================================
    print("\n" + "="*60)
    print("🚀 PHASE 2: Main Training with Best Config + LR Scheduler")
    print("="*60)
    
    final_model = ImprovedLSTMModel(
        input_size=INPUT_SIZE,
        num_classes=num_classes,
        dropout=best_config['Dropout']
    ).to(DEVICE)
    
    optimizer = optim.AdamW(final_model.parameters(), lr=best_config['LR'])
    criterion = nn.CrossEntropyLoss()

    # [CHANGE 2] ReduceLROnPlateau — halves LR when val loss stops improving
    # Note: verbose= was removed in newer PyTorch versions; LR is printed manually via current_lr below
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # [CHANGE 2]
        optimizer,
        mode='min',         # watching val loss (lower = better)
        factor=0.5,         # multiply LR by 0.5 on plateau
        patience=5,         # wait 5 epochs before reducing
        min_lr=1e-6         # never go below this
    )

    early_stopping = EarlyStopping(
        patience=10,
        path=MODEL_DIR / 'best_model.pth'
    )
    
    training_history = []
    
    for epoch in range(1, 101):
        train_loss = train_epoch(final_model, train_loader, optimizer, criterion)
        val_loss, val_prec, val_rec, val_f1, _, _ = evaluate_model(final_model, val_loader, criterion)

        # [CHANGE 2] Step the scheduler every epoch using val_loss
        scheduler.step(val_loss)  # [CHANGE 2]

        # Log current LR for visibility
        current_lr = optimizer.param_groups[0]['lr']  # [CHANGE 2]
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'lr': current_lr  # [CHANGE 2] also track LR in history
        })
        
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        
        early_stopping(val_loss, final_model)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            break
    
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(MODEL_DIR / 'training_history.csv', index=False)

    # =========================================================
    # PHASE 3: FINE-TUNE PASS AFTER EARLY STOPPING
    # [CHANGE 3] New phase entirely — did not exist before
    #            Reason: After early stopping, we reload the best
    #            checkpoint and do one more short training pass at
    #            10x lower LR. This squeezes out final performance
    #            without the risk of overfitting (tight patience=5).
    #            Think of it as "polishing" the best weights found.
    # =========================================================
    print("\n" + "="*60)
    print("🔧 PHASE 3: Fine-Tune Pass (10x lower LR)")
    print("="*60)

    # [CHANGE 3] Load best weights from main training
    final_model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth'))

    # [CHANGE 3] Reduce LR by 10x for fine-tuning
    finetune_lr = best_config['LR'] / 10
    print(f"   Fine-tune LR: {finetune_lr} (was {best_config['LR']})")

    # [CHANGE 3] Fresh optimizer at lower LR
    ft_optimizer = optim.AdamW(final_model.parameters(), lr=finetune_lr)

    # [CHANGE 3] Tighter early stopping for fine-tune (patience=5 not 10)
    early_stopping.reset(new_patience=5)

    finetune_history = []

    for epoch in range(1, 26):  # [CHANGE 3] max 25 fine-tune epochs
        train_loss = train_epoch(final_model, train_loader, ft_optimizer, criterion)
        val_loss, val_prec, val_rec, val_f1, _, _ = evaluate_model(final_model, val_loader, criterion)

        current_lr = ft_optimizer.param_groups[0]['lr']

        finetune_history.append({
            'ft_epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'lr': current_lr
        })

        print(f"FT Epoch {epoch:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | F1: {val_f1:.4f}")

        # [CHANGE 3] Early stopping still applied — fine-tune stops if no improvement
        early_stopping(val_loss, final_model)
        if early_stopping.early_stop:
            print(f"\n🛑 Fine-tune early stopping at epoch {epoch}")
            break

    # [CHANGE 3] Save fine-tune history separately
    ft_df = pd.DataFrame(finetune_history)
    ft_df.to_csv(MODEL_DIR / 'finetune_history.csv', index=False)
    print(f"📊 Fine-tune history saved.")

    # =========================================================
    # PHASE 4: FINAL EVALUATION
    # (was Phase 3 before — renamed to reflect new phase numbering)
    # =========================================================
    print("\n" + "="*60)
    print("📊 PHASE 4: Final Model Evaluation")
    print("="*60)
    
    # Load best model (may have been updated during fine-tune)
    final_model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth'))
    
    test_loss, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate_model(
        final_model, test_loader, criterion
    )
    
    print(f"\n🎯 TEST SET RESULTS:")
    print(f"   Loss:      {test_loss:.4f}")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall:    {test_rec:.4f}")
    print(f"   F1 Score:  {test_f1:.4f}")
    
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'classes': train_ds.classes,
        'class_to_idx': train_ds.class_to_idx,
        'num_classes': num_classes,
        'input_size': INPUT_SIZE,
        'best_config': best_config,
        'test_metrics': {
            'loss': test_loss,
            'precision': test_prec,
            'recall': test_rec,
            'f1': test_f1
        }
    }, MODEL_DIR / 'final_model_complete.pth')
    
    metadata = {
        'classes': train_ds.classes,
        'class_to_idx': train_ds.class_to_idx,
        'num_classes': num_classes,
        'input_size': INPUT_SIZE,
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
    print("   - best_model.pth            (best weights from main + fine-tune)")
    print("   - final_model_complete.pth  (weights + metadata)")
    print("   - model_metadata.json")
    print("   - tuning_results.csv")
    print("   - training_history.csv")
    print("   - finetune_history.csv      (new)")  # [CHANGE 3]
    print("="*60)

if __name__ == "__main__":
    train_with_tuning()