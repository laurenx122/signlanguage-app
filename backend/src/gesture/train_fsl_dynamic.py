"""
train_fsl_dynamic.py
Train LSTM model for dynamic sign language recognition (video sequences)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic'
MODEL_DIR = PROJECT_ROOT / 'models' / 'lstm_dynamic'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 🎯 TESTING MODE
# ============================================
TESTING_MODE = True  # ← Set to False for full training
# ============================================

# Training parameters
if TESTING_MODE:
    print("⚡ TESTING MODE ENABLED - Fast training")
    BATCH_SIZE = 8
    NUM_EPOCHS = 30      # Quick test
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 128    # Smaller model
    NUM_LAYERS = 2       # Fewer layers
    DROPOUT = 0.3
else:
    print("🚀 FULL TRAINING MODE - Best accuracy")
    BATCH_SIZE = 16
    NUM_EPOCHS = 150     # Full training
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 256    # Larger model
    NUM_LAYERS = 3       # More layers
    DROPOUT = 0.4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DynamicSignDataset(Dataset):
    """Dataset for dynamic sign sequences"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )


class DynamicLSTMModel(nn.Module):
    """LSTM model for sequence classification"""
    def __init__(self, input_size=126, hidden_size=256, num_layers=3, 
                 num_classes=10, dropout=0.4):
        super(DynamicLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        out = self.fc1(attended)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for sequences, labels in tqdm(dataloader, desc='Training'):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(dataloader), 100.*correct/total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc='Validation'):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(dataloader), 100.*correct/total


def main():
    mode_text = "⚡ TESTING" if TESTING_MODE else "🚀 FULL TRAINING"
    print("="*60)
    print(f"{mode_text} - FSL Dynamic LSTM Training Pipeline")
    print("="*60)
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("\n📂 Loading preprocessed sequences...")
    X = np.load(DATA_DIR / 'sequences_X.npy')
    y = np.load(DATA_DIR / 'labels_y.npy')
    
    with open(DATA_DIR / 'label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
    
    print(f"✅ Loaded {len(X)} sequences")
    print(f"📊 Sequence shape: {X.shape}")
    print(f"📊 Classes ({len(label_mapping['label_to_idx'])}): {list(label_mapping['label_to_idx'].keys())}")
    
    if TESTING_MODE:
        print(f"\n⚡ TESTING MODE:")
        print(f"   - Quick training ({NUM_EPOCHS} epochs)")
        print(f"   - Smaller model (Hidden: {HIDDEN_SIZE}, Layers: {NUM_LAYERS})")
        print(f"   - Faster results, lower accuracy")
        print(f"   - Set TESTING_MODE = False for full training")
    else:
        print(f"\n🚀 FULL TRAINING MODE:")
        print(f"   - Full training ({NUM_EPOCHS} epochs)")
        print(f"   - Larger model (Hidden: {HIDDEN_SIZE}, Layers: {NUM_LAYERS})")
        print(f"   - Best accuracy, longer training")
    
    # Create dataset
    full_dataset = DynamicSignDataset(X, y)
    
    # Split train/validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    num_classes = len(label_mapping['label_to_idx'])
    model = DynamicLSTMModel(
        input_size=126,  # 2 hands * 21 landmarks * 3 coords
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_acc = 0.0
    
    print(f"\n🏋️ Training Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  LSTM layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_mapping': label_mapping,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'sequence_length': X.shape[1],
                'input_size': 126
            }
            
            model_path = MODEL_DIR / 'best_fsl_dynamic_model.pth'
            torch.save(checkpoint, model_path)
            
            print(f"\n✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print(f"✅ Training complete!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: {MODEL_DIR / 'best_fsl_dynamic_model.pth'}")
    print("="*60)


if __name__ == '__main__':
    main()