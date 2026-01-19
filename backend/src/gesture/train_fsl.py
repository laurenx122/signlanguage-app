"""
train_fsl.py - Updated to use preprocessed landmarks
Trains LSTM model on normalized, preprocessed FSL landmarks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models' / 'lstm'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PreprocessedFSLDataset(Dataset):
    """Dataset that loads preprocessed landmarks"""
    def __init__(self, landmarks, labels):
        self.landmarks = landmarks
        self.labels = labels
    
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.landmarks[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )


class LSTMGestureModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=26, dropout=0.3):
        super(LSTMGestureModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        
        out = self.dropout(self.relu(self.batch_norm1(self.fc1(out))))
        out = self.dropout(self.relu(self.batch_norm2(self.fc2(out))))
        return self.fc3(out)


def load_preprocessed_data():
    """Load preprocessed landmarks and labels"""
    print("📂 Loading preprocessed data...")
    
    X = np.load(DATA_DIR / 'fsl_landmarks_X.npy')
    y = np.load(DATA_DIR / 'fsl_labels_y.npy')
    class_to_idx = np.load(DATA_DIR / 'fsl_class_to_idx.npy', allow_pickle=True).item()
    
    print(f"✅ Loaded {len(X)} samples")
    print(f"📊 Feature shape: {X.shape}")
    print(f"📊 Classes: {list(class_to_idx.keys())}")
    
    return X, y, class_to_idx


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for landmarks, labels in tqdm(dataloader, desc='Training'):
        landmarks, labels = landmarks.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        loss.backward()
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
        for landmarks, labels in tqdm(dataloader, desc='Validation'):
            landmarks, labels = landmarks.to(device), labels.to(device)
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(dataloader), 100.*correct/total


def main():
    print("="*60)
    print("🚀 FSL LSTM Training Pipeline (Preprocessed Data)")
    print("="*60)
    print(f"Using device: {DEVICE}")
    
    # Load preprocessed data
    X, y, class_to_idx = load_preprocessed_data()
    
    # Create dataset
    full_dataset = PreprocessedFSLDataset(X, y)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
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
    
    # Get feature dimension and number of classes
    feature_dim = X.shape[1]  # Should be 63 (21 landmarks * 3)
    num_classes = len(class_to_idx)
    
    print(f"\n📊 Training Configuration:")
    print(f"  Classes: {num_classes}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Initialize model
    model = LSTMGestureModel(
        input_size=feature_dim,
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
                'classes': list(class_to_idx.keys()),
                'class_to_idx': class_to_idx,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'feature_dim': feature_dim,
                'normalization_method': 'wrist_relative',
                'wrist_relative': True
            }
            
            model_path = MODEL_DIR / 'best_fsl_lstm_model.pth'
            torch.save(checkpoint, model_path)
            
            print(f"\n✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "="*60)
    print(f"✅ Training complete!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: {MODEL_DIR / 'best_fsl_lstm_model.pth'}")
    print("="*60)


if __name__ == '__main__':
    main()