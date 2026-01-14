import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.interpolate import interp1d

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed_sequences"
SEQ_LEN = 60
FEATURES = 126
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# AUGMENTATION FUNCTIONS
# =========================

def add_noise(seq, std=0.01):
    """Add small Gaussian noise to landmarks"""
    return seq + np.random.normal(0, std, seq.shape)

def scale_sequence(seq, factor_range=(0.9, 1.1)):
    """Randomly scale landmarks to simulate hand closer/farther"""
    factor = np.random.uniform(*factor_range)
    return seq * factor

def time_warp(seq, target_len=SEQ_LEN):
    """Stretch or compress sequence in time using linear interpolation"""
    original_len = seq.shape[0]
    if original_len == target_len:
        return seq
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, target_len)
    seq_new = interp1d(x_old, seq, axis=0)(x_new)
    return seq_new

def mirror_sequence(seq):
    """Mirror left/right hands (swap first 63 features with last 63 features)"""
    mirrored = np.copy(seq)
    mirrored[:, :63], mirrored[:, 63:] = mirrored[:, 63:], mirrored[:, :63]
    return mirrored

def normalize(seq):
    """Scale sequence to [-1,1] per sample for stability"""
    max_val = np.max(seq)
    min_val = np.min(seq)
    if max_val - min_val == 0:
        return seq
    return 2 * (seq - min_val) / (max_val - min_val) - 1

# =========================
# LOAD DATA + AUGMENT
# =========================
X_data = []
y_data = []

label_map = {}
label_index = 0

for label in sorted(os.listdir(DATA_PATH)):
    label_dir = os.path.join(DATA_PATH, label)
    if not os.path.isdir(label_dir):
        continue

    label_map[label] = label_index

    for file in os.listdir(label_dir):
        if not file.endswith(".npy"):
            continue

        seq = np.load(os.path.join(label_dir, file))
        if seq.shape != (SEQ_LEN, FEATURES):
            continue

        # 1️⃣ Normalize original sequence
        seq = normalize(seq)

        # 2️⃣ Append original
        X_data.append(seq)
        y_data.append(label_index)

        # 3️⃣ Augmentations

        # a) Random noise
        X_data.append(add_noise(seq))
        y_data.append(label_index)

        # b) Scaling
        X_data.append(scale_sequence(seq))
        y_data.append(label_index)

        # c) Time warping / sequence stretching
        X_data.append(time_warp(seq))
        y_data.append(label_index)

        # d) Mirroring
        X_data.append(mirror_sequence(seq))
        y_data.append(label_index)

    label_index += 1

# Convert to tensors
X = torch.tensor(np.array(X_data), dtype=torch.float32)
y = torch.tensor(np.array(y_data), dtype=torch.long)

print("✅ Dataset loaded + augmented:", X.shape, y.shape)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODEL
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = LSTMModel(FEATURES, 128, len(label_map)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Acc: {acc:.2f}%")

# =========================
# SAVE
# =========================
os.makedirs("models/lstm", exist_ok=True)
torch.save(model.state_dict(), "models/lstm/lstm_model.pth")

print("🎉 Model saved")
print("Label map:", label_map)
