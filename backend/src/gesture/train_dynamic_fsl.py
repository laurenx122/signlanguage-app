"""
train_dynamic_fsl.py
Trains LSTM model for FSL dynamic sign recognition.

OVERFITTING FIXES APPLIED:
  [FIX 1] Smaller model: hidden 256→128, layers 3→2, conv channels 256→128
  [FIX 2] Removed MultiheadAttention (overkill for ~52 train samples/class)
  [FIX 3] Stronger runtime augmentation: noise, scale jitter, speed warp, dropout on input
  [FIX 4] weight_decay raised from 0 → 0.01 in AdamW
  [FIX 5] Dropout raised to 0.5 baseline in tuning options
  [FIX 6] Phase 3 fine-tune is now gated: only runs if val F1 >= 0.65
  [FIX 7] Tuning epochs reduced back to 20 (30 overfit each candidate config)
  [FIX 8] Label smoothing added to CrossEntropyLoss (0.1) — punishes overconfident predictions
  [FIX 9] Input feature dropout layer added before LSTM
  [FIX 10] Gap monitoring: prints train-val gap each epoch so you can spot overfit early
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import json
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'fsl_dynamic_sequences'
MODEL_DIR = PROJECT_ROOT / 'models' / 'lstm_dynamic_final'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 252   # 126 position + 126 velocity
SEQ_LEN    = 30

print(f"🖥️  Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=0.001, path='checkpoint.pt'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False
        self.path       = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self._save(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model):
        torch.save(model.state_dict(), self.path)

    def reset(self, new_patience=None):
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False
        if new_patience is not None:
            self.patience = new_patience


# ---------------------------------------------------------------------------
# [FIX 1 + 2] Smaller model — no MultiheadAttention
# ---------------------------------------------------------------------------
class ImprovedLSTMModel(nn.Module):
    """
    Compact BiLSTM with Conv1D preprocessing.
    Removed MultiheadAttention — it was adding ~500K parameters for a dataset
    with only ~52 training samples per class, which is a recipe for overfitting.
    Mean-pooling over LSTM outputs is sufficient and far more regularized.
    """

    def __init__(
        self,
        input_size  = INPUT_SIZE,
        hidden_size = 128,       # [FIX 1] was 256
        num_layers  = 2,         # [FIX 1] was 3
        num_classes = 44,
        dropout     = 0.5,       # [FIX 5] higher baseline
    ):
        super().__init__()

        # [FIX 9] Input dropout — randomly zeros entire feature channels each step.
        # Forces the model to not rely on any single landmark dimension.
        self.input_dropout = nn.Dropout(p=0.1)

        # Conv1D for local feature extraction
        self.conv1   = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)  # [FIX 1] was 256
        self.bn_conv = nn.BatchNorm1d(128)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            128, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # [FIX 2] Simple mean pooling — no MultiheadAttention
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),   # [FIX 1] was hidden*2 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # [FIX 9] Apply input dropout
        x = self.input_dropout(x)

        # Conv preprocessing: (B, T, F) → (B, F, T) → conv → (B, T, 128)
        x = x.transpose(1, 2)
        x = torch.relu(self.bn_conv(self.conv1(x)))
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # [FIX 2] Mean pool over time — simpler, less overfit than attention
        pooled = torch.mean(lstm_out, dim=1)

        return self.fc(pooled)


# ---------------------------------------------------------------------------
# [FIX 3] Dataset with stronger runtime augmentation
# ---------------------------------------------------------------------------
class FSLSequenceDataset(Dataset):
    """
    Load preprocessed .npy sequences.
    Training split gets runtime augmentation (noise, scale, speed warp)
    applied randomly each epoch — the model sees a different version of
    each sample every time, making static memorization much harder.
    """

    def __init__(self, split='train'):
        self.split   = split
        self.samples = []
        self.labels  = []

        split_dir = DATA_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.classes = sorted(
            [d.name for d in split_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls_name in self.classes:
            for npy_file in (split_dir / cls_name).glob("*.npy"):
                self.samples.append(npy_file)
                self.labels.append(self.class_to_idx[cls_name])

        print(f"📂 Loaded {len(self.samples)} samples from '{split}' split")

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    # [FIX 3] Augmentation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _temporal_shift(data, max_shift=5):
        """Shift the sequence forward or backward in time."""
        shift    = np.random.randint(-max_shift, max_shift + 1)
        feat_dim = data.shape[1]
        if shift > 0:
            data = np.concatenate(
                [np.zeros((shift, feat_dim), dtype=np.float32), data[:-shift]], axis=0
            )
        elif shift < 0:
            data = np.concatenate(
                [data[-shift:], np.zeros((-shift, feat_dim), dtype=np.float32)], axis=0
            )
        return data

    @staticmethod
    def _gaussian_noise(data, sigma=0.01):
        """Add small Gaussian noise — simulates sensor noise and hand position variation."""
        noise = np.random.normal(0, sigma, data.shape).astype(np.float32)
        return data + noise

    @staticmethod
    def _scale_jitter(data, low=0.85, high=1.15):
        """
        Random global scale — simulates different hand sizes and
        distances from the camera.
        Only scale position channels (first 126), not velocity channels,
        because velocity is already relative.
        """
        scale = np.random.uniform(low, high)
        augmented = data.copy()
        augmented[:, :126] = augmented[:, :126] * scale
        return augmented

    @staticmethod
    def _speed_warp(data, seq_len=SEQ_LEN, warp_range=(0.8, 1.25)):
        """
        Randomly speed up or slow down the gesture by resampling frames.
        This is the most powerful augmentation for temporal sequences —
        it prevents the model from keying on exact timing patterns.
        """
        factor   = np.random.uniform(*warp_range)
        new_len  = max(int(seq_len * factor), seq_len // 2)  # floor at half length
        indices  = np.linspace(0, len(data) - 1, new_len).astype(int)
        warped   = data[indices]
        # Resample back to SEQ_LEN
        back_idx = np.linspace(0, len(warped) - 1, seq_len).astype(int)
        return warped[back_idx]

    def __getitem__(self, idx):
        data = np.load(self.samples[idx]).astype(np.float32)

        if self.split == 'train':
            # Apply augmentations independently with their own probabilities.
            # Each is applied stochastically so every epoch produces a
            # different composite augmentation.

            # Temporal shift — always applied (was already in original code)
            data = self._temporal_shift(data, max_shift=5)

            # [FIX 3a] Gaussian noise — 80% chance
            if np.random.rand() < 0.80:
                data = self._gaussian_noise(data, sigma=0.012)

            # [FIX 3b] Scale jitter — 70% chance
            if np.random.rand() < 0.70:
                data = self._scale_jitter(data, low=0.85, high=1.15)

            # [FIX 3c] Speed warp — 60% chance (most impactful, applied less often
            # so training doesn't become too noisy early on)
            if np.random.rand() < 0.60:
                data = self._speed_warp(data, seq_len=SEQ_LEN, warp_range=(0.8, 1.25))

        return (
            torch.from_numpy(data).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_model(model, dataloader, criterion):
    """Evaluate model — returns loss, precision, recall, F1, preds, labels."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss   = criterion(output, target)

            total_loss += loss.item()
            all_preds.extend(output.argmax(1).cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    return avg_loss, precision, recall, f1, all_preds, all_labels


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch — returns average training loss."""
    model.train()
    total_loss = 0.0

    for data, target in dataloader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def get_train_accuracy(model, dataloader):
    """Quick train accuracy check for gap monitoring."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            preds   = model(data).argmax(1)
            correct += (preds == target).sum().item()
            total   += target.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def train_with_tuning():

    print("=" * 60)
    print("🎯 FSL Dynamic Sign Language — LSTM Training Pipeline")
    print("=" * 60)

    # --- Load datasets ---
    train_ds = FSLSequenceDataset('train')
    val_ds   = FSLSequenceDataset('val')
    test_ds  = FSLSequenceDataset('test')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

    num_classes = len(train_ds.classes)
    print(f"\n📊 Dataset: {num_classes} classes")
    print(f"   Train : {len(train_ds)} samples  ({len(train_ds)//num_classes} per class)")
    print(f"   Val   : {len(val_ds)} samples")
    print(f"   Test  : {len(test_ds)} samples")

    # =========================================================
    # PHASE 1: HYPERPARAMETER TUNING
    # [FIX 7] Tuning epochs reduced to 20 (was bumped to 30).
    #         With only ~52 samples/class, 30 tuning epochs cause each
    #         candidate to overfit — the "best" config was selected based
    #         on a memorized val set, not genuine generalization.
    # [FIX 4] weight_decay=0.01 added to AdamW in all phases.
    # [FIX 5] Dropout options now 0.4/0.6 (was 0.3/0.5).
    # =========================================================
    print("\n" + "=" * 60)
    print("🔎 PHASE 1: Hyperparameter Tuning")
    print("=" * 60)

    lr_options      = [0.001, 0.0005]
    dropout_options = [0.4, 0.6]          # [FIX 5]
    TUNING_EPOCHS   = 20                  # [FIX 7]
    tuning_results  = []

    for lr in lr_options:
        for dropout in dropout_options:
            print(f"\n🧪 Testing: LR={lr}, Dropout={dropout}")

            model = ImprovedLSTMModel(
                input_size  = INPUT_SIZE,
                num_classes = num_classes,
                dropout     = dropout,
            ).to(DEVICE)

            optimizer = optim.AdamW(
                model.parameters(),
                lr           = lr,
                weight_decay = 0.01,      # [FIX 4]
            )

            # [FIX 8] Label smoothing — penalizes overconfident softmax outputs.
            # Without this, the model can drive one logit to +∞ while driving
            # all others to −∞, which perfectly fits training labels but
            # collapses generalization. smoothing=0.1 keeps predictions humble.
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # [FIX 8]

            for epoch in range(1, TUNING_EPOCHS + 1):
                train_loss = train_epoch(model, train_loader, optimizer, criterion)

                if epoch % 10 == 0:
                    val_loss, _, _, val_f1, _, _ = evaluate_model(
                        model, val_loader, criterion
                    )
                    print(
                        f"   Epoch {epoch:2d} | Train: {train_loss:.4f} "
                        f"| Val: {val_loss:.4f} | F1: {val_f1:.4f}"
                    )

            val_loss, val_prec, val_rec, val_f1, _, _ = evaluate_model(
                model, val_loader, criterion
            )
            tuning_results.append({
                "LR": lr, "Dropout": dropout,
                "Val_Loss": val_loss, "Val_Precision": val_prec,
                "Val_Recall": val_rec, "Val_F1": val_f1,
            })
            print(f"   ✅ Final Val F1: {val_f1:.4f}")

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(MODEL_DIR / 'tuning_results.csv', index=False)
    print(f"\n📊 Tuning results saved: {MODEL_DIR / 'tuning_results.csv'}")

    best_config = max(tuning_results, key=lambda x: x['Val_F1'])
    print("\n" + "=" * 60)
    print("🏆 BEST CONFIGURATION:")
    print(f"   LR:      {best_config['LR']}")
    print(f"   Dropout: {best_config['Dropout']}")
    print(f"   Val F1:  {best_config['Val_F1']:.4f}")
    print("=" * 60)

    # =========================================================
    # PHASE 2: MAIN TRAINING
    # [FIX 10] Gap monitoring added — prints (train acc - val F1)
    #          each epoch so you can see overfit developing in real time.
    #          A growing gap (>0.2) is the signal to stop or increase dropout.
    # =========================================================
    print("\n" + "=" * 60)
    print("🚀 PHASE 2: Main Training with Best Config + LR Scheduler")
    print("=" * 60)

    final_model = ImprovedLSTMModel(
        input_size  = INPUT_SIZE,
        num_classes = num_classes,
        dropout     = best_config['Dropout'],
    ).to(DEVICE)

    optimizer = optim.AdamW(
        final_model.parameters(),
        lr           = best_config['LR'],
        weight_decay = 0.01,              # [FIX 4]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # [FIX 8]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'min',
        factor   = 0.5,
        patience = 5,
        min_lr   = 1e-6,
    )

    early_stopping = EarlyStopping(
        patience = 12,
        path     = MODEL_DIR / 'best_model.pth',
    )

    training_history = []

    for epoch in range(1, 101):
        train_loss = train_epoch(final_model, train_loader, optimizer, criterion)
        val_loss, val_prec, val_rec, val_f1, _, _ = evaluate_model(
            final_model, val_loader, criterion
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # [FIX 10] Overfitting gap monitor
        train_acc = get_train_accuracy(final_model, train_loader)
        gap       = train_acc - val_f1
        gap_warn  = "  ⚠️  OVERFIT" if gap > 0.25 else ""

        training_history.append({
            'epoch'     : epoch,
            'train_loss': train_loss,
            'train_acc' : train_acc,
            'val_loss'  : val_loss,
            'val_f1'    : val_f1,
            'gap'       : gap,
            'lr'        : current_lr,
        })

        print(
            f"Epoch {epoch:3d} | Train: {train_loss:.4f} (acc={train_acc:.3f}) "
            f"| Val: {val_loss:.4f} (F1={val_f1:.4f}) "
            f"| Gap: {gap:+.3f} | LR: {current_lr:.6f}{gap_warn}"
        )

        early_stopping(val_loss, final_model)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            break

    pd.DataFrame(training_history).to_csv(
        MODEL_DIR / 'training_history.csv', index=False
    )
    print(f"📊 Training history saved.")

    # =========================================================
    # PHASE 3: FINE-TUNE PASS (GATED)
    # [FIX 6] Only runs if val F1 >= 0.65 — if the model is already
    #         overfit (val F1 low despite train acc high), fine-tuning
    #         at a lower LR just deepens the memorization.
    #         The gate ensures we only polish a model that has actually
    #         learned something generalizable.
    # =========================================================
    print("\n" + "=" * 60)
    print("🔧 PHASE 3: Fine-Tune Pass (gated)")
    print("=" * 60)

    # Reload best weights and check val F1 before committing to fine-tune
    final_model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth'))
    _, _, _, gate_f1, _, _ = evaluate_model(final_model, val_loader, criterion)
    print(f"   Best checkpoint val F1: {gate_f1:.4f}")

    # [FIX 6] Gate check
    FINE_TUNE_GATE = 0.65
    if gate_f1 < FINE_TUNE_GATE:
        print(
            f"\n⚠️  Val F1 ({gate_f1:.4f}) < gate threshold ({FINE_TUNE_GATE}).\n"
            f"   Skipping fine-tune — the model is likely underfit or overfit.\n"
            f"   Recommendation: collect more training videos per class."
        )
        finetune_history = []
    else:
        finetune_lr = best_config['LR'] / 10
        print(f"   ✅ Gate passed. Fine-tune LR: {finetune_lr}")

        ft_optimizer = optim.AdamW(
            final_model.parameters(),
            lr           = finetune_lr,
            weight_decay = 0.01,          # [FIX 4]
        )
        early_stopping.reset(new_patience=5)
        finetune_history = []

        for epoch in range(1, 26):
            train_loss = train_epoch(final_model, train_loader, ft_optimizer, criterion)
            val_loss, _, _, val_f1, _, _ = evaluate_model(
                final_model, val_loader, criterion
            )

            train_acc = get_train_accuracy(final_model, train_loader)
            gap       = train_acc - val_f1

            finetune_history.append({
                'ft_epoch'  : epoch,
                'train_loss': train_loss,
                'val_loss'  : val_loss,
                'val_f1'    : val_f1,
                'gap'       : gap,
            })
            print(
                f"FT Epoch {epoch:2d} | Train: {train_loss:.4f} "
                f"| Val: {val_loss:.4f} (F1={val_f1:.4f}) | Gap: {gap:+.3f}"
            )

            early_stopping(val_loss, final_model)
            if early_stopping.early_stop:
                print(f"\n🛑 Fine-tune early stopping at epoch {epoch}")
                break

    if finetune_history:
        pd.DataFrame(finetune_history).to_csv(
            MODEL_DIR / 'finetune_history.csv', index=False
        )
        print("📊 Fine-tune history saved.")

    # =========================================================
    # PHASE 4: FINAL EVALUATION
    # =========================================================
    print("\n" + "=" * 60)
    print("📊 PHASE 4: Final Model Evaluation")
    print("=" * 60)

    final_model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth'))

    test_loss, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate_model(
        final_model, test_loader, criterion
    )

    print(f"\n🎯 TEST SET RESULTS:")
    print(f"   Loss:      {test_loss:.4f}")
    print(f"   Precision: {test_prec:.4f}")
    print(f"   Recall:    {test_rec:.4f}")
    print(f"   F1 Score:  {test_f1:.4f}")

    # --- Save full checkpoint ---
    torch.save(
        {
            'model_state_dict': final_model.state_dict(),
            'classes'         : train_ds.classes,
            'class_to_idx'    : train_ds.class_to_idx,
            'num_classes'     : num_classes,
            'input_size'      : INPUT_SIZE,
            'best_config'     : best_config,
            'test_metrics'    : {
                'loss'     : test_loss,
                'precision': test_prec,
                'recall'   : test_rec,
                'f1'       : test_f1,
            },
        },
        MODEL_DIR / 'final_model_complete.pth',
    )

    metadata = {
        'classes'            : train_ds.classes,
        'class_to_idx'       : train_ds.class_to_idx,
        'num_classes'        : num_classes,
        'input_size'         : INPUT_SIZE,
        'best_hyperparameters': best_config,
        'test_performance'   : {
            'loss'     : float(test_loss),
            'precision': float(test_prec),
            'recall'   : float(test_rec),
            'f1'       : float(test_f1),
        },
        'overfitting_fixes'  : [
            'Smaller model: hidden 256→128, layers 3→2',
            'Removed MultiheadAttention — mean pooling instead',
            'Runtime augmentation: noise, scale jitter, speed warp',
            'weight_decay=0.01 in AdamW',
            'Dropout 0.4/0.6 options (was 0.3/0.5)',
            'Phase 3 gated by val F1 >= 0.65',
            'Label smoothing 0.1',
            'Input feature dropout 0.1',
            'Gap monitoring every epoch',
        ],
    }

    with open(MODEL_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print(f"📁 Models saved to: {MODEL_DIR}")
    print("   - best_model.pth")
    print("   - final_model_complete.pth")
    print("   - model_metadata.json")
    print("   - tuning_results.csv")
    print("   - training_history.csv")
    if finetune_history:
        print("   - finetune_history.csv")
    print("=" * 60)


if __name__ == "__main__":
    train_with_tuning()