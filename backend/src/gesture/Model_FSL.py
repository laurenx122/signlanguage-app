"""
Model_FSL.py
Trains Random Forest classifier on normalized FSL landmarks
- Loads preprocessed features
- Trains Random Forest model
- Evaluates on test set
- Saves trained model
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models' / 'random_forest'

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Random Forest parameters
N_ESTIMATORS = 200
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
RANDOM_STATE = 42


def load_data():
    """Load preprocessed landmarks and labels"""
    print("📂 Loading preprocessed data...")
    
    X = np.load(DATA_DIR / 'fsl_landmarks_X.npy')
    y = np.load(DATA_DIR / 'fsl_labels_y.npy')
    class_to_idx = np.load(DATA_DIR / 'fsl_class_to_idx.npy', allow_pickle=True).item()
    
    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"📊 Classes: {list(class_to_idx.keys())}")
    
    return X, y, class_to_idx


def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    print("\n🌲 Training Random Forest model...")
    print(f"Parameters:")
    print(f"  - n_estimators: {N_ESTIMATORS}")
    print(f"  - max_depth: {MAX_DEPTH}")
    print(f"  - min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"  - min_samples_leaf: {MIN_SAMPLES_LEAF}")
    
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Training complete!")
    return model


def evaluate_model(model, X_test, y_test, class_to_idx):
    """Evaluate model on test set"""
    print("\n📊 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")
    
    # Create inverse mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix (using matplotlib only)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - FSL Recognition')
    plt.colorbar()
    
    # Add tick marks
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = MODEL_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    print(f"\n💾 Confusion matrix saved to: {cm_path}")
    plt.close()
    
    return accuracy


def save_model(model, class_to_idx, accuracy):
    """Save trained model and metadata"""
    model_path = MODEL_DIR / 'fsl_rf_model.pkl'
    metadata_path = MODEL_DIR / 'model_metadata.pkl'
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'class_to_idx': class_to_idx,
        'idx_to_class': {v: k for k, v in class_to_idx.items()},
        'accuracy': accuracy,
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'feature_dim': 63  # 21 landmarks * 3 coordinates
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("🚀 FSL Random Forest Training Pipeline")
    print("="*60)
    
    # Load data
    X, y, class_to_idx = load_data()
    
    # Split data
    print("\n📊 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test, class_to_idx)
    
    # Save model
    save_model(model, class_to_idx, accuracy)
    
    print("\n" + "="*60)
    print("✅ Training pipeline complete!")
    print(f"🎯 Final Test Accuracy: {accuracy * 100:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()