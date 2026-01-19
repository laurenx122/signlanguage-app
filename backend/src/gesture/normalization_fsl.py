# src/gesture/normalization_fsl.py
import numpy as np
import json

class LandmarkNormalizer:
    def __init__(self, method='zscore', wrist_relative=False):
        self.method = method
        self.wrist_relative = wrist_relative
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
    
    def fit(self, landmarks_array):
        """Fit normalizer on training data"""
        if self.method == 'zscore':
            self.mean = np.mean(landmarks_array, axis=0)
            self.std = np.std(landmarks_array, axis=0)
            # Prevent division by zero
            self.std = np.where(self.std == 0, 1, self.std)
        elif self.method == 'minmax':
            self.min = np.min(landmarks_array, axis=0)
            self.max = np.max(landmarks_array, axis=0)
            # Prevent division by zero
            range_val = self.max - self.min
            self.max = np.where(range_val == 0, self.min + 1, self.max)
    
    def transform(self, landmarks):
        """Transform landmarks"""
        landmarks = landmarks.copy()
        
        if self.wrist_relative:
            # Make coordinates relative to wrist (landmark 0)
            wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
            for i in range(0, len(landmarks), 3):
                landmarks[i] -= wrist_x
                landmarks[i+1] -= wrist_y
                landmarks[i+2] -= wrist_z
        
        if self.method == 'zscore' and self.mean is not None:
            landmarks = (landmarks - self.mean) / self.std
        elif self.method == 'minmax' and self.min is not None:
            landmarks = (landmarks - self.min) / (self.max - self.min)
        
        return landmarks
    
    def save(self, filepath):
        """Save normalizer parameters"""
        params = {
            'method': self.method,
            'wrist_relative': self.wrist_relative,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'min': self.min.tolist() if self.min is not None else None,
            'max': self.max.tolist() if self.max is not None else None,
        }
        with open(filepath, 'w') as f:
            json.dump(params, f)
        print(f"✅ Normalizer saved to {filepath}")
    
    def load(self, filepath):
        """Load normalizer parameters"""
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.method = params.get('method', 'zscore')
        self.wrist_relative = params.get('wrist_relative', False)
        self.mean = np.array(params['mean']) if params.get('mean') else None
        self.std = np.array(params['std']) if params.get('std') else None
        self.min = np.array(params['min']) if params.get('min') else None
        self.max = np.array(params['max']) if params.get('max') else None
        print(f"✅ Loaded normalizer: method={self.method}, wrist_relative={self.wrist_relative}")

def create_normalizer(method='zscore', wrist_relative=False):
    """Factory function to create normalizer"""
    return LandmarkNormalizer(method=method, wrist_relative=wrist_relative)