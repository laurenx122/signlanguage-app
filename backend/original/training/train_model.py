import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import os

def create_lstm_model(num_classes, sequence_length=30, num_features=126):
    """
    Create LSTM model for sign language recognition
    num_features = 21 hand landmarks * 3 (x,y,z) * 2 hands = 126
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data():
    """
    Prepare your training data here
    You'll need to collect sign language data first
    """
    # This is where you'll load your collected data
    # X_train, y_train = load_your_collected_data()
    pass

if __name__ == "__main__":
    # Example: Create model for 10 sign language gestures
    model = create_lstm_model(num_classes=10)
    model.summary()
    
    # Save model architecture
    model.save('backend/models/sign_language_model.h5')
    print("Model created and saved!")