# src/train.py

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
DATA_PATH = "data/raw"
MODEL_PATH = "models/speech_disease_model.h5"

# Parameters
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Extract MFCC features from an audio file
def extract_features(file_path, max_pad_len=100):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset
def load_dataset(data_path):
    X, y = [], []
    for emotion in os.listdir(data_path):
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for file in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(emotion)
    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_dataset(DATA_PATH)
print(f"Loaded {len(X)} samples.")

# Reshape for CNN input
X = X[..., np.newaxis]

# Encode labels
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("Training model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
