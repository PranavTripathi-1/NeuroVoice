import os
import numpy as np
import librosa
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.utils import to_categorical

# Path to dataset
DATA_PATH = "dataset"  # change to your dataset folder name

# Supported emotions
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract MFCC features
def extract_features(file_name):
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    except Exception as e:
        print(f"Error extracting {file_name}: {e}")
        return None
    return mfccs

# Load dataset
features, labels = [], []
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                # Extract emotion code from filename (RAVDESS format)
                emotion_code = file.split("-")[2]
                labels.append(emotions.get(emotion_code, "unknown"))

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add channel dimension for Conv1D
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Build Model
model = Sequential()
model.add(Conv1D(256, 5, padding='same', activation='relu', input_shape=(40, 1)))
model.add(MaxPooling1D(pool_size=8))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Save as .h5
model.save("models/speech_disease_model.h5")
print("✅ Model saved to models/speech_disease_model.h5")