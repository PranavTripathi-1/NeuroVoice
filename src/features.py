# src/features.py

import numpy as np
import librosa

def extract_features(file_path, max_pad_len=100):
    try:
        X, sample_rate = librosa.load(file_path, sr=None, mono=True)
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)

        # Pad / truncate to make shape (40, 100)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        # Add channel dimension
        mfccs = np.expand_dims(mfccs, axis=-1)  # (40, 100, 1)
        return mfccs

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
