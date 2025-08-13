# src/features.py

import numpy as np
import librosa

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    """
    Extract audio features (MFCC, Chroma, Mel) from an audio file.
    
    Parameters:
        file_path (str): Path to the audio file.
        mfcc (bool): Whether to extract MFCC features.
        chroma (bool): Whether to extract Chroma features.
        mel (bool): Whether to extract Mel Spectrogram features.
    
    Returns:
        np.ndarray: Combined feature vector.
    """
    try:
        X, sample_rate = librosa.load(file_path, sr=None, mono=True)
        features = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            features = np.hstack((features, mfccs))

        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            features = np.hstack((features, chroma_feat))

        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            features = np.hstack((features, mel_feat))

        return features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
