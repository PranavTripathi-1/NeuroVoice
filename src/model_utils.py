# src/model_utils.py
import joblib
import os
from sklearn.preprocessing import LabelEncoder

def save_model_bundle(model, label_encoder, path):
    """
    Saves a dict bundle with 'model' and 'label_encoder' using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'model': model, 'label_encoder': label_encoder}, path)

def load_model_bundle(path):
    return joblib.load(path)
