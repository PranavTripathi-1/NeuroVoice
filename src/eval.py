# src/eval.py
"""
Evaluate saved model on manifest (computes classification report & confusion matrix)
Usage:
  python src/eval.py --manifest data/manifest.csv --model models/speech_disease_model.pkl --cache data/processed/features_cache.npz
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.model_utils import load_model_bundle
from src.features import extract_wav2vec2
import joblib

def main(manifest_path, model_path, cache=None):
    manifest = pd.read_csv(manifest_path)
    bundle = load_model_bundle(model_path)
    model = bundle['model']
    le = bundle['label_encoder']

    # Load or compute features
    X = []
    for fp in manifest['filepath'].tolist():
        try:
            X.append(extract_wav2vec2(fp))
        except Exception as e:
            print("Failed feature for", fp, e)
            X.append(np.zeros(768))
    X = np.vstack(X)
    y_true = le.transform(manifest['label'].astype(str).values)

    y_pred = model.predict(X)
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.manifest, args.model)
