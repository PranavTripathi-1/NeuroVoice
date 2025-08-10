import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from src.features import extract_wav2vec2
from pathlib import Path

DATA_DIR = Path("data/processed")
labels = []
features = []

for label_dir in DATA_DIR.iterdir():
    if label_dir.is_dir():
        for file in label_dir.glob("*.wav"):
            emb = extract_wav2vec2(str(file))
            features.append(emb)
            labels.append(label_dir.name)

features = np.array(features)
le = LabelEncoder()
y = le.fit_transform(labels)

model = LogisticRegression(max_iter=200)
model.fit(features, y)

y_pred = model.predict(features)
print(classification_report(y, y_pred, target_names=le.classes_))

# Save model + label encoder
os.makedirs("models", exist_ok=True)
with open("models/speech_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model trained and saved to models/speech_disease_model.pkl")

