# src/features.py
"""
Wav2Vec2 embedding extractor + fallback MFCC extractor.
Lazy-loads the HuggingFace processor + model so imports are light until used.
"""

import numpy as np
import librosa

# Lazy imports for transformers & torch (only when wav2vec is used)
_wav2vec_processor = None
_wav2vec_model = None

def _ensure_wav2vec():
    global _wav2vec_processor, _wav2vec_model
    if _wav2vec_processor is None or _wav2vec_model is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import torch
        _wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        _wav2vec_model.eval()
        # No .to(device) here — leave to user/GPU if desired
    return _wav2vec_processor, _wav2vec_model

def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr)
    return y

def extract_mfcc(path, sr=16000, n_mfcc=40):
    """Lightweight MFCC aggregate features (mean over time)."""
    y = load_audio(path, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])  # 2*n_mfcc dims
    return feat

def extract_wav2vec2(path, target_sr=16000):
    """
    Returns a 1D numpy array embedding (mean-pooled over time) from Wav2Vec2.
    """
    processor, model = _ensure_wav2vec()
    import torch
    y = load_audio(path, sr=target_sr)
    inputs = processor(y, sampling_rate=target_sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state shape: (1, T, C) -> mean pool on T -> (1, C)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb
