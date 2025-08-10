import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = "models/emotion_model.h5"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

model = load_emotion_model()

# Emotion labels (update according to your training)
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry"]

# Feature extraction
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

# Predict emotion
def predict_emotion(audio_file):
    features = extract_features(audio_file)
    prediction = model.predict(features)
    emotion_index = np.argmax(prediction)
    return EMOTIONS[emotion_index], prediction[0]

# Streamlit UI
st.set_page_config(page_title="NeuroVoice - Emotion Detection", page_icon="🎤", layout="centered")

st.title("🎙️ NeuroVoice - AI Speech Emotion Detection")
st.write("Upload or record an audio file to detect emotions in speech.")

# Audio Upload
uploaded_audio = st.file_uploader("Upload an audio file (WAV/MP3)", type=["wav", "mp3"])

# Record Audio
st.write("Or record audio directly in your browser:")
audio_recorded = st.audio_input("Record your voice")

audio_path = None

if uploaded_audio is not None:
    audio_path = os.path.join(UPLOAD_DIR, uploaded_audio.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.getbuffer())
    st.audio(audio_path, format="audio/wav")

elif audio_recorded is not None:
    audio_path = os.path.join(UPLOAD_DIR, "recorded.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_recorded.getbuffer())
    st.audio(audio_path, format="audio/wav")

# Predict button
if audio_path and st.button("🔍 Detect Emotion"):
    with st.spinner("Analyzing audio..."):
        emotion, probabilities = predict_emotion(audio_path)
    
    st.success(f"Detected Emotion: **{emotion}**")
    
    # Show probabilities
    fig, ax = plt.subplots()
    ax.bar(EMOTIONS, probabilities, color='skyblue')
    ax.set_ylabel("Confidence")
    ax.set_title("Emotion Prediction Probabilities")
    st.pyplot(fig)

