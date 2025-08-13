import sys
import os
import random
import numpy as np
import librosa
import streamlit as st
from tensorflow.keras.models import load_model

# ======================
# FIX PATH ISSUE
# ======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.features import extract_features  # your existing function

# ======================
# LOAD MODEL
# ======================
MODEL_PATH = os.path.join(BASE_DIR, "models", "speech_disease_model.h5")
model = load_model(MODEL_PATH)

# ======================
# DATA PATHS
# ======================
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EMOTIONS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

# ======================
# STREAMLIT APP
# ======================
st.title("🎤 NeuroVoice - Emotion Detection from Speech")
st.markdown("Upload your voice or test with a sample audio from the dataset.")

# ----------------------
# Prediction Function
# ----------------------
def predict_emotion(file_path):
    try:
        features = extract_features(file_path)
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        return EMOTIONS[predicted_index]
    except Exception as e:
        return f"Error processing file: {str(e)}"

# ----------------------
# File Upload Section
# ----------------------
st.header("📂 Upload Your Own Voice")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    temp_path = os.path.join(BASE_DIR, "temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_path)
    if st.button("Predict Uploaded Audio"):
        emotion = predict_emotion(temp_path)
        st.success(f"Predicted Emotion: **{emotion}**")

# ----------------------
# Sample Data Section
# ----------------------
st.header("🎯 Test with Sample Data")
if st.button("Pick Random Sample"):
    # Choose random emotion folder
    emotion = random.choice(EMOTIONS)
    emotion_path = os.path.join(PROCESSED_DIR, emotion)

    # Pick random file from that folder
    sample_file = random.choice(os.listdir(emotion_path))
    sample_path = os.path.join(emotion_path, sample_file)

    st.audio(sample_path)
    predicted = predict_emotion(sample_path)
    st.info(f"Sample Emotion (Folder): **{emotion}**")
    st.success(f"Predicted Emotion: **{predicted}**")
