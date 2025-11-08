# 🎙️ NeuroVoice - AI-Powered Speech Emotion Detection

## 📌 Overview
NeuroVoice is an AI-powered application that detects emotions in speech using deep learning.  
It allows users to **upload** or **record** their voice, then instantly analyzes it to identify emotions like **Happy**, **Sad**, **Neutral**, and **Angry**.

This project uses **Python, TensorFlow, Librosa, and Streamlit** to create a simple yet powerful speech emotion detection system.

---

## 🚀 Features
- 🎤 **Upload or record audio**
- 🤖 **Real-time emotion detection**
- 📊 **Confidence score visualization**
- 📂 **Dataset creation script for training**
- 🧠 **Custom-trained deep learning model**

---

## 🛠 Tech Stack
- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **Audio Processing:** [Librosa](https://librosa.org/)
- **Model Training:** [TensorFlow/Keras](https://www.tensorflow.org/)
- **Language:** Python 3.9+

---

## 📂 Project Structure
NeuroVoice/
│── app/
│ ├── streamlit_app.py # Main Streamlit app
│── datasets/ # Generated training dataset
│── models/
│ ├── emotion_model.h5 # Trained deep learning model
│── notebooks/
│ ├── demo_dataset.py # Script to generate dataset
│── sample_audios/ # Sample audio files
│── uploads/ # Temporary uploaded files
│── README.md # Project documentation

---

## ⚡ Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/PranavTripathi-1/NeuroVoice.git
cd NeuroVoice

🏆 Future Improvements

1. Add more emotion categories (fear, surprise, disgust)
2.Deploy online for public use
3.Support for multi-language emotion detection

