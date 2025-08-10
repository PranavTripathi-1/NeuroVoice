import os
import librosa
import soundfile as sf
import numpy as np
import pyttsx3

# === CONFIG ===
EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"]
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SAMPLE_TEXT = {
    "happy": "I am feeling amazing today!",
    "sad": "I am not feeling well today.",
    "angry": "Why did you do this to me?",
    "neutral": "The sky is blue and the grass is green.",
    "fear": "I am scared to go outside.",
    "surprise": "Wow, I did not expect that!"
}
SAMPLES_PER_EMOTION = 3  # Number of audio samples to generate

# === Step 1: Create raw sample audios using TTS ===
def generate_tts_samples():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("rate", 150)

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(RAW_DIR, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        for i in range(SAMPLES_PER_EMOTION):
            file_path = os.path.join(emotion_dir, f"{emotion}_{i}.wav")
            engine.save_to_file(SAMPLE_TEXT[emotion], file_path)
        engine.runAndWait()
    print("✅ TTS audio files generated in:", RAW_DIR)

# === Step 2: Augment data and save to processed folder ===
def augment_and_save():
    for emotion in EMOTIONS:
        raw_path = os.path.join(RAW_DIR, emotion)
        processed_path = os.path.join(PROCESSED_DIR, emotion)
        os.makedirs(processed_path, exist_ok=True)

        for file in os.listdir(raw_path):
            if file.endswith(".wav"):
                file_path = os.path.join(raw_path, file)
                y, sr = librosa.load(file_path, sr=None)

                # Original
                sf.write(os.path.join(processed_path, file), y, sr)

                # Pitch shifted
                pitch = librosa.effects.pitch_shift(y, sr, n_steps=2)
                sf.write(os.path.join(processed_path, f"{file[:-4]}_pitch.wav"), pitch, sr)

                # Time stretched
                stretch = librosa.effects.time_stretch(y, rate=0.85)
                sf.write(os.path.join(processed_path, f"{file[:-4]}_slow.wav"), stretch, sr)

                # Add noise
                noise = y + 0.005 * np.random.randn(len(y))
                sf.write(os.path.join(processed_path, f"{file[:-4]}_noise.wav"), noise, sr)

    print("🎯 Dataset generation & augmentation complete!")

if __name__ == "__main__":
    generate_tts_samples()
    augment_and_save()
