# src/preprocess.py
"""
Simple preprocessing utilities:
- convert audio to mono 16k wav
- trim silence (optional)
- build manifest CSV listing processed files and labels
"""
import argparse
from pathlib import Path
from pydub import AudioSegment
import librosa
import soundfile as sf
import csv

def convert_to_wav(in_path: Path, out_path: Path, sr: int = 16000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path

def trim_silence(wav_path: Path, top_db: int = 20):
    y, sr = librosa.load(str(wav_path), sr=None)
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    sf.write(str(wav_path), yt, sr)

def build_manifest(processed_root: Path, manifest_path: Path):
    rows = []
    for label_dir in processed_root.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name.lower()
        for wav in label_dir.glob("**/*.wav"):
            subject = wav.stem.split("_")[0]
            rows.append({'filepath': str(wav.resolve()), 'label': label, 'subject': subject})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'label', 'subject'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Manifest written to {manifest_path} ({len(rows)} entries)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="raw audio folder (subfolders=labels)")
    parser.add_argument("--out", required=True, help="processed output root")
    parser.add_argument("--manifest", required=True, help="manifest CSV path")
    parser.add_argument("--sr", default=16000, type=int)
    parser.add_argument("--trim", action="store_true")
    args = parser.parse_args()

    raw = Path(args.root)
    processed = Path(args.out)
    for label_dir in raw.iterdir():
        if not label_dir.is_dir(): continue
        for f in label_dir.glob("**/*"):
            if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
                outp = processed / label_dir.name / (f.stem + ".wav")
                print("Converting:", f, "->", outp)
                convert_to_wav(f, outp, sr=args.sr)
                if args.trim:
                    trim_silence(outp)
    build_manifest(processed, Path(args.manifest))
