import os
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rf_sound_demographics_pipeline.joblib"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"wav"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "replace-with-a-secure-key"

model = joblib.load(MODEL_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features_from_audio(path, n_mfcc=13):
    y, sr = librosa.load(path, sr=None)
    if y.size == 0:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    rms_vals = librosa.feature.rms(S=S)
    rms = float(rms_vals.mean())

    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr) if len(y) > 0 else np.array([0.0])
    dom_freq = float(freqs[np.argmax(fft)]) if fft.size > 0 else 0.0

    feats = {}
    for i, (m, s) in enumerate(zip(mfcc_mean, mfcc_std), start=1):
        feats[f"mfcc{i}_mean"] = float(m)
        feats[f"mfcc{i}_std"] = float(s)

    feats.update({
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr,
        "rms": rms,
        "dom_freq": dom_freq,
        "sr": int(sr),
        "Age": 0,
        "BMI (kg/m2)": 0,
        "Sex": "Unknown",
    })
    return feats


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        if "audio_file" not in request.files:
            error = "No audio file part in the request."
        else:
            file = request.files["audio_file"]
            if file.filename == "":
                error = "Please select a .wav file to upload."
            elif not allowed_file(file.filename):
                error = "Only .wav files are allowed."
            else:
                filename = secure_filename(file.filename)
                save_path = UPLOAD_FOLDER / filename
                file.save(save_path)

                try:
                    features = extract_features_from_audio(str(save_path))
                    if features is None:
                        error = "Unable to process the uploaded audio file."
                    else:
                        df = pd.DataFrame([features])
                        prediction = model.predict(df)[0]
                        result = prediction
                except Exception:
                    error = "Failed to generate prediction from the uploaded file."
                finally:
                    try:
                        save_path.unlink(missing_ok=True)
                    except Exception:
                        pass

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
