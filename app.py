import os
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import wave
import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import model_utils

BASE_DIR = Path(__file__).resolve().parent
# allow overriding model path via env for deployment flexibility
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(BASE_DIR / "rf_sound_demographics_pipeline.joblib")))
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"wav"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))
# read secret from env with safe default for local dev
app.secret_key = os.environ.get("SECRET_KEY", "replace-with-a-secure-key")

# load model (path can be overridden via MODEL_PATH env var)
model = None
model_ready = False
model_load_error = None
numeric_cols = []
cat_cols = []
model_class_labels = []
model_expected_feature_count = 0
model_expected_preview = []

try:
    model = joblib.load(MODEL_PATH)
    preproc = model.named_steps["preproc"]
    numeric_cols = preproc.transformers[0][2]
    cat_cols = preproc.transformers[1][2]
    model_ready = True
    model_class_labels = model.named_steps["rf"].classes_.tolist()
    model_expected_feature_count = len(numeric_cols) + len(cat_cols)
    model_expected_preview = numeric_cols[:6] + cat_cols
except Exception as exc:
    model_load_error = str(exc)


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


def validate_wav(path, min_duration=0.5, min_size=200):
    """Quickly validate that the file is a WAV and has a minimum duration and size."""
    try:
        path = str(path)
        # check file size first
        size = Path(path).stat().st_size
        if size < min_size:
            return False, "Uploaded file is too small to be a valid audio file."
        try:
            with wave.open(path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate) if rate else 0.0
        except wave.Error:
            # fallback: some WAVs (e.g., non-PCM) may not be readable by wave; use librosa to get duration
            try:
                y, sr = librosa.load(str(path), sr=None)
                duration = len(y) / float(sr) if sr else 0.0
            except Exception:
                return False, "Uploaded file is not a valid WAV file."

        if duration < min_duration:
            return False, f"Uploaded audio is too short ({duration:.2f}s). Minimum is {min_duration}s."
    except wave.Error:
        return False, "Uploaded file is not a valid WAV file."
    except Exception as exc:
        return False, f"Cannot validate audio file: {exc}"
    return True, None


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    audio_filename = None
    audio_url = None
    view_requested = False

    if request.method == "POST":
        try:
            if "audio_file" in request.files and request.files["audio_file"].filename != "":
                file = request.files["audio_file"]
                if not allowed_file(file.filename):
                    error = "Only .wav files are allowed."
                else:
                    filename = secure_filename(file.filename)
                    save_path = UPLOAD_FOLDER / filename
                    file.save(save_path)

                    valid, msg = validate_wav(save_path)
                    if not valid:
                        error = msg or "Uploaded file failed validation."
                        save_path.unlink(missing_ok=True)
                    else:
                        audio_filename = filename

            elif request.form.get("view_file"):
                audio_filename = secure_filename(request.form.get("view_file"))
                save_path = UPLOAD_FOLDER / audio_filename
                if not save_path.exists():
                    error = "The selected recording could not be found. Please upload again."
                else:
                    valid, msg = validate_wav(save_path)
                    if not valid:
                        error = msg or "Uploaded file failed validation."
                    else:
                        audio_url = url_for("static", filename=f"uploads/{audio_filename}")
                        view_requested = True

            elif request.form.get("process_file"):
                if not model_ready:
                    error = "The model is not available right now. Please try again later."
                else:
                    audio_filename = secure_filename(request.form.get("process_file"))
                    save_path = UPLOAD_FOLDER / audio_filename
                    if not save_path.exists():
                        error = "The selected recording could not be found. Please upload again."
                    else:
                        audio_url = url_for("static", filename=f"uploads/{audio_filename}")
                        valid, msg = validate_wav(save_path)
                        if not valid:
                            error = msg or "Uploaded file failed validation."
                        else:
                            features = extract_features_from_audio(str(save_path))
                            if features is None:
                                error = "Unable to process the uploaded audio file."
                            else:
                                df = model_utils.prepare_input(features, numeric_cols, cat_cols)
                                prediction = model.predict(df)[0]
                                result = prediction
        except Exception as exc:
            app.logger.exception('Unexpected error in upload/process workflow')
            error = "An unexpected server error occurred. Please try again with a valid WAV recording."

    return render_template(
        "index.html",
        result=result,
        error=error,
        audio_url=audio_url,
        audio_filename=audio_filename,
        model_ready=model_ready,
        model_load_error=model_load_error,
        model_expected_feature_count=model_expected_feature_count,
        model_expected_preview=model_expected_preview,
        model_class_labels=model_class_labels,
        view_requested=view_requested,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
