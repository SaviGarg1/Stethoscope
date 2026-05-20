# Stethoscope

A simple Flask web app for lung sound classification.

## What it does

- Serves a single homepage at `/`
- Allows upload of a `.wav` lung sound file
- Validates the uploaded file before processing
- Extracts audio features and passes them into a pre-trained model
- Displays a single diagnosis label on the same page

## Project structure

- `app.py` — Flask app and upload/prediction logic
- `templates/index.html` — simple single-page UI
- `static/uploads/` — temporary storage for uploaded files
- `rf_sound_demographics_pipeline.joblib` — pre-trained sklearn pipeline
- `model_utils.py` — helper that prepares feature input for the pipeline

## Does the model work?

Yes. The prediction pipeline was tested locally using the provided `mic.wav` sample file, and the app successfully returned a diagnosis label.

## Local setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Flask app:

```bash
python app.py
```

3. Open the app in your browser:

```text
http://127.0.0.1:5000
```

4. Upload a `.wav` file and submit to see the prediction.

## Environment variables

The app supports these optional environment variables:

- `SECRET_KEY` — Flask secret key
- `MODEL_PATH` — path to the model file (defaults to `rf_sound_demographics_pipeline.joblib`)
- `MAX_CONTENT_LENGTH` — max upload size in bytes (defaults to `16777216`)

## Deployment on Render

Use `gunicorn` as the entrypoint for Render:

```bash
gunicorn app:app
```

If you want, I can also add a small `Procfile` and deployment-ready README section next.
