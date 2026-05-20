# Stethoscope

This project contains an AI model and an Arduino-based stethoscope that have been connected and displayed through an interface.

The hardware shows a low-fidelity model of a stethoscope. The software is an AI model trained for lung sound diagnosis. The web interface allows users to upload audio, preview it, and see a diagnosis result.

Arduino audio integration:
- The Flask app exposes `POST /arduino/upload` for receiving `.wav` files from Arduino/Python serial capture.
- The website renders current Arduino upload status and diagnosis via the `arduino_state` display card.
- Use `arduino_to_flask.py` to send a saved `mic.wav` file from local Python to the Flask endpoint.
