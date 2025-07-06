import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, jsonify, send_file, render_template
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Initialize Firebase
try:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "PATH TO JSON FILE"))
    firebase_admin.initialize_app(cred, {
        "databaseURL": "databaseURL"
    })
    print("✅ Firebase initialized successfully")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")

# User ID for fetching data
USER_ID = "nHUTGVGOCIaa9MemnWn4AchbWGG2"

# Load deep learning model
try:
    MODEL_PATH = os.getenv("MODEL_PATH", "ecg_transformer.keras")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

def fetch_latest_ecg():
    """Fetch the latest ECG reading from Firebase."""
    try:
        ref_path = f"/UsersData/{USER_ID}/ecgReadings"
        ecg_readings = db.reference(ref_path).get()

        if not ecg_readings:
            print("❌ No ECG data found")
            return None

        latest_reading = list(ecg_readings.values())[-1]  # Get the latest entry
        latest_ecg = latest_reading.get("ecg", [])
        
        if not latest_ecg:
            print("❌ No 'ecg' key found in latest reading")
            return None

        return np.array(latest_ecg, dtype=np.float32)
    except Exception as e:
        print(f"❌ Error fetching ECG data: {e}")
        return None

def normalize_ecg(ecg_signal):
    """Normalize ECG signal between -1 and 1."""
    if ecg_signal is None:
        return None

    ecg_min = np.min(ecg_signal)
    ecg_max = np.max(ecg_signal)

    if ecg_max != ecg_min:
        ecg_signal = 2 * ((ecg_signal - ecg_min) / (ecg_max - ecg_min)) - 1
    else:
        ecg_signal = np.zeros_like(ecg_signal)

    return ecg_signal
    
def preprocess_ecg(ecg_signal):
    """Preprocess ECG signal for model inference."""
    if ecg_signal is None:
        return None

    ecg_signal = normalize_ecg(ecg_signal)

    target_length = 100
    if len(ecg_signal) < target_length:
        ecg_signal = np.pad(ecg_signal, (0, target_length - len(ecg_signal)), 'constant')
    else:
        ecg_signal = ecg_signal[:target_length]

    ecg_signal = np.expand_dims(ecg_signal, axis=-1)
    ecg_signal = np.expand_dims(ecg_signal, axis=0) 

    return ecg_signal

def classify_ecg(ecg_signal):
    """Classify ECG signal using the deep learning model."""
    if ecg_signal is None or model is None:
        return "No ECG data or model available"

    ecg_signal = preprocess_ecg(ecg_signal)
    prediction = model.predict(ecg_signal)
    predicted_class = np.argmax(prediction)

    arrhythmia_classes = ["Normal", "PVC", "APC", "LBBB", "RBBB"]
    predicted_label = arrhythmia_classes[predicted_class] if predicted_class < len(arrhythmia_classes) else "Unknown"

    return predicted_label

@app.route("/")
def index():
    """Render the main webpage."""
    ecg_signal = fetch_latest_ecg()
    classification = classify_ecg(ecg_signal)
    return render_template("index.html", classification=classification)

@app.route("/fetch_ecg", methods=["GET"])
def get_ecg():
    """Fetch ECG data and return classification."""
    ecg_signal = fetch_latest_ecg()
    classification = classify_ecg(ecg_signal)

    if ecg_signal is None:
        return jsonify({"error": "No ECG data found"}), 404

    return jsonify({
        "ecg_signal": ecg_signal.tolist(),
        "arrhythmia_class": classification
    })

@app.route("/ecg_plot", methods=["GET"])
def plot_ecg():
    """Generate and serve an ECG plot image."""
    ecg_signal = fetch_latest_ecg()

    if ecg_signal is None:
        return jsonify({"error": "No ECG data available"}), 404

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(ecg_signal)), ecg_signal, label="ECG Signal", color="blue")
    plt.xlabel("Sample Index")
    plt.ylabel("ECG Amplitude")
    plt.title("Real-Time ECG Signal")
    plt.legend()
    plt.grid(True)

    os.makedirs("static", exist_ok=True)
    plot_filename = "ecg_plot.png"
    plot_path = os.path.join("static", plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return send_file(plot_path, mimetype="image/png")

@app.route("/health", methods=["GET"])
def health_check():
    """Check if the API is running and connected to Firebase."""
    try:
        db.reference("/").get()
        if model is None:
            raise Exception("Model not loaded")
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
