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
CORS(app)

try:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH", "rt-ecg-12-firebase-adminsdk-fbsvc-009271066f.json"))
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://rt-ecg-12-default-rtdb.asia-southeast1.firebasedatabase.app"
    })
except Exception as e:
    print(f"Firebase initialization failed: {e}")

USER_ID = "nHUTGVGOCIaa9MemnWn4AchbWGG2"

try:
    MODEL_PATH = os.getenv("MODEL_PATH", "ecg_transformer.keras")
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

def fetch_latest_ecg():
    try:
        ref_path = f"/UsersData/{USER_ID}/ecgReadings"
        ecg_readings = db.reference(ref_path).get()

        if not ecg_readings:
            return None

        latest_reading = next(iter(ecg_readings.values()))
        latest_ecg = latest_reading["ecg"]
        return np.array(latest_ecg, dtype=np.float32)
    except Exception as e:
        print(f"Error fetching ECG data: {e}")
        return None

def calculate_vitals(ecg_signal, sampling_rate=250):
    if ecg_signal is None or len(ecg_signal) < 2:
        return None, None, None

    peaks, _ = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    r_peaks = np.where(peaks['ECG_R_Peaks'] == 1)[0]

    if len(r_peaks) < 2:
        return None, None, None

    rr_intervals = np.diff(r_peaks) / sampling_rate
    bpm = 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else None
    sbp = 0.4 * bpm + 90 if bpm else None
    dbp = 0.2 * bpm + 60 if bpm else None
    
    return bpm, sbp, dbp

def classify_ecg(ecg_signal):
    if ecg_signal is None or model is None:
        return "No ECG data or model available", "No suggestions available"

    ecg_signal = preprocess_ecg(ecg_signal)
    prediction = model.predict(ecg_signal)
    predicted_class = np.argmax(prediction)

    arrhythmia_classes = ["Normal", "PVC", "APC", "LBBB", "RBBB"]
    suggestions = [
        "Heart rhythm appears normal. Maintain a healthy lifestyle.",
        "Possible Premature Ventricular Contraction (PVC). Avoid caffeine and manage stress.",
        "Possible Atrial Premature Complex (APC). Regular check-ups recommended.",
        "Possible Left Bundle Branch Block (LBBB). Consult a cardiologist.",
        "Possible Right Bundle Branch Block (RBBB). Monitor symptoms and consult a specialist."
    ]
    
    predicted_label = arrhythmia_classes[predicted_class] if predicted_class < len(arrhythmia_classes) else "Unknown"
    suggestion = suggestions[predicted_class] if predicted_class < len(suggestions) else "Consult a doctor for further evaluation."
    
    return predicted_label, suggestion

@app.route("/")
def index():
    ecg_signal = fetch_latest_ecg()
    classification, suggestion = classify_ecg(ecg_signal)
    bpm, sbp, dbp = calculate_vitals(ecg_signal)
    return render_template("index.html", classification=classification, suggestion=suggestion, heart_rate=bpm, sbp=sbp, dbp=dbp)

@app.route("/fetch_ecg", methods=["GET"])
def get_ecg():
    ecg_signal = fetch_latest_ecg()
    classification, suggestion = classify_ecg(ecg_signal)
    bpm, sbp, dbp = calculate_vitals(ecg_signal)

    if ecg_signal is None:
        return jsonify({"error": "No ECG data found"}), 404

    return jsonify({
        "ecg_signal": ecg_signal.tolist(),
        "arrhythmia_class": classification,
        "suggestion": suggestion,
        "heart_rate": bpm,
        "sbp": sbp,
        "dbp": dbp
    })

@app.route("/ecg_plot", methods=["GET"])
def plot_ecg():
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
    try:
        db.reference("/").get()
        if model is None:
            raise Exception("Model not loaded")
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
