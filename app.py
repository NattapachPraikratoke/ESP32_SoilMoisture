from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os
import requests
from datetime import datetime, timedelta

# ===============================
# App Setup
# ===============================
app = Flask(__name__)

MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

# ===============================
# Load Labels
# ===============================
def load_labels(path):
    if not os.path.exists(path):
        return ["dry", "normal", "wet"]
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

labels = load_labels(LABELS_PATH)

# ===============================
# TensorFlow / TFLite Lazy Loader
# ===============================
interpreter = None
input_details = None
output_details = None
INPUT_SIZE = None

def load_interpreter():
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter

def get_interpreter():
    global interpreter, input_details, output_details, INPUT_SIZE

    if interpreter is None:
        Interpreter = load_interpreter()
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        INPUT_SIZE = input_details[0]["shape"][1]

        print("✅ TFLite model loaded")
        print("📐 Input size:", INPUT_SIZE)
        print("🏷 Labels:", labels)

    return interpreter

# ===============================
# Telegram Config
# ===============================
TELEGRAM_TOKEN = os.getenv("8323059048:AAH6K8q48-0wiF-aEnc0Tro0o2s49opVkRs")
TELEGRAM_CHAT_ID = os.getenv("6423545257")

def thai_time():
    return (datetime.utcnow() + timedelta(hours=7)).strftime("%H:%M:%S")

def send_telegram_photo(photo_bytes, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram not configured")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    response = requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML",
            "disable_notification": "false"
        },
        files={
            "photo": ("image.jpg", photo_bytes, "image/jpeg")
        },
        timeout=20
    )

    if response.status_code != 200:
        print("❌ Telegram error:", response.text)

    return response.ok

# ===============================
# Health Check
# ===============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model_loaded": interpreter is not None,
        "labels": labels,
        "telegram_ready": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "time": thai_time()
    })

# ===============================
# Prediction Endpoint
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.json or "image" not in request.json:
            return jsonify({"error": "No image provided"}), 400

        # Lazy-load interpreter
        interp = get_interpreter()

        # Decode image
        image_bytes = base64.b64decode(request.json["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE))

        # Preprocess
        input_data = np.expand_dims(
            np.array(image, dtype=np.float32) / 255.0,
            axis=0
        )

        # Inference
        interp.set_tensor(input_details[0]["index"], input_data)
        interp.invoke()
        output = interp.get_tensor(output_details[0]["index"])[0]

        # Softmax (safe even if already applied)
        exp = np.exp(output - np.max(output))
        probs = exp / exp.sum()

        class_id = int(np.argmax(probs))
        label = labels[class_id]
        confidence = float(probs[class_id]) * 100.0

        # Telegram caption (ALWAYS SEND)
        caption = (
            f"🌱 <b>Soil Moisture Prediction</b>\n"
            f"📌 Class: <b>{label.upper()}</b>\n"
            f"📊 Confidence: {confidence:.1f}%\n"
            f"⏰ Time: {thai_time()}"
        )

        send_telegram_photo(image_bytes, caption)

        return jsonify({
            "class": label,
            "confidence": round(confidence, 2),
            "time": thai_time()
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500

# ===============================
# Run (Render / Gunicorn)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
