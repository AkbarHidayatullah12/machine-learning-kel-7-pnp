import os
import time
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

# =====================
# CONFIGURATION
# =====================
MODEL_PATH = "ad_detection_model.keras"

app = Flask(__name__)

# =====================
# HEALTH CHECK (WAJIB UNTUK KOYEB)
# =====================
@app.route("/health")
def health():
    return "ok", 200


# =====================
# LOAD MODEL (GLOBAL)
# =====================
print("Loading model...")
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# =====================
# ROOT ROUTE (AMAN UNTUK PUBLIC)
# =====================
@app.route("/")
def home():
    return "Ad Detection Service is running"


# =====================
# UI PAGE
# =====================
@app.route("/detector")
def detector():
    return render_template("index.html")


# =====================
# PREDICTION API
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        start_time = time.time()

        # Load & preprocess image
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction_score = float(model.predict(img_array, verbose=0)[0][0])
        processing_time = round(time.time() - start_time, 2)

        # Interpret result
        if prediction_score < 0.5:
            is_ad = True
            confidence = 1.0 - prediction_score
        else:
            is_ad = False
            confidence = prediction_score

        return jsonify({
            "is_ad": is_ad,
            "confidence": confidence,
            "raw_score": prediction_score,
            "processing_time": processing_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================
# LOCAL DEV & PRODUCTION
# =====================
if __name__ == "__main__":
    # Gunakan PORT dari environment variable, atau default ke 8000 jika tidak ada
    port = int(os.environ.get("PORT", 8000))
    # Host harus 0.0.0.0 agar bisa diakses dari luar container
    app.run(host="0.0.0.0", port=port)