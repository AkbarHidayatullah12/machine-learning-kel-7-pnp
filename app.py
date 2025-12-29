import os
import time
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import threading

# ==========================================
# PERBAIKAN LOGIKA 1: PENENTUAN PORT
# Logika Lama: PORT = 5000 (Koyeb marah karena port dia 8000)
# Logika Baru: Cek dulu environment variable, kalau tidak ada baru pakai 5000
# ==========================================
PORT = int(os.environ.get("PORT", 5000))
MODEL_PATH = 'ad_detection_model.keras'

app = Flask(__name__)

# Load Model (Global)
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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detector')
def detector():
    return render_template('index.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        start_time = time.time()
        
        # Preprocess Image
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Predict
        prediction_score = float(model.predict(img_array, verbose=0)[0][0])
        processing_time = round(time.time() - start_time, 2)
        
        # Logic: Class 0 is 'iklan', Class 1 is 'noniklan'
        # Prediction score is probability of 'noniklan' (Class 1)
        
        if prediction_score < 0.5:
            is_ad = True
            confidence = 1.0 - prediction_score
        else:
            is_ad = False
            confidence = prediction_score
            
        return jsonify({
            'is_ad': is_ad,
            'confidence': confidence,
            'raw_score': prediction_score,
            'processing_time': processing_time
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # ==========================================
    # PERBAIKAN LOGIKA 2: HOST BINDING
    # Logika Lama: app.run(...) -> Jalan di localhost (Koyeb tidak bisa masuk)
    # Logika Baru: host='0.0.0.0' -> Jalan di semua jaringan (Koyeb bisa masuk)
    # ==========================================
    app.run(debug=True, host='0.0.0.0', port=PORT, threaded=True)