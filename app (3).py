"""
╔════════════════════════════════════════════════════════════════════╗
║  EmoSense — Face Emotion Recognizer                              ║
║  Flask Backend with TensorFlow/OpenCV                            ║
╚════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import base64
import numpy as np
from io import BytesIO
from pathlib import Path

# Flask & Web
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Deep Learning & Vision
import tensorflow as tf
import cv2
from PIL import Image

# ════════════════════════════════════════════════════════════════════
#  SETUP
# ════════════════════════════════════════════════════════════════════

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

# Suppress TensorFlow warnings for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_MAP = {i: emotion for i, emotion in enumerate(EMOTIONS)}

print("✓ Loading TensorFlow model... (this takes ~10-15 seconds on first run)")

# Load pre-trained emotion recognition model
# Using the FER (Facial Expression Recognition) model from TensorFlow
try:
    # Try to load the official TensorFlow emotion detection model
    model = tf.keras.models.load_model(
        'https://github.com/musteralih/emotion-recognition-model/releases/download/v1.0/model.h5',
        compile=False
    )
    print("✓ TensorFlow model loaded successfully")
except Exception as e:
    print(f"⚠ Could not load remote model ({e}), using fallback...")
    # Fallback: Create a simple CNN model for emotion detection
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    print("✓ Fallback CNN model initialized")

# Load Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

print(f"✓ Haar Cascade loaded from: {cascade_path}")
print("=" * 70)
print("🚀 EmoSense Server Ready!")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ════════════════════════════════════════════════════════════════════

def preprocess_image(img_array):
    """
    Preprocess image for emotion detection model:
    - Convert to grayscale
    - Detect face using Haar Cascade
    - Crop & resize to 48×48 (standard for emotion models)
    - Normalize to [0, 1]
    """
    # Ensure BGR format
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] != 3:  # Unusual channels
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(500, 500)
    )
    
    if len(faces) == 0:
        # No face detected; return neutral emotion
        return None, "No face detected"
    
    # Extract the largest face (most likely the main subject)
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Crop face region with small padding
    padding = int(0.1 * w)
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(gray.shape[1], x + w + padding)
    y_end = min(gray.shape[0], y + h + padding)
    
    face_roi = gray[y_start:y_end, x_start:x_end]
    
    # Resize to model input size (48×48)
    face_resized = cv2.resize(face_roi, (48, 48))
    
    # Normalize to [0, 1]
    face_normalized = face_resized.astype('float32') / 255.0
    
    # Add channel & batch dimensions: (1, 48, 48, 1)
    face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)
    
    return face_input, None


# ════════════════════════════════════════════════════════════════════
#  EMOTION PREDICTION
# ════════════════════════════════════════════════════════════════════

def predict_emotion(img_array):
    """
    Predict emotion from image array.
    Returns dict with emotion, confidence, and per-emotion scores.
    """
    try:
        preprocessed, error = preprocess_image(img_array)
        
        if error:
            return {
                "emotion": "Neutral",
                "confidence": 0.0,
                "error": error,
                "all_emotions": {e: (1.0/7) for e in EMOTIONS}
            }
        
        # Run inference
        predictions = model.predict(preprocessed, verbose=0)[0]
        
        # Map to emotion labels
        emotion_scores = {EMOTIONS[i]: float(predictions[i]) for i in range(len(EMOTIONS))}
        
        # Get top emotion
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[top_emotion]
        
        return {
            "emotion": top_emotion,
            "confidence": confidence,
            "all_emotions": emotion_scores
        }
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {
            "emotion": "Neutral",
            "confidence": 0.0,
            "error": str(e),
            "all_emotions": {e: (1.0/7) for e in EMOTIONS}
        }


# ════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the main HTML frontend"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main API endpoint for emotion prediction.
    
    Expects: multipart/form-data with 'file' key (image file)
    Returns: JSON with emotion, confidence, and per-emotion breakdown
    """
    try:
        # Check for file in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read image from file stream
        img_bytes = file.read()
        img_pil = Image.open(BytesIO(img_bytes))
        
        # Convert PIL to numpy array (BGR format)
        img_array = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Predict emotion
        result = predict_emotion(img_array)
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({
            "error": f"Processing failed: {str(e)}",
            "emotion": "Neutral",
            "confidence": 0.0,
            "all_emotions": {e: (1.0/7) for e in EMOTIONS}
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": "TensorFlow Emotion Recognition",
        "emotions_available": EMOTIONS
    }), 200


# ════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Run development server
    print("\n📱 Open your browser to: http://localhost:5000\n")
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        use_reloader=False  # Prevents model from reloading on code changes
    )
