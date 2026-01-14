from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import cv2
import numpy as np
import base64
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

@app.route('/')
def home():
    return jsonify({
        "message": "Sign Language Recognition API",
        "status": "running",
        "endpoints": {
            "predict": "POST /predict - Process sign language image"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        
        # Convert base64 to OpenCV format
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image_rgb)
        
        # Check if hands are detected
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # For now, return basic hand detection
            prediction = "Hand Detected"
            confidence = 0.95
            
            return jsonify({
                "prediction": prediction,
                "confidence": confidence,
                "hands_detected": num_hands,
                "status": "success"
            })
        else:
            return jsonify({
                "prediction": "No hands detected",
                "confidence": 0.0,
                "hands_detected": 0,
                "status": "no_hands"
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "sign-language-api"})

if __name__ == '__main__':
    print("🚀 Starting Sign Language Recognition API...")
    print("📡 Server running on http://0.0.0.0:5000")
    print("🔗 Home: http://localhost:5000")
    print("🖐️  Predict: POST http://localhost:5000/predict")
    app.run(host='0.0.0.0', port=5000, debug=True)