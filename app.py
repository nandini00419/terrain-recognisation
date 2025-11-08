"""
Flask Web Application for Terrain Recognition
Provides web interface for uploading images and getting terrain predictions.
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'terrain-recognition-secret-key'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Model and class names
model = None
class_names = None
IMG_SIZE = (224, 224)

# Prediction history file
PREDICTION_HISTORY_FILE = 'prediction_history.json'

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model and class indices."""
    global model, class_names
    
    model_path = 'terrain_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Load class indices
    class_indices_path = 'class_indices.json'
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        # Reverse mapping: index -> class name
        class_names = {v: k for k, v in class_indices.items()}
        print(f"Loaded {len(class_names)} terrain classes")
    else:
        print("Warning: class_indices.json not found. Using default class names.")
        class_names = {i: f"class_{i}" for i in range(model.output_shape[1])}

def preprocess_image(img_path):
    """
    Preprocess image for model prediction.
    Resizes to model input size and normalizes.
    """
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def get_terrain_emoji(terrain_class):
    """Get emoji for terrain class."""
    emoji_map = {
        'desert': '🏜️',
        'forest': '🌲',
        'mountain': '⛰️',
        'plain': '🌾',
        'urban': '🏙️',
        'water': '🌊'
    }
    return emoji_map.get(terrain_class.lower(), '🌍')

def predict_terrain(img_path):
    """
    Predict terrain type from image.
    Returns predicted class and confidence scores.
    """
    if model is None:
        raise RuntimeError("Model not loaded. Please restart the application.")
    
    # Preprocess image
    img_array = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get class name
    predicted_class = class_names.get(predicted_class_idx, f"Class {predicted_class_idx}")
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_predictions = [
        {
            'class': class_names.get(idx, f"Class {idx}"),
            'confidence': float(predictions[0][idx]),
            'confidence_percent': round(float(predictions[0][idx]) * 100, 1),
            'emoji': get_terrain_emoji(class_names.get(idx, ''))
        }
        for idx in top_indices
    ]
    
    return predicted_class, confidence, top_predictions

@app.route('/')
def index():
    """Home page with image upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return render_template('result.html', 
                             error='No file uploaded. Please select an image.')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('result.html', 
                             error='No file selected. Please choose an image.')
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict terrain
            predicted_class, confidence, top_predictions = predict_terrain(filepath)
            
            # Get image URL for display
            image_url = url_for('static', filename=f'uploads/{filename}')
            
            # Get emoji for predicted class
            predicted_emoji = get_terrain_emoji(predicted_class)
            
            # Add to prediction history
            add_prediction_to_history(predicted_class, confidence, image_url, top_predictions)
            
            return render_template('result.html',
                                 predicted_class=predicted_class,
                                 predicted_emoji=predicted_emoji,
                                 confidence=confidence,
                                 confidence_percent=round(confidence * 100, 2),
                                 image_url=image_url,
                                 top_predictions=top_predictions)
        except Exception as e:
            return render_template('result.html', 
                                 error=f'Prediction error: {str(e)}')
        finally:
            # Optionally delete the file after prediction to save space
            # os.remove(filepath)
            pass
    else:
        return render_template('result.html', 
                             error='Invalid file type. Please upload an image (png, jpg, jpeg, gif, bmp).')

@app.route('/predict_ajax', methods=['POST'])
def predict_ajax():
    """
    AJAX endpoint for real-time prediction (used by AI cursor feature).
    Returns JSON response.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Save temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
        file.save(filepath)
        
        try:
            # Predict terrain
            predicted_class, confidence, top_predictions = predict_terrain(filepath)
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up temp file
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def load_prediction_history():
    """Load prediction history from file."""
    if os.path.exists(PREDICTION_HISTORY_FILE):
        with open(PREDICTION_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_prediction_history(history):
    """Save prediction history to file."""
    with open(PREDICTION_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def add_prediction_to_history(predicted_class, confidence, image_url, top_predictions):
    """Add a prediction to history."""
    history = load_prediction_history()
    history.append({
        'class': predicted_class,
        'confidence': confidence,
        'confidence_percent': round(confidence * 100, 2),
        'emoji': get_terrain_emoji(predicted_class),
        'image_url': image_url,
        'top_predictions': top_predictions,
        'timestamp': datetime.now().isoformat()
    })
    # Keep only last 1000 predictions
    if len(history) > 1000:
        history = history[-1000:]
    save_prediction_history(history)

def get_statistics():
    """Calculate statistics from prediction history."""
    history = load_prediction_history()
    
    if not history:
        return {
            'total_predictions': 0,
            'model_accuracy': 99.9,  # From training
            'terrain_classes': len(class_names) if class_names else 0,
            'avg_confidence': 0,
            'training_accuracy': 99.5,
            'validation_accuracy': 99.9,
            'training_loss': 0.0101,
            'validation_loss': 0.0035,
            'terrain_distribution': {},
            'confidence_distribution': [0, 0, 0, 0, 0, 0]
        }
    
    # Calculate statistics
    total = len(history)
    avg_confidence = sum(p['confidence'] for p in history) / total * 100
    
    # Terrain distribution
    terrain_dist = defaultdict(int)
    for pred in history:
        terrain_dist[pred['class']] += 1
    
    # Confidence distribution
    confidence_ranges = [0, 0, 0, 0, 0, 0]  # 90-100, 80-90, 70-80, 60-70, 50-60, <50
    for pred in history:
        conf = pred['confidence_percent']
        if conf >= 90:
            confidence_ranges[0] += 1
        elif conf >= 80:
            confidence_ranges[1] += 1
        elif conf >= 70:
            confidence_ranges[2] += 1
        elif conf >= 60:
            confidence_ranges[3] += 1
        elif conf >= 50:
            confidence_ranges[4] += 1
        else:
            confidence_ranges[5] += 1
    
    return {
        'total_predictions': total,
        'model_accuracy': 99.9,
        'terrain_classes': len(class_names) if class_names else 0,
        'avg_confidence': round(avg_confidence, 1),
        'training_accuracy': 99.5,
        'validation_accuracy': 99.9,
        'training_loss': 0.0101,
        'validation_loss': 0.0035,
        'terrain_distribution': dict(terrain_dist),
        'confidence_distribution': confidence_ranges
    }

@app.route('/dashboard')
def dashboard():
    """Dashboard page with statistics and visualizations."""
    stats = get_statistics()
    return render_template('dashboard.html', stats=stats)

@app.route('/history')
def history():
    """Prediction history page."""
    return render_template('history.html')

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics."""
    stats = get_statistics()
    # Prepare data for charts
    terrain_dist = stats['terrain_distribution']
    all_classes = ['forest', 'mountain', 'plain', 'desert', 'urban', 'water']
    terrain_data = [terrain_dist.get(cls, 0) for cls in all_classes]
    
    # Calculate percentages
    total = sum(terrain_data) if sum(terrain_data) > 0 else 1
    terrain_percentages = [round(count / total * 100, 1) for count in terrain_data]
    
    return jsonify({
        'terrain_distribution': terrain_percentages,
        'confidence_distribution': stats['confidence_distribution'],
        'total_predictions': stats['total_predictions'],
        'avg_confidence': stats['avg_confidence']
    })

@app.route('/api/history')
def api_history():
    """API endpoint for prediction history."""
    history = load_prediction_history()
    return jsonify(history)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': len(class_names) if class_names else 0
    })

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load model on startup
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first using: python train_model.py")
        # Continue anyway for development (model will error on prediction)
    
    # Run Flask app
    print("\n" + "=" * 60)
    print("Terrain Recognition Web Application")
    print("=" * 60)
    print(f"Server running at: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

