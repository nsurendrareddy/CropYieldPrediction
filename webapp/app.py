from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Global variables to store model and encoder
model = None
label_encoder = None

def load_model():
    """Load the trained model and label encoder"""
    global model, label_encoder
    
    try:
        # Load the model
        with open('model/crop_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load the label encoder
        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        print("Model and label encoder loaded successfully!")
        return True
    
    except FileNotFoundError:
        print("Model files not found. Please run train_model.py first.")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        crop = request.form['crop']
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        area = float(request.form['area'])
        
        # Validate inputs
        if rainfall < 0 or temperature < -50 or temperature > 60:
            return jsonify({'error': 'Invalid input values'}), 400
        
        if humidity < 0 or humidity > 100:
            return jsonify({'error': 'Humidity must be between 0 and 100'}), 400
        
        if area <= 0:
            return jsonify({'error': 'Area must be positive'}), 400
        
        # Encode crop name
        try:
            crop_encoded = label_encoder.transform([crop])[0]
        except ValueError:
            return jsonify({'error': f'Unknown crop type: {crop}'}), 400
        
        # Prepare input for model
        input_features = [[crop_encoded, rainfall, temperature, humidity, area]]
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Calculate total yield
        total_yield = prediction * area
        
        # Prepare response
        result = {
            'yield_per_hectare': round(prediction, 2),
            'total_yield': round(total_yield, 2),
            'crop': crop,
            'rainfall': rainfall,
            'temperature': temperature,
            'humidity': humidity,
            'area': area
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': 'Invalid input format'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/crops')
def get_crops():
    """Get available crop types"""
    if label_encoder is not None:
        crops = label_encoder.classes_.tolist()
        return jsonify({'crops': crops})
    else:
        return jsonify({'crops': ['Rice', 'Wheat', 'Maize', 'Barley', 'Soybean', 'Cotton']})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and label_encoder is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'model not loaded',
        'model_loaded': model_loaded
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please run train_model.py first.")