from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to the Treatment Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        try:
            features = data['features']
            # Check if features are all numbers
            if not all(isinstance(item, (int, float)) for item in features):
                return jsonify({'error': 'All features must be numeric.'}), 400

            prediction = model.predict(np.array([features]))
            return jsonify({'prediction': prediction[0]})
        except KeyError:
            return jsonify({'error': 'Invalid input, please provide features in the request body.'}), 400
    else:
        return jsonify({'error': 'Request must be JSON'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
