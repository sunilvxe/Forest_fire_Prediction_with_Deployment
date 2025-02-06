from flask import Flask, request, jsonify
import numpy as np
import joblib  # For loading the trained model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('fire_prediction_model.pkl')  # Replace with your model's path

@app.route('/')
def home():
    with open('templates/index.html', 'r') as file:
        return file.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from POST request
        data = request.json
        
        # Ensure these keys match the features used in the model
        required_features = ['temperature', 'humidity', 'TVOC', 'eCO2', 'raw_h2', 'raw_ethanol', 'pressure', 'pm1', 'pm2.5']
        
        # Validate input
        if not all(key in data for key in required_features):
            return jsonify({'error': 'Missing features in input data'}), 400
        
        # Create feature vector for prediction
        features = np.array([[data[feature] for feature in required_features]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return the result
        return jsonify({'fire_alarm': bool(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
