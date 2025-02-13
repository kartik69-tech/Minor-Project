from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__, static_folder='../frontend')

# Load model and scaler
model = tf.keras.models.load_model('ml_model/fabric_grader.h5')
scaler = joblib.load('ml_model/scaler.pkl')

# Function to predict fabric score
def predict_score(defects, color_var, texture_var):
    # Scale input data
    input_data = scaler.transform([[defects, color_var, texture_var]])
    
    # Predict grades (1-4 for each attribute)
    pred_defects, pred_color, pred_texture = model.predict(input_data)
    
    # Convert probabilities to grades (1-4)
    grade_defects = np.argmax(pred_defects, axis=1)[0] + 1
    grade_color = np.argmax(pred_color, axis=1)[0] + 1
    grade_texture = np.argmax(pred_texture, axis=1)[0] + 1
    
    # Calculate score out of 100
    weights = {'defects': 0.5, 'color': 0.3, 'texture': 0.2}
    score_defects = ((4 - grade_defects) / 3) * weights['defects'] * 100
    score_color = ((4 - grade_color) / 3) * weights['color'] * 100
    score_texture = ((4 - grade_texture) / 3) * weights['texture'] * 100
    
    total_score = score_defects + score_color + score_texture
    return total_score

# API endpoint to grade fabric
@app.route('/api/grade', methods=['POST'])
def grade_fabric():
    try:
        # Get input data from request
        data = request.json
        defects = data['defects']
        color_var = data['color_var']
        texture_var = data['texture_var']
        
        # Predict score
        score = predict_score(defects, color_var, texture_var)
        
        # Return score as JSON response
        return jsonify({
            'score': round(score, 2),
            'grade': 'Excellent' if score >= 85 else 'Good' if score >= 70 else 'Average' if score >= 50 else 'Poor'
        })
    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)}), 400

# Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)