""" 
Flask web service for Crime Prediction Model
"""

from flask import Flask, request, jsonify
import traceback
from predict import CrimePredictionModel

app = Flask(__name__)

print("Loading Model")
model = CrimePredictionModel()
print("Model Loaded Succesfully!")

@app.route('/ping', methods=['GET'])
def ping():
    """
    Status ok endpoint
    """
    return jsonify({
        'status': 'ok', 
        'message': 'Crime Prediction Service is running'
    })
    
    
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expects JSON with country features
    
    Example request:
    {
        "continent": "Asia",
        "school_enrollment_secondary_gross_wdi": 85.96,
        "trade_of_gdp_wdi": 67.58,
        ...
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please send JSONM data in the request body'
            }), 400
            
        # Make prediction
        prediction = model.predict(data)
        
        return jsonify({
            'criminality_score': round(prediction, 2),
            'status': 'success',
            # 'input_features': list(data.keys())
        })
        
         
    except KeyError as e:
        return jsonify({
            'error': 'Missing required feature',
            'message': f'Feature {str(e)} is required but not provided',
            'status': 'error'
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'status': 'error'
        }), 500   
        
@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Returns information about the loaded model
    """
    try:
        return jsonify({
            'model_type': str(type(model.model).__name__),
            'features': model.vectorizer.feature_names_ if hasattr(model.vectorizer, 'feature_names_') else 'N/A',
            'n_features': len(model.vectorizer.feature_names_) if hasattr(model.vectorizer, 'feature_names_') else 'N/A',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e),
            'status': 'error'
        }), 500
        
        
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)