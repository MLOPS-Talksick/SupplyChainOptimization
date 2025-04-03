# main.py - Flask application to serve the latest model from Vertex AI Registry
from flask import Flask, request, jsonify
import os
import json
import base64
from google.cloud import aiplatform
from dotenv import load_dotenv
from flask_cors import CORS
from utils import generate_prediction_data
import pandas as pd

load_dotenv()
app = Flask(__name__)
CORS(app)

# Global variable to store the loaded model
model = None
model_name = os.environ.get("MODEL_NAME", "model.pkl")

def load_latest_model():
    """
    Loads the latest version of the model from Vertex AI Model Registry
    """
    global model
    
    # Initialize Vertex AI
    aiplatform.init(project=os.environ.get("PROJECT_ID"))
    
    # Get the latest model version
    models = aiplatform.Model.list(
        filter=f'display_name="{model_name}"',
        order_by="create_time desc"
    )
    
    if not models:
        raise ValueError(f"No models found with name: {model_name}")
    
    latest_model = models[0]
    print(f"Loading latest model: {latest_model.display_name}, version: {latest_model.version_id}")
    
    # Load the model
    model = latest_model
    return model

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions using the loaded model
    """
    global model

    # Ensure model is loaded
    if model is None:
        try:
            model = load_latest_model()
        except Exception as e:
            return jsonify({"error": f"Failed to load model: {str(e)}"}), 500
    
    # Get input data from request
    try:
        content = request.json

        # Make prediction using the loaded model
        prediction_df = generate_prediction_data(7, "product_name", pd.to_datetime('2025-04-02'))
        response_json = prediction_df.to_dict(orient='records')

        return jsonify(response_json)
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """
    Endpoint to reload the latest model version
    Can be triggered manually or by Pub/Sub or Eventarc
    """
    global model
    try:

        if request.headers.get('Content-Type') == 'application/json' and not request.headers.get('X-Pubsub-Message'):
            try:
                model = load_latest_model()
                return jsonify({"message": f"Model reloaded successfully: {model.display_name}, version: {model.version_id}"}), 200
            
            except Exception as e:
                print(f"Error processing direct API call: {e}")
                return f"Error: {e}", 500

        envelope = request.get_json()
        if not envelope:
            return "No Pub/Sub message received", 400
        
        if not isinstance(envelope, dict) or "message" not in envelope:
            return "Invalid Pub/Sub message format", 400
        
        # Extract the message data
        pubsub_message = envelope["message"]
        if isinstance(pubsub_message, dict) and "data" in pubsub_message:
            try:
                # Decode the Pub/Sub data
                data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
                print(f"Received Pub/Sub message: {data}")
                event_data = json.loads(data)
                print(f"Event Data from Pub/Sub message: {event_data}")
                
                model = load_latest_model()
                return jsonify({"message": f"Model reloaded successfully: {model.display_name}, version: {model.version_id}"}), 200
            except Exception as e:
                print(f"Error processing Pub/Sub message: {e}")
                return f"Error: {e}", 500
        
        return "No message data found", 400
        
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    global model
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model.display_name if model else None,
        "model_version": model.version_id if model else None,
        "test_env": os.environ.get("PROJECT_ID"),
    }), 200
    
# Load the model when the server starts
with app.app_context():
    # global model
    try:
        model = load_latest_model()
    except Exception as e:
        print(f"Warning: Failed to load model at startup: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)