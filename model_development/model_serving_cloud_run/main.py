# main.py - Flask application to serve the latest model from Vertex AI Registry
from flask import Flask, request, jsonify
import os
import json
import base64
from tensorflow.keras.models import load_model
from google.cloud import storage
from dotenv import load_dotenv
from flask_cors import CORS
from utils import get_latest_data_from_cloud_sql, upsert_df, predict_future_demand, correct_prediction_bias, load_preprocessing_objects, generate_future_features, create_features, apply_log_transform, apply_rounding_strategy

from datetime import datetime

import pandas as pd
import numpy as np
import pickle
import logging

import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO)

load_dotenv()
app = Flask(__name__)
CORS(app)

# Global variable to store the loaded model
model = None
model_name = os.environ.get("MODEL_NAME", "lstm_model.keras")

def load_from_gcs(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(source_blob_name)

    print(f"File {source_blob_name} downloaded.")

    return os.path.abspath(source_blob_name)

def load_latest_model():
    """
    Loads the latest version of the model from Vertex AI Model Registry
    """
    global model
    model_path = load_from_gcs("trained-model-1", model_name)
    return load_model(model_path)

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
        data = request.get_json()
        # latest_date = pd.to_datetime(data.get('date', pd.Timestamp.today()))
        days = data.get('days') 
        time_steps = 5

        scaler_X_path = load_from_gcs("model_training_1", 'scaler_X.pkl')
        scaler_y_path = load_from_gcs("model_training_1",'scaler_y.pkl')
        label_encoder_path = load_from_gcs("model_training_1",'label_encoder.pkl')
        log_transformed_path = load_from_gcs("model_training_1",'transform_info.pkl')

           
        if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path) or not os.path.exists(label_encoder_path):
            raise FileNotFoundError("Preprocessor files (scalers or encoder) not found in folder")
        
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        # Load transformation flag
        with open(log_transformed_path, 'rb') as f:
            transform_info = pickle.load(f)
            log_transformed = transform_info.get('log_transformed', False)
        
        
        print("Preprocessors loaded successfully")

        query = f"""
            SELECT 
                sale_date AS 'Date', 
                product_name AS 'Product Name', 
                total_quantity AS 'Total Quantity'
            FROM SALES
            ORDER BY Date;
        """

        df = get_latest_data_from_cloud_sql(query=query)
        print(df.head())

        # Get unique product names
        unique_products = df['Product Name'].unique()
        print(f"Found {len(unique_products)} unique products")
        
        # Initialize empty dataframe to store all predictions
        all_predictions_df = pd.DataFrame()

        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        latest_date = df['Date'].max()
        # Get the current date
        # ++++++++++++++++++++++++++++++++++++++++++++++++++
        current_date = datetime.now().date()
        days_difference = (current_date - latest_date.date()).days

        days = days + days_difference
        # ++++++++++++++++++++++++++++++++++++++++++++++++++

        df = df[
            df['Date'] >= latest_date - pd.Timedelta(days=60)
        ]


        feature_df, _ = create_features(df)
        feature_df, log_transformed = apply_log_transform(feature_df)


        # ——— pick your feature list ———
        features = [
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter',
            'product_encoded', 'is_weekend', 'is_month_start', 'is_month_end', 'is_holiday',
            'rolling_mean_7d', 'rolling_std_7d', 'rolling_median_7d', 'rolling_min_7d', 'rolling_max_7d',
            'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d', 'lag_14d',
            'diff_1d', 'diff_7d', 'ewm_alpha_0.3', 'ewm_alpha_0.7',
            'product_mean', 'product_median', 'product_std', 'product_min', 'product_max'
        ]
        available_features = [f for f in features if f in feature_df.columns]

        # ——— generate only the FUTURE feature rows ———
        future_data = generate_future_features(
            feature_df, days_to_predict=days, features=available_features
        )

        # ——— grab the last `time_steps` historical feature‐rows per product ———
        seed_history = (
            feature_df
            .sort_values('Date')
            .groupby('Product Name')
            .tail(time_steps)
            [['Date', 'Product Name'] + available_features]
        )

        # ——— combine the seed + future frames ———
        combined = pd.concat([seed_history, future_data], ignore_index=True)
        combined = combined.sort_values(['Product Name','Date']).reset_index(drop=True)

        # ——— finally forecast *every* one of those future days ———
        predictions_df = predict_future_demand(
            model,
            combined,
            available_features,
            scaler_X,
            scaler_y,
            time_steps=time_steps,
            bias_correction_func=None,
            log_transformed=log_transformed
        )

        predictions_df = apply_rounding_strategy(predictions_df, safety_stock=1)
        
        # Make prediction using the loaded model
        # Return error if no predictions were made
        if len(predictions_df) == 0:
            return jsonify({"preds": "No predictions could be generated for any product"}), 500
        
        logging.info("Adding data to SQL..")
        upsert_df(predictions_df, 'PREDICT')
        df_json = predictions_df.to_dict(orient='records')

        return jsonify({"preds": df_json})
    
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
    app.run(host="0.0.0.0", port=port,debug=True)