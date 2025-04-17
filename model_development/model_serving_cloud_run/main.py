# main.py - Flask application to serve the latest model from Vertex AI Registry
from flask import Flask, request, jsonify
import os
import json
import base64
from tensorflow.keras.models import load_model
from google.cloud import storage
from dotenv import load_dotenv
from flask_cors import CORS
from utils import get_latest_data_from_cloud_sql
import pandas as pd
import numpy as np
import pickle

load_dotenv()
app = Flask(_name_)
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
        latest_date = pd.to_datetime(data.get('date'))
        print(f"Received date: {latest_date}")
        days = data.get('days') 

        scaler_X_path = load_from_gcs("model_training_1", 'scaler_X.pkl')
        scaler_y_path = load_from_gcs("model_training_1",'scaler_y.pkl')
        label_encoder_path = load_from_gcs("model_training_1",'label_encoder.pkl')

           
        if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path) or not os.path.exists(label_encoder_path):
            raise FileNotFoundError("Preprocessor files (scalers or encoder) not found in folder")
        
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
        
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
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
        df_copy = df.copy()
        # Process each product
        for product_name in unique_products:
            try:
                print(f"Processing product: {product_name}")

                df = df_copy[df_copy['Product Name'] == product_name].copy()
                

                # Filter the DataFrame to get only the last 60 days from the latest date
                df = df[df['Date'] >= (latest_date - pd.Timedelta(days=60))]
                
                # Convert data types
                df['Total Quantity'] = df['Total Quantity'].astype(int)
                
                # Sort data by date
                df = df.sort_values('Date')
        
                # Extract date features
                df['year'] = df['Date'].dt.year
                df['month'] = df['Date'].dt.month
                df['day'] = df['Date'].dt.day
                df['dayofweek'] = df['Date'].dt.dayofweek
                df['dayofyear'] = df['Date'].dt.dayofyear
                df['quarter'] = df['Date'].dt.quarter
        
                # Calculate rolling statistics for the product
                df = df.sort_values(['Product Name', 'Date'])
                df['rolling_mean_7d'] = df.groupby('Product Name')['Total Quantity'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean())
                df['rolling_std_7d'] = df.groupby('Product Name')['Total Quantity'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))
                
                # Create lag features
                for lag in [1, 2, 3, 7]:
                    df[f'lag_{lag}d'] = df.groupby('Product Name')['Total Quantity'].shift(lag)
                
                # Fill NaN values
                df = df.fillna(0)

                # Encode product name
                try:
                    product_idx = label_encoder.transform([product_name])[0]
                    df['product_encoded'] = product_idx
                except ValueError:
                    print(f"Product '{product_name}' not found in the training data, skipping")
                    continue
                
                # Define features (must match what was used for training)
                features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 
                            'product_encoded', 'rolling_mean_7d', 'rolling_std_7d',
                            'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d']
        
                product_data = df[df['product_encoded'] == product_idx].sort_values('Date')
                
                if len(product_data) < 5:  # Need at least time_steps data points (assuming 5 time steps)
                    raise ValueError(f"Not enough historical data for product '{product_name}'")

                future_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=days)
                
                # Get the features from the most recent data
                # Assuming time_steps=5 as in the original code
                time_steps = 5
                
                # Extract the features for the last time_steps periods
                recent_data = product_data.iloc[-time_steps:][features].values
                
                # Scale the input
                recent_data_scaled = scaler_X.transform(recent_data)
                
                # Reshape for LSTM [samples, time_steps, features]
                current_sequence = recent_data_scaled.reshape(1, time_steps, len(features))
                
                predictions = []
                current_sequence = current_sequence[0]  # Get the sequence as a 2D array
                
                for i in range(days):
                    # Reshape for prediction
                    current_sequence_reshaped = current_sequence.reshape(1, time_steps, len(features))
                    
                    # Predict next day
                    next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)
                    next_pred = scaler_y.inverse_transform(next_pred_scaled)[0][0]
                    predictions.append(next_pred)
                    
                    # Create features for the next day
                    next_date = future_dates[i]
                    next_features = np.zeros(len(features))
                    
                    # Update date features
                    next_features[0] = next_date.year
                    next_features[1] = next_date.month
                    next_features[2] = next_date.day
                    next_features[3] = next_date.dayofweek
                    next_features[4] = next_date.dayofyear
                    next_features[5] = next_date.quarter
                    
                    # Product feature remains the same
                    next_features[6] = product_idx  # product_encoded
                    
                    # Update lag features based on predictions
                    if i == 0:
                        # For the first prediction, use values from the dataset
                        next_features[7] = product_data['rolling_mean_7d'].iloc[-1]  # rolling_mean_7d
                        next_features[8] = product_data['rolling_std_7d'].iloc[-1]   # rolling_std_7d
                        next_features[9] = product_data['Total Quantity'].iloc[-1]   # lag_1d
                        next_features[10] = product_data['Total Quantity'].iloc[-2] if len(product_data) > 1 else 0  # lag_2d
                        next_features[11] = product_data['Total Quantity'].iloc[-3] if len(product_data) > 2 else 0  # lag_3d
                        next_features[12] = product_data['Total Quantity'].iloc[-7] if len(product_data) > 6 else 0  # lag_7d
                    else:
                        # For subsequent predictions, use the predicted values
                        if i >= 7:
                            next_features[7] = np.mean(predictions[i-7:i])  # rolling_mean_7d
                            next_features[8] = np.std(predictions[i-7:i]) if len(predictions[i-7:i]) > 1 else 0  # rolling_std_7d
                        else:
                            # For the first few days, use a mix of historical and predicted
                            historical = list(product_data['Total Quantity'].iloc[-(7-i):])
                            predicted = predictions[:i]
                            combined = historical + predicted
                            next_features[7] = np.mean(combined)  # rolling_mean_7d
                            next_features[8] = np.std(combined) if len(combined) > 1 else 0  # rolling_std_7d
                        
                        next_features[9] = predictions[i-1]  # lag_1d
                        next_features[10] = predictions[i-2] if i >= 2 else product_data['Total Quantity'].iloc[-1]  # lag_2d
                        next_features[11] = predictions[i-3] if i >= 3 else product_data['Total Quantity'].iloc[-2]  # lag_3d
                        next_features[12] = predictions[i-7] if i >= 7 else product_data['Total Quantity'].iloc[-6]  # lag_7d
                    
                    # Scale the features
                    next_features_scaled = scaler_X.transform(next_features.reshape(1, -1))
                    
                    # Update the sequence for the next iteration
                    current_sequence = np.vstack([current_sequence[1:], next_features_scaled])
                
                # Create a DataFrame with the predictions
                future_df = pd.DataFrame({
                    'sale_date': future_dates,
                    'product_name': product_name,
                    'total_quantity': [max(1, int(round(pred))) for pred in predictions]
                })

                # Append to combined dataframe
                all_predictions_df = pd.concat([all_predictions_df, future_df], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing product '{product_name}': {str(e)}")
                continue
        

        # Make prediction using the loaded model
        # Return error if no predictions were made
        if len(all_predictions_df) == 0:
            return jsonify({"error": "No predictions could be generated for any product"}), 500

        return jsonify({"preds": f"{all_predictions_df}"})
    
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

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port,debug=True)