import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from google.cloud import storage

import pandas as pd
from google.cloud.sql.connector import Connector
import sqlalchemy
import os

def get_latest_data_from_cloud_sql(query, port="3306"):
    """
    Connects to a Google Cloud SQL instance using TCP (public IP or Cloud SQL Proxy)
    and returns query results as a DataFrame.

    Args:
        host (str): The Cloud SQL instance IP address or localhost (if using Cloud SQL Proxy).
        port (int): The port number (typically 3306 for MySQL).
        user (str): Database username.
        password (str): Database password.
        database (str): Database name.
        query (str): SQL query to execute.

    Returns:
        pd.DataFrame: Query results.
    """

    host = "35.192.172.104"
    user = "shrey"
    password = "shrey"
    database = "combined_transaction_data"
    connector = Connector()

    def getconn():
        conn = connector.connect(
            "primordial-veld-450618-n4:us-central1:mlops-sql",  # Cloud SQL instance connection name
            "pymysql",  # Database driver
            user=user,  # Database user
            password=password,  # Database password
            db=database,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
        creator=getconn,
    )
    # with pool.connect() as db_conn:
        # result = db_conn.execute(sqlalchemy.text(query))
        # print(result.scalar())
    df = pd.read_sql(query, pool)
    # print(df.head())
    connector.close()
    return df


def download_model(bucket_name, model_name):
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/svaru/Downloads/cloud_run.json"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_name)
    blob.download_to_filename(model_name)

    print(f"File {model_name} downloaded.")

    return load_model(os.path.abspath(model_name))



def predict_product_demand(product_name, days=30):
    """
    Generate demand predictions for a specific product using a pre-trained LSTM model.
    
    Parameters:
    -----------
    model_folder : str
        Path to the folder containing the saved model and preprocessors
    product_name : str
        Name of the product to predict
    days : int, default=30
        Number of days to predict into the future
    data_file : str, optional
        Path to the CSV file with historical data. If None, will look for 'processed_data.csv' in model_folder
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with date, product name, and predicted quantity
    """

    # # Load the model
    # model_path = 'lstm_model.keras'
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Model file not found at '{model_path}'")
    
    # model = load_model(model_path)
    # print(f"Model loaded successfully from {model_path}")

    model = download_model("trained-model-1", 'lstm_model.keras')

    model.summary()  # Print model summary for debugging
    
    # Load preprocessors
    scaler_X_path = 'scaler_X.pkl'
    scaler_y_path = 'scaler_y.pkl'
    label_encoder_path = 'label_encoder.pkl'
    
    if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path) or not os.path.exists(label_encoder_path):
        raise FileNotFoundError("Preprocessor files (scalers or encoder) not found in model folder")
    
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
        WHERE product_name = '{product_name}'
        ORDER BY Date;
    """

    df = get_latest_data_from_cloud_sql(query=query)

    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Get the latest date from the DataFrame
    latest_date = df['Date'].max()

    # Filter the DataFrame to get only the last 60 days from the latest date
    df = df[df['Date'] >= (latest_date - pd.Timedelta(days=60))]
    
    # # Load historical data
    # if data_file is None:
    #     data_file = "C:/Users/svaru/Downloads/processed_data.csv"
        
    # if not os.path.exists(data_file):
    #     raise FileNotFoundError(f"Data file not found at '{data_file}'")
    
    # df = pd.read_csv(data_file)
    
    # Convert data types
    df['Total Quantity'] = df['Total Quantity'].astype(int)
    df['Date'] = pd.to_datetime(df['Date'])
    
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
    
    # Encode product names if not already encoded
    if 'product_encoded' not in df.columns:
        df['product_encoded'] = label_encoder.transform(df['Product Name'])
    
    # Define features (must match what was used for training)
    features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 
                'product_encoded', 'rolling_mean_7d', 'rolling_std_7d',
                'lag_1d', 'lag_2d', 'lag_3d', 'lag_7d']
    
    # Get data for the specific product
    try:
        product_idx = label_encoder.transform([product_name])[0]
    except ValueError:
        raise ValueError(f"Product '{product_name}' not found in the training data")
    
    product_data = df[df['product_encoded'] == product_idx].sort_values('Date')
    
    if len(product_data) < 5:  # Need at least time_steps data points (assuming 5 time steps)
        raise ValueError(f"Not enough historical data for product '{product_name}'")
    
    # Get the last date in the dataset
    last_date = product_data['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    
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
        'Date': future_dates,
        'Product_Name': product_name,
        'Predicted_Quantity': [max(1, int(round(pred))) for pred in predictions]
    })
    
    return future_df

# Example usage:
if __name__ == "__main__":
    # Example parameters
    model_folder = "./model_folder"  # Folder with saved model and preprocessors
    product_name = "pizza"   # Product to predict
    days = 7                         # Number of days to predict
    
    try:
        # Generate predictions
        predictions = predict_product_demand(
            product_name=product_name,
            days=days
        )
        
        print("Predictions generated successfully:")
        print(predictions)
        
    except Exception as e:
        print(f"Error generating predictions: {e}")