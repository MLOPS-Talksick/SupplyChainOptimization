# ML_Models/scripts/model_health_check.py

import os
import sys
import time
import pickle
import json
import numpy as np
import requests
import smtplib
from email.mime.text import MIMEText
from google.cloud import storage
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
import subprocess

# Set path and load .env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
load_dotenv()

# ENV variables
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
MODEL_FILE_NAME = "model.pkl"
GCP_MODEL_PATH = f"models/{MODEL_FILE_NAME}"
SERVICE_EMAIL = "talksick530@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
DB_CONFIG = {
    'host': os.getenv("MYSQL_HOST"),
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASSWORD"),
    'database': os.getenv("MYSQL_DATABASE")
}

# Check GCP for model.pkl

def model_exists():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return bucket.blob(GCP_MODEL_PATH).exists(client)

# Trigger training script if model not found

def trigger_training():
    subprocess.run(["python", "ML_Models/scripts/train_lstm_model.py"])  # Adjust path if needed

# Load model from pickle

def load_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(GCP_MODEL_PATH)
    blob.download_to_filename("temp_model.pkl")
    with open("temp_model.pkl", "rb") as f:
        return pickle.load(f)

# Health check

def run_health_check():
    if not model_exists():
        trigger_training()
        return

    model_artifacts = load_model()

    y_true = model_artifacts["y_true"]
    y_pred = model_artifacts["y_pred"]
    train_loss = model_artifacts.get("train_loss", None)
    test_loss = model_artifacts.get("test_loss", None)
    hyperparams = model_artifacts["hyperparameters"]
    train_features = model_artifacts.get("train_features", [])
    current_features = model_artifacts.get("latest_input_features", [])

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    p_value = ks_2samp(train_features, current_features).pvalue if train_features and current_features else None

    status = "healthy" if rmse < 15.0 and (p_value is None or p_value > 0.05) else "drift_detected"

    log_stats_to_db(hyperparams, train_loss, test_loss, rmse, mape, p_value, status)

# Send alert

def send_alert_email(message):
    msg = MIMEText(f"Model health check failed or detected drift: {message}")
    msg['Subject'] = 'Model Health Alert - LSTM'
    msg['From'] = SERVICE_EMAIL
    msg['To'] = SERVICE_EMAIL
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(SERVICE_EMAIL, EMAIL_PASSWORD)
        server.send_message(msg)

# Log stats

def log_stats_to_db(params, train_loss, test_loss, rmse, mape, p_value, status):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        sql = """
            INSERT INTO stats (
                timestamp, model_path, units_1, activation_1, dropout_1,
                units_2, activation_2, dropout_2, dense_units, dense_activation,
                optimizer, learning_rate, train_loss, test_loss,
                rmse, mape, p_value_ks, status
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            GCP_MODEL_PATH,
            params['units_1'], params['activation_1'], params['dropout_1'],
            params['units_2'], params['activation_2'], params['dropout_2'],
            params['dense_units'], params['dense_activation'],
            params['optimizer'], params['learning_rate'],
            train_loss, test_loss, rmse, mape, p_value, status
        )
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        send_alert_email(str(e))

if __name__ == "__main__":
    run_health_check()
