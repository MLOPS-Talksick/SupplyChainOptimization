# scripts/health_check.py

import os
import requests
import time
import json
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

MODEL_SERVING_URL = os.getenv("MODEL_SERVING_URL")
SERVICE_EMAIL = "talksick530@gmail.com"
RMSE_THRESHOLD = 15.0
HEALTH_CHECK_INTERVAL = 3600  # Limit checks to once per hour

last_health_check_time = 0

# Dummy function for RMSE check (replace with real logic)
def perform_health_check():
    # Send a dummy request to model and analyze response
    dummy_input = {
        "product": "example_product",
        "days_to_predict": 7
    }

    try:
        response = requests.post(f"{MODEL_SERVING_URL}/predict", json=dummy_input)
        if response.status_code != 200:
            raise Exception("Model prediction failed")

        predictions = response.json().get("predictions", [])
        if not predictions:
            raise Exception("Empty predictions returned")

        # Example: Fake RMSE check (substitute with actual RMSE logic if needed)
        rmse = 4.5  # Replace with real computation
        if rmse > RMSE_THRESHOLD:
            send_alert_email(rmse)

    except Exception as e:
        send_alert_email(str(e))


def send_alert_email(message):
    msg = MIMEText(f"Model health check failed: {message}")
    msg['Subject'] = '🚨 Model Health Alert - LSTM Model'
    msg['From'] = SERVICE_EMAIL
    msg['To'] = SERVICE_EMAIL

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(SERVICE_EMAIL, os.getenv("EMAIL_PASSWORD"))
        server.send_message(msg)


# Update your FastAPI backend/main.py to import from scripts.health_check
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from scripts.health_check import perform_health_check, last_health_check_time, HEALTH_CHECK_INTERVAL

app = FastAPI()

@app.post("/predict")
def predict(request: Request, background_tasks: BackgroundTasks):
    # your existing prediction logic goes here
    result = some_predict_function()

    # run async health check in background
    global last_health_check_time
    if time.time() - last_health_check_time > HEALTH_CHECK_INTERVAL:
        background_tasks.add_task(perform_health_check)
        last_health_check_time = time.time()

    return result