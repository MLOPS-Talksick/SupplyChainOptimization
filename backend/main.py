# main.py (excerpt)
import os
from datetime import date, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from google.cloud import storage
import pymysql
from google.cloud import aiplatform
import requests
from typing import List
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth
import time
from dotenv import load_dotenv
import sqlalchemy
import pandas as pd
from google.cloud.sql.connector import Connector


# load_dotenv()
app = FastAPI()

# Configuration from environment
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
# Database config
host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")
conn_name = os.getenv("INSTANCE_CONN_NAME")
connector = Connector()
# Vertex AI config
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
VERTEX_REGION = os.environ.get("VERTEX_REGION")
VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
API_TOKEN = os.environ.get("API_TOKEN")  # our simple token for auth
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID")
VM_IP = os.environ.get("VM_IP")
AIRFLOW_URL = f"http://{VM_IP}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_ADMIN_USERNAME")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_ADMIN_PASSWORD")


# Simple token-based authentication dependency
def verify_token(token: str = Header(None)):
    if API_TOKEN is None:
        # If no token is set on server, we could disable auth (not recommended for prod)
        return True
    if token is None or token != API_TOKEN:
        # If the token header is missing or doesn't match, reject the request
        raise HTTPException(status_code=401, detail="Unauthorized: invalid token")
    return True


@app.post("/upload", dependencies=[Depends(verify_token)])
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads an Excel file to GCS and triggers an Airflow DAG run.
    """
    # 1. Upload the file to Google Cloud Storage
    storage_client = storage.Client()  
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(file.filename)
    try:
        # Use upload_from_file to stream the file to GCS
        blob.upload_from_file(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")
    
    # 2. Trigger the Airflow DAG after successful upload
    # Prepare the DAG run payload
    dag_run_id = f"manual_{int(time.time())}"  # e.g., manual_1697059096
    payload = {
        "dag_run_id": dag_run_id,
        "conf": { "filename": file.filename }
    }
    try:
        response = requests.post(
            AIRFLOW_URL,
            json=payload,
            auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
        )
        response.raise_for_status()
    except requests.RequestException as e:
        # If the DAG trigger fails, you might still want to inform the user.
        # Here we raise an error. Alternatively, you could return a success for upload 
        # and a warning for the DAG trigger.
        raise HTTPException(status_code=500, detail=f"Airflow DAG trigger failed: {e}")
    
    # 3. Return a response indicating success
    return {
        "message": "File uploaded to GCS and Airflow DAG triggered successfully.",
        "file": file.filename,
        "dag_run_id": dag_run_id
    }

@app.get("/data", dependencies=[Depends(verify_token)])
def get_data(n: int = 5):
    """Fetch the last n records from the database."""
    import logging
    logging.basicConfig(level=logging.INFO)
    # Validate n
    if n <= 0:
        raise HTTPException(status_code=400, detail="Parameter 'n' must be positive.")
    # Connect to Cloud SQL
    try:
        logging.info("Starting /data endpoint")
        logging.info(f"Conn Name: {conn_name}")
        logging.info(f"User: {user}, DB: {database}")

        def getconn():
            logging.info("Attempting DB connection...")
            conn = connector.connect(
                conn_name,  # Cloud SQL instance connection name
                "pymysql",  # Database driver
                user=user,  # Database user
                password=password,  # Database password
                db=database,
            )
            logging.info("DB connected.")
            return conn

        pool = sqlalchemy.create_engine(
            "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
            creator=getconn,
        )

        db_conn = pool.connect()
        logging.info("DB engine created.")

        result = db_conn.execute(sqlalchemy.text("SELECT NOW();"))
        logging.info("Query executed.")
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    try:
        
        query = f"""
        SELECT 
            sale_date, product_name, total_quantity
        FROM SALES
        ORDER BY sale_date DESC LIMIT {n};"""
        with pool.connect() as db_conn:
            result = db_conn.execute(sqlalchemy.text(query))
            print(result.scalar())
        df = pd.read_sql(query, pool)
    except Exception as e:
        # conn.close()
        logging.error(f"ERROR in /data: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    # Convert rows to list of dicts
    # result = [dict(zip(columns, row)) for row in rows]
    return {"records": df.to_json(), "count": len(df)}




# (Optional) Define a Pydantic model for the request body
class PredictRequest(BaseModel):
    product_names: List[str]
    dates: List[str]

class PredictRequest(BaseModel):
    product_name: str
    days: int

@app.post("/predict")
async def get_prediction(request: PredictRequest):
    # Prepare the payload using the same keys as the working curl example
    payload = {
        "product_name": request.product_name,
        "days": request.days
    }
    try:
        response = requests.post(
            "https://model-serving-148338842941.us-central1.run.app/predict", 
            json=payload
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
    predictions = response.json()
    return {"predictions": predictions}