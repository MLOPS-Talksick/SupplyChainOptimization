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


load_dotenv()
app = FastAPI()

# Configuration from environment
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# Database config
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
INSTANCE_CONN_NAME = os.environ.get("INSTANCE_CONNECTION_NAME")  # project:region:instance
# Vertex AI config
PROJECT_ID = os.environ.get("PROJECT_ID")
VERTEX_REGION = os.environ.get("VERTEX_REGION")
VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
API_TOKEN = os.environ.get("API_TOKEN")  # our simple token for auth
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID")
VM_IP = os.environ.get("VM_IP")
AIRFLOW_URL = f"http://{VM_IP}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD")


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
# async def upload_file(file: UploadFile = File(...)):
#     # 1. Validate file type by extension or MIME type
#     filename = file.filename
#     if not filename:
#         raise HTTPException(status_code=400, detail="No file provided.")
#     # Accept .xls or .xlsx
#     if not (filename.lower().endswith(".xls") or filename.lower().endswith(".xlsx")):
#         raise HTTPException(status_code=400, detail="Only .xls or .xlsx files are allowed.")
#     # Optionally, check MIME type as well for extra safety
#     if file.content_type not in ["application/vnd.ms-excel", 
#                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
#         raise HTTPException(status_code=400, detail="Invalid file type. Must be an Excel file.")

#     # 2. Read file (in chunks to be memory-safe) and check size
#     file_contents = await file.read()  # read the entire file into memory (be careful with very large files)
#     file_size = len(file_contents)
#     max_size = 50 * 1024 * 1024  # 50 MB in bytes
#     if file_size > max_size:
#         raise HTTPException(status_code=400, detail="File too large. Must be <= 50 MB.")
    
#     # 3. Upload to Cloud Storage
#     try:
#         storage_client = storage.Client()  # uses ADC credentials (should be available on Cloud Run)
#         bucket = storage_client.bucket(BUCKET_NAME)
#         blob = bucket.blob(filename)
#         # Upload the file contents
#         blob.upload_from_string(file_contents)
#     except Exception as e:
#         # Log the exception (omitted here) and return error
#         raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    
#     # 4. Return success response
#     return {"message": f"File '{filename}' uploaded successfully to bucket {BUCKET_NAME}.", "size": file_size}

@app.get("/data", dependencies=[Depends(verify_token)])
def get_data(n: int = 5):
    """Fetch the last n records from the database."""
    # Validate n
    if n <= 0:
        raise HTTPException(status_code=400, detail="Parameter 'n' must be positive.")
    # Connect to Cloud SQL
    try:
                

        # Use an environment variable or a flag to switch between connection methods.
        # For example, if you're running locally on Windows, set USE_TCP to 'true'
        # USE_TCP = os.environ.get("USE_TCP", "false").lower() == "true"

        # if USE_TCP:
        #     # Replace with your Cloud SQL instance's public IP and port (default MySQL port is 3306)
        #     connection = pymysql.connect(
        #         host="YOUR_CLOUD_SQL_PUBLIC_IP",
        #         user=os.environ.get("DB_USER"),
        #         password=os.environ.get("DB_PASS"),
        #         database=os.environ.get("DB_NAME"),
        #         port=3306
        #     )
        # else:
        #     # For Cloud Run (or Linux) use the Unix socket
        #     connection = pymysql.connect(
        #         user=os.environ.get("DB_USER"),
        #         password=os.environ.get("DB_PASS"),
        #         database=os.environ.get("DB_NAME"),
        #         unix_socket=f"/cloudsql/{os.environ.get('INSTANCE_CONNECTION_NAME')}"
        #     )

        host = os.getenv("DB_HOST")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        database = os.getenv("DB_NAME")
        conn_name = os.getenv("INSTANCE_CONN_NAME")
        connector = Connector()

        def getconn():
            conn = connector.connect(
                conn_name,  # Cloud SQL instance connection name
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
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    try:
        # with conn.cursor() as cursor:
        #     # Example query: adjust table/columns as needed.
        #     cursor.execute("SELECT * FROM sales_data ORDER BY id DESC LIMIT %s;", (n,))
        #     rows = cursor.fetchall()
        #     # Get column names for constructing dict (cursor.description has info)
        #     columns = [desc[0] for desc in cursor.description]
        # conn.close()
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
