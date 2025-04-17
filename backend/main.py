import os
from datetime import date, timedelta, datetime
from dateutil import parser 
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Query, APIRouter
from fastapi.responses import JSONResponse
from google.cloud import storage, scheduler_v1, aiplatform
from google.protobuf import field_mask_pb2
import pymysql
import requests
from typing import List
from pydantic import BaseModel
from requests.auth import HTTPBasicAuth
import time
from dotenv import load_dotenv
import sqlalchemy
import pandas as pd
from google.cloud.sql.connector import Connector
import io  # Needed for file pointer operations
from collections import Counter
import json
import logging
from google.cloud.scheduler_v1.types import RetryConfig
from google.cloud.scheduler_v1.types import Job, HttpTarget  # ensure we have these types
from google.protobuf import duration_pb2


# Configure logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
app = FastAPI()

# Configuration from environment
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
logging.info(f"GCS_BUCKET_NAME: {BUCKET_NAME}")

# Database config
host = os.getenv("MYSQL_HOST")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")
conn_name = os.getenv("INSTANCE_CONN_NAME")
connector = Connector()
logging.info("Database configuration loaded.")

# Vertex AI config
PROJECT_ID = os.environ.get("PROJECT_ID")
VERTEX_REGION = os.environ.get("VERTEX_REGION")
# VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
API_TOKEN = os.environ.get("API_TOKEN")  # our simple token for auth
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID")
VM_IP = os.environ.get("VM_IP")
AIRFLOW_URL = f"http://{VM_IP}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
# AIRFLOW_URL = os.environ.get("AIRFLOW_URL")
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_ADMIN_USERNAME")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_ADMIN_PASSWORD")
logging.info("Airflow configuration loaded.")
logging.info(f"AIRFLOW_URL: {AIRFLOW_URL}")

# Serving
MODEL_SERVING_URL = os.environ.get("MODEL_SERVING_URL")
logging.info("Model serving URL loaded.")

# Simple token-based authentication dependency
def verify_token(token: str = Header(None)):
    if API_TOKEN is None:
        logging.warning("No API_TOKEN set on server; skipping token verification.")
        return True
    if token is None or token != API_TOKEN:
        logging.error("Invalid or missing token in request.")
        raise HTTPException(status_code=401, detail="Unauthorized: invalid token")
    logging.info("Token verification passed.")
    return True


@app.post("/upload", dependencies=[Depends(verify_token)])
async def upload_file(
    file: UploadFile = File(...),
    deny_list: str = Query(None, description="Comma-separated product names to remove"),
    rename_dict: str = Query(None, description='JSON dict of {"oldName":"newName"} pairs')
):
    logging.info(f"Received /upload request with file: {file.filename}")

    # Parse and validate query parameters
    deny_items = []
    if deny_list:
        deny_items = [name.strip() for name in deny_list.split(",") if name.strip()]
    logging.info(f"Parsed deny_list: {deny_items}")

    rename_map = {}
    if rename_dict:
        try:
            rename_map = json.loads(rename_dict)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse rename_dict: {e}")
            raise HTTPException(status_code=400, detail="rename_dict is not valid JSON")
        if not isinstance(rename_map, dict):
            logging.error("rename_dict is not a dict.")
            raise HTTPException(status_code=400, detail="rename_dict must be a JSON object (dict)")
    logging.info(f"Parsed rename_dict: {rename_map}")

    # Validate file extension
    filename = file.filename
    if not filename or not filename.lower().endswith((".xls", ".xlsx")):
        logging.error("Invalid file format received.")
        raise HTTPException(status_code=400, detail="Invalid file format. Only .xls or .xlsx files are supported.")
    
    # Read Excel file into DataFrame
    try:
        if filename.lower().endswith(".xls"):
            df = pd.read_excel(file.file, engine="xlrd")
        else:
            df = pd.read_excel(file.file, engine="openpyxl")
        logging.info(f"Excel file read successfully. Rows: {df.shape[0]}, Columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Failed to read Excel file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read Excel file. Ensure the file is a valid .xls or .xlsx.")

    # (Optional duplicate read removed)
    # Canonicalize column names: lowercase, strip, replace spaces with underscores
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    logging.info(f"Canonicalized columns: {df.columns.tolist()}")

    # Validate presence of 'product_name'
    if "product_name" not in df.columns:
        logging.error("Missing 'product_name' column in Excel file.")
        raise HTTPException(status_code=400, detail="Missing 'product_name' column in the Excel file.")
    
    # Drop rows with denied product names
    if deny_items:
        original_count = len(df)
        df = df.loc[~df['product_name'].isin(deny_items)].copy()
        logging.info(f"Dropped {original_count - len(df)} rows based on deny_list.")

    # Rename product names as per mapping
    if rename_map:
        df['product_name'].replace(rename_map, inplace=True)
        logging.info(f"Renamed product names as per mapping: {rename_map}")

    # Save the modified DataFrame to a new Excel file in memory
    output_buffer = io.BytesIO()
    try:
        if filename.lower().endswith(".xls"):
            df.to_excel(output_buffer, index=False, engine="xlwt")
            content_type = "application/vnd.ms-excel"
        else:
            df.to_excel(output_buffer, index=False, engine="openpyxl")
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        logging.info("Modified Excel file written to memory.")
    except Exception as e:
        logging.error(f"Error writing DataFrame to Excel format: {e}")
        raise HTTPException(status_code=500, detail="Error writing DataFrame to Excel format.")
    output_buffer.seek(0)

    # Upload the file to Google Cloud Storage
    logging.info(f"Uploading file to GCS bucket: {GCS_BUCKET_NAME}")
    storage_client = storage.Client()  
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(file.filename)
    try:
        # Reset the original file pointer (if needed) or use the in-memory file.
        # In this example, we'll upload the original file. 
        file.file.seek(0)
        blob.upload_from_file(file.file, content_type=content_type)
        logging.info("File successfully uploaded to GCS.")
    except Exception as e:
        logging.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")
    
    # Trigger the Airflow DAG after successful upload
    dag_run_id = f"manual_{int(time.time())}"
    payload = {
        "dag_run_id": dag_run_id,
        "conf": { "filename": file.filename }
    }
    logging.info(f"Triggering Airflow DAG with payload: {payload}")
    try:
        logging.info(f"Airflow URL: {AIRFLOW_URL}")
        logging.info(f"Airflow USERNAME: {AIRFLOW_USERNAME}")
        logging.info(f"Airflow PASSWORD: {AIRFLOW_PASSWORD}")
        response = requests.post(
            AIRFLOW_URL,
            json=payload,
            auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
        )
        response.raise_for_status()
        logging.info("Airflow DAG triggered successfully.")
    except requests.RequestException as e:
        logging.error(f"Airflow DAG trigger failed: {e}")
        raise HTTPException(status_code=500, detail=f"Airflow DAG trigger failed: {e}")
    
    return {
        "message": "File uploaded to GCS and Airflow DAG triggered successfully.",
        "file": file.filename,
        "dag_run_id": dag_run_id
    }


@app.get("/data", dependencies=[Depends(verify_token)])
def get_data(n: int = 5, predictions: bool = False):
    logging.info("Received /data request.")
    if predictions:
        table = "PREDS"
    else:
        table = "SALES"
    n = max(0, n)
    try:
        def getconn():
            conn = connector.connect(
                conn_name,      # Cloud SQL instance connection name
                "pymysql",      # Database driver
                user=user,      # Database user
                password=password,  # Database password
                db=database,    # Database name
                ip_type="PRIVATE"
            )
            return conn

        pool = sqlalchemy.create_engine(
            "mysql+pymysql://",
            creator=getconn,
        )
        logging.info("Database connection pool created successfully.")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    try:
        query = f"""
        SELECT 
            sale_date, product_name, total_quantity
        FROM {table}
        ORDER BY sale_date DESC LIMIT {n};"""
        with pool.connect() as db_conn:
            result = db_conn.execute(sqlalchemy.text(query))
            logging.info("Database query executed. First scalar value: " + str(result.scalar()))
        df = pd.read_sql(query, pool)
        logging.info(f"Data retrieved successfully. Rows count: {len(df)}")
    except Exception as e:
        logging.error(f"Database query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    return {"records": df.to_json(), "count": len(df)}


# Prediction endpoint
class PredictRequest(BaseModel):
    product_name: str
    days: int

def get_db_connection() -> sqlalchemy.engine.base.Engine:
    db_user = user
    db_pass = password
    db_name = database
    instance_connection_name = conn_name
    ip_type = "PRIVATE"  # Always use PRIVATE for production
    
    # initialize Cloud SQL Python Connector object
    conn = Connector(ip_type=ip_type, refresh_strategy="LAZY")

    def getconn() -> pymysql.connections.Connection:
        return conn.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
    )
    logging.info("Database connection pool for prediction established.")
    return pool

def upsert_df(df: pd.DataFrame, engine):
    data = df.to_dict(orient='records')
    columns = df.columns.tolist()
    col_names = ", ".join(columns)
    placeholders = ", ".join(":" + col for col in columns)
    update_clause = ", ".join(f"{col} = VALUES({col})" for col in columns)
    sql = sqlalchemy.text(
        f"INSERT INTO PREDS ({col_names}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )
    with engine.begin() as conn:
        conn.execute(sql, data)
    logging.info("Data upserted into PREDS table successfully.")

@app.post("/predict")
async def get_prediction(request: PredictRequest):
    logging.info(f"Received /predict request for product: {request.product_name}")
    payload = {
        "product_name": request.product_name,
        "days": request.days
    }
    try:
        response = requests.post(
            f"{MODEL_SERVING_URL}/predict", 
            json=payload
        )
        response.raise_for_status()
        logging.info("Model serving response received successfully.")
    except requests.RequestException as e:
        logging.error(f"Model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
    
    predictions = response.json().get('preds')
    if predictions is None:
        logging.error("Response from model serving did not include 'preds'.")
        raise HTTPException(status_code=500, detail="Invalid response from model serving.")
    
    try:
        engine = get_db_connection()
        upsert_df(predictions, engine)
        logging.info("Predictions upserted into database.")
        return {"Success": "True, Uploaded to DB"}
    except Exception as e:
        logging.error(f"Database upload failed: {e}")
        return {"Success": "False, DB upload Failed"}

@app.post("/validate_excel", dependencies=[Depends(verify_token)])
async def validate_excel(file: UploadFile = File(...)):
    logging.info(f"Received /validate_excel request with file: {file.filename}")
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ("xls", "xlsx"):
        logging.error("Invalid file type for validate_excel.")
        raise HTTPException(status_code=400, detail="Invalid file type. Only .xls or .xlsx files are allowed.")
    
    file.file.seek(0, io.SEEK_END)
    file_size = file.file.tell()
    if file_size > 50 * 1024 * 1024:
        logging.error("File too large in validate_excel.")
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 50MB.")
    file.file.seek(0)
    logging.info("File pointer reset successfully in validate_excel.")
    
    try:
        if ext == "xls":
            df = pd.read_excel(file.file, engine="xlrd")
        else:
            df = pd.read_excel(file.file, engine="openpyxl")
        logging.info(f"Excel file read in validate_excel. Rows: {df.shape[0]}")
    except Exception as e:
        logging.error(f"Failed to read Excel file in validate_excel: {e}")
        raise HTTPException(status_code=400, detail="Failed to read the Excel file. Ensure it is a valid .xls or .xlsx file.")
    
    def canonicalize(col):
        return col.strip().lower().replace(" ", "_")
    expected_columns = ['date', 'unit_price', 'transaction_id', 'quantity', 'producer_id', 'store_location', 'product_name']
    actual_columns_original = df.columns.tolist()
    actual_canonical = [canonicalize(col) for col in actual_columns_original]
    
    if Counter(actual_canonical) != Counter(expected_columns):
        logging.error("Excel header validation failed in validate_excel.")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid Excel header. Expected columns (any order): {expected_columns}. Found (canonicalized): {actual_canonical}"
        )
    
    mapping = {orig: canonicalize(orig) for orig in actual_columns_original}
    df.rename(columns=mapping, inplace=True)
    logging.info("Excel columns canonicalized successfully in validate_excel.")
    
    try:
        def getconn():
            conn = connector.connect(
                conn_name,
                "pymysql",
                user=user,
                password=password,
                db=database,
            )
            return conn

        pool = sqlalchemy.create_engine(
            "mysql+pymysql://",
            creator=getconn,
        )
        db_query = "SELECT DISTINCT product_name FROM SALES"
        with pool.connect() as conn:
            result = conn.execute(sqlalchemy.text(db_query))
            db_products = {row[0] for row in result}
        logging.info(f"Fetched {len(db_products)} unique product names from database.")
    except Exception as e:
        logging.error(f"Database query in validate_excel failed: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred while fetching products.")

    excel_products = set(df["product_name"].dropna().unique().tolist())
    new_products = list(excel_products.difference(db_products))
    logging.info(f"Identified {len(new_products)} new products in Excel file.")
    
    return {"new_products": new_products}


@app.post("/update-cron-time", tags=["Scheduler"], dependencies=[Depends(verify_token)])
async def update_cron_time(datetime: str):
    logging.info(f"Received /update-cron-time request with datetime: {datetime}")
    if not datetime:
        logging.error("Missing 'datetime' query parameter in update-cron-time.")
        raise HTTPException(status_code=400, detail="Missing 'datetime' query parameter.")
    try:
        parsed_dt = parser.parse(datetime)
        logging.info(f"Parsed datetime: {parsed_dt}")
    except Exception as e:
        logging.error(f"Failed to parse datetime: {e}")
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO 8601 or a common date/time string.")
    
    cron_schedule = f"{parsed_dt.minute} {parsed_dt.hour} * * *"
    logging.info(f"Converted datetime to cron expression: {cron_schedule}")
    
    try:
        update_scheduler_job(
            project_id=PROJECT_ID,
            location_id=VERTEX_REGION,
            job_id='lstm-health-check-job',
            schedule=cron_schedule,
        )
        logging.info("Scheduler job update invoked successfully.")
    except Exception as e:
        logging.error(f"Failed to update scheduler job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update scheduler job: {e}")
    
    return {"message": f"Schedule updated to '{cron_schedule}' successfully."}


def update_scheduler_job(
    project_id,
    location_id,
    job_id,
    schedule=None,
    time_zone=None,
    http_method=None,
    url=None,
    service_account_email=None,
    headers=None,
    body=None,
    retry_attempts=None,
    retry_min_backoff=None,
    retry_max_backoff=None,
    max_retry_duration=None
):
    logging.info(f"Updating scheduler job: {job_id} in project {project_id}, location {location_id}")
    client = scheduler_v1.CloudSchedulerClient()
    job_name = f"projects/{project_id}/locations/{location_id}/jobs/{job_id}"
    logging.info(f"Job resource name: {job_name}")
    current_job = client.get_job(name=job_name)
    update_mask = []
    
    updated_job = Job(name=job_name)
    
    if schedule:
        updated_job.schedule = schedule
        update_mask.append("schedule")
        logging.info(f"Updated schedule to: {schedule}")
    
    if time_zone:
        updated_job.time_zone = time_zone
        update_mask.append("time_zone")
        logging.info(f"Updated time_zone to: {time_zone}")
    
    if url or http_method or service_account_email or headers or body:
        updated_job.http_target = HttpTarget()
        if not url:
            updated_job.http_target.uri = current_job.http_target.uri
        else:
            updated_job.http_target.uri = url
            update_mask.append("http_target.uri")
        if not http_method:
            updated_job.http_target.http_method = current_job.http_target.http_method
        else:
            updated_job.http_target.http_method = http_method
            update_mask.append("http_target.http_method")
        if service_account_email:
            updated_job.http_target.oidc_token.service_account_email = service_account_email
            updated_job.http_target.oidc_token.audience = url or current_job.http_target.uri
            update_mask.append("http_target.oidc_token.service_account_email")
            update_mask.append("http_target.oidc_token.audience")
        elif hasattr(current_job.http_target, 'oidc_token') and current_job.http_target.oidc_token.service_account_email:
            updated_job.http_target.oidc_token.service_account_email = current_job.http_target.oidc_token.service_account_email
            updated_job.http_target.oidc_token.audience = current_job.http_target.oidc_token.audience
        if headers:
            for key, value in headers.items():
                updated_job.http_target.headers[key] = value
            update_mask.append("http_target.headers")
        else:
            for key, value in current_job.http_target.headers.items():
                updated_job.http_target.headers[key] = value
        if body:
            updated_job.http_target.body = body
            update_mask.append("http_target.body")
        elif current_job.http_target.body:
            updated_job.http_target.body = current_job.http_target.body
    if any([retry_attempts is not None, retry_min_backoff is not None, 
            retry_max_backoff is not None, max_retry_duration is not None]):
        
        updated_job.retry_config = RetryConfig()
        if retry_attempts is not None:
            updated_job.retry_config.retry_count = retry_attempts
            update_mask.append("retry_config.retry_count")
        else:
            updated_job.retry_config.retry_count = current_job.retry_config.retry_count
        if retry_min_backoff is not None:
            
            updated_job.retry_config.min_backoff_duration = duration_pb2.Duration(seconds=retry_min_backoff)
            update_mask.append("retry_config.min_backoff_duration")
        else:
            updated_job.retry_config.min_backoff_duration = current_job.retry_config.min_backoff_duration
        if retry_max_backoff is not None:
            
            updated_job.retry_config.max_backoff_duration = duration_pb2.Duration(seconds=retry_max_backoff)
            update_mask.append("retry_config.max_backoff_duration")
        else:
            updated_job.retry_config.max_backoff_duration = current_job.retry_config.max_backoff_duration
        if max_retry_duration is not None:
            
            updated_job.retry_config.max_retry_duration = duration_pb2.Duration(seconds=max_retry_duration)
            update_mask.append("retry_config.max_retry_duration")
        else:
            updated_job.retry_config.max_retry_duration = current_job.retry_config.max_retry_duration
    
    logging.info(f"Update mask: {update_mask}")
    result = client.update_job(
        request={
            "job": updated_job,
            "update_mask": {"paths": update_mask}
        }
    )
    logging.info(f"Updated scheduler job: {result.name}")
    return result