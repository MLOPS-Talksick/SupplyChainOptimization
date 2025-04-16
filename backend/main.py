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

load_dotenv()
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
# VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
API_TOKEN = os.environ.get("API_TOKEN")  # our simple token for auth
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID")
VM_IP = os.environ.get("VM_IP")
AIRFLOW_URL = f"http://{VM_IP}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD")

# Serving
MODEL_SERVING_URL = os.environ.get("MODEL_SERVING_URL")


# Simple token-based authentication dependency
def verify_token(token: str = Header(None)):
    if API_TOKEN is None:
        # If no token is set on server, we could disable auth (not recommended for prod)
        return True
    if token is None or token != API_TOKEN:
        # If the token header is missing or doesn't match, reject the request
        raise HTTPException(status_code=401, detail="Unauthorized: invalid token")
    return True


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    deny_list: str = Query(None, description="Comma-separated product names to remove"),
    rename_dict: str = Query(None, description='JSON dict of {"oldName":"newName"} pairs')
):
    # Parse and validate query parameters
    deny_items = []
    if deny_list:
        deny_items = [name.strip() for name in deny_list.split(",") if name.strip()]
    rename_map = {}
    if rename_dict:
        try:
            rename_map = json.loads(rename_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="rename_dict is not valid JSON")
        if not isinstance(rename_map, dict):
            raise HTTPException(status_code=400, detail="rename_dict must be a JSON object (dict)")
    
    # Validate file extension
    filename = file.filename
    if not filename or not filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(status_code=400, detail="Invalid file format. Only .xls or .xlsx files are supported.")
    
    # Read Excel file into DataFrame
    try:
        if filename.lower().endswith(".xls"):
            df = pd.read_excel(file.file, engine="xlrd")
        else:
            df = pd.read_excel(file.file, engine="openpyxl")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read Excel file. Ensure the file is a valid .xls or .xlsx.")
    try:
        if filename.lower().endswith(".xls"):
            df = pd.read_excel(file.file, engine="xlrd")
        else:
            df = pd.read_excel(file.file, engine="openpyxl")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read Excel file. Ensure the file is a valid .xls or .xlsx.")

    # Canonicalize column names: lowercase, strip, replace spaces with underscores
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Now validate presence of 'product_name'
    if "product_name" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'product_name' column in the Excel file.")
    # Validate required column
    if "product_name" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing 'product_name' column in the Excel file.")
    
    # Drop rows with denied product names
    if deny_items:
        df = df.loc[~df['product_name'].isin(deny_items)].copy()
    
    # Rename product names as per mapping
    if rename_map:
        df['product_name'].replace(rename_map, inplace=True)
    
    # Save the modified DataFrame to a new Excel file in memory
    output_buffer = io.BytesIO()
    try:
        if filename.lower().endswith(".xls"):
            df.to_excel(output_buffer, index=False, engine="xlwt")
            content_type = "application/vnd.ms-excel"
        else:
            df.to_excel(output_buffer, index=False, engine="openpyxl")
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error writing DataFrame to Excel format.")
    output_buffer.seek(0)    # 1. Upload the file to Google Cloud Storage
    storage_client = storage.Client()  
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(file.filename)
    try:
        # Use upload_from_file to stream the file to GCS
        file.file.seek(0)
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
        raise HTTPException(status_code=500, detail=f"Airflow DAG trigger failed: {e}")
    
    return {
        "message": "File uploaded to GCS and Airflow DAG triggered successfully.",
        "file": file.filename,
        "dag_run_id": dag_run_id
    }




@app.get("/data", dependencies=[Depends(verify_token)])
def get_data(n: int = 5, predictions: bool = False):
    """Fetch the last n records from the database."""
    if predictions:
        table = "PREDS"
    else:
        table = "SALES"
    n = max(0,n)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    try:
        query = f"""
        SELECT 
            sale_date, product_name, total_quantity
        FROM {table}
        ORDER BY sale_date DESC LIMIT {n};"""
        with pool.connect() as db_conn:
            result = db_conn.execute(sqlalchemy.text(query))
            print(result.scalar())
        df = pd.read_sql(query, pool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    return {"records": df.to_json(), "count": len(df)}


# Prediction endpoints
class PredictRequest(BaseModel):
    product_name: str
    days: int

def get_db_connection() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    db_user = user
    db_pass = password
    db_name = database
    instance_connection_name = conn_name
    
    # ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
    ip_type = IPTypes.PRIVATE

    # initialize Cloud SQL Python Connector object
    connector = Connector(ip_type=ip_type, refresh_strategy="LAZY")

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool


def upsert_df(df: pd.DataFrame, engine):
    """
    Inserts or updates rows in a MySQL table based on duplicate keys.
    If a record with the same primary key exists, it will be replaced with the new record.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to insert/update.
        table_name (str): The target table name.
        engine: SQLAlchemy engine.
    """
    # Convert DataFrame to a list of dictionaries (each dict represents a row)
    data = df.to_dict(orient='records')
    
    # Build dynamic column list and named placeholders
    columns = df.columns.tolist()
    col_names = ", ".join(columns)
    placeholders = ", ".join(":" + col for col in columns)
    
    # Build the update clause to update every column with its new value
    update_clause = ", ".join(f"{col} = VALUES({col})" for col in columns)
    
    # Construct the SQL query using ON DUPLICATE KEY UPDATE
    sql = text(
        f"INSERT INTO PREDS ({col_names}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )
    
    # Execute the query in a transactional scope
    with engine.begin() as conn:
        conn.execute(sql, data)

@app.post("/predict")
async def get_prediction(request: PredictRequest):
    payload = {
        "product_name": request.product_name,
        "days": request.days
        "product_name": request.product_name,
        "days": request.days
    }
    try:
        response = requests.post(
            f"{MODEL_SERVING_URL}/predict", 
            json=payload
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
    predictions = response['preds']
    try:
        upsert_df(predictions,get_db_connection())
        return {"Success": "True, Uploaded to DB"}
    except:
        return {"Success": "False, DB upload Failed"}



@app.post("/validate_excel", dependencies=[Depends(verify_token)])
async def validate_excel(file: UploadFile = File(...)):
    """
    Validates an uploaded Excel file. This endpoint:
    1. Checks that the file is .xls or .xlsx and under 50MB.
    2. Reads the Excel file and verifies that the headers (ignoring case and spacing)
       exactly match: sale_date, product_name, total_quantity.
    3. Renames columns to canonical names (all lowercase, spaces replaced with underscores).
    4. Queries the SQL database to obtain unique product names from the SALES table.
    5. Compares the product names from the Excel file with those in the database,
       and returns any new product names.
    """
    # 1. Validate file extension.
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ("xls", "xlsx"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .xls or .xlsx files are allowed.")
    
    # 2. Enforce file size limit (50MB)
    file.file.seek(0, io.SEEK_END)
    file_size = file.file.tell()
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 50MB.")
    file.file.seek(0)  # Reset pointer to beginning

    # 3. Read the Excel file into a DataFrame using the appropriate engine.
    try:
        if ext == "xls":
            df = pd.read_excel(file.file, engine="xlrd")
        else:  # .xlsx
            df = pd.read_excel(file.file, engine="openpyxl")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read the Excel file. Ensure it is a valid .xls or .xlsx file.")

    # 4. Canonicalize column names and validate headers.
    def canonicalize(col):
        return col.strip().lower().replace(" ", "_")

    expected_columns = ['date', 'unit_price', 'transaction_id', 'quantity', 'producer_id', 'store_location', 'product_name']
    actual_columns_original = df.columns.tolist()
    actual_canonical = [canonicalize(col) for col in actual_columns_original]
    
    # Use Counter to check counts and content regardless of order.
    if Counter(actual_canonical) != Counter(expected_columns):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid Excel header. Expected columns (any order): {expected_columns}. Found (canonicalized): {actual_canonical}"
        )
    
    # Rename DataFrame columns to the canonical names.
    mapping = {orig: canonicalize(orig) for orig in actual_columns_original}
    df.rename(columns=mapping, inplace=True)

    # 5. Query the database for existing unique product names.
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
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database error occurred while fetching products.")

    # 6. Extract product names from the Excel (using the canonical column name).
    # This will work because we renamed the columns.
    excel_products = set(df["product_name"].dropna().unique().tolist())
    new_products = list(excel_products.difference(db_products))

    # 7. Return the list of new product names.
    return {"new_products": new_products}

@router.post("/update-cron-time", tags=["Scheduler"], dependencies=[Depends(verify_token)])
async def update_cron_time(datetime: str):
    """
    Update the Cloud Scheduler job 'my-cloud-run-job' to a new daily cron schedule 
    derived from the given datetime.
    """
    # 1. Validate and parse the datetime string
    if not datetime:
        raise HTTPException(status_code=400, detail="Missing 'datetime' query parameter.")
    try:
        # Parse the datetime using dateutil for flexibility
        parsed_dt = parser.parse(datetime)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO 8601 or a common date/time string.")
    
    # 2. Convert parsed datetime to cron expression "M H * * *"
    cron_schedule = f"{parsed_dt.minute} {parsed_dt.hour} * * *"
    

    # Build the fully qualified job name
    update_scheduler_job(
        project_id=PROJECT_ID,
        location_id=VERTEX_REGION,
        job_id='lstm-health-check-job',
        schedule=cron_schedule,
    )
    # 5. Return success message with the new cron expression
    return {"message": f"Schedule updated to '{cron_schedule}' successfully."}


    # helper function
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
    """
    Updates an existing Cloud Scheduler job.
    
    Args:
        project_id (str): GCP project ID
        location_id (str): Region where the job is located
        job_id (str): Unique identifier for the job
        schedule (str, optional): New cron expression for the schedule
        time_zone (str, optional): New time zone for the schedule
        http_method (str, optional): New HTTP method to use
        url (str, optional): New URL of the Cloud Run function
        service_account_email (str, optional): New service account email
        headers (dict, optional): New HTTP headers
        body (bytes, optional): New request body data
        retry_attempts (int, optional): New maximum number of retry attempts
        retry_min_backoff (int, optional): New minimum backoff time in seconds
        retry_max_backoff (int, optional): New maximum backoff time in seconds
        max_retry_duration (int, optional): New maximum retry duration in seconds
        
    Returns:
        Job: The updated Cloud Scheduler job
    """
    # Initialize the Cloud Scheduler client
    client = scheduler_v1.CloudSchedulerClient()
    
    # Construct the job resource name
    job_name = f"projects/{project_id}/locations/{location_id}/jobs/{job_id}"
    
    # Get the current job configuration
    current_job = client.get_job(name=job_name)
    
    # Use update_mask to specify which fields to update
    update_mask = []
    
    # Create job object with the same name
    updated_job = Job(name=job_name)
    
    # Update schedule if provided
    if schedule:
        updated_job.schedule = schedule
        update_mask.append("schedule")
    
    # Update time zone if provided
    if time_zone:
        updated_job.time_zone = time_zone
        update_mask.append("time_zone")
    
    # Update HTTP target properties if provided
    if url or http_method or service_account_email or headers or body:
        updated_job.http_target = HttpTarget()
        
        # Preserve existing values for fields we're not updating
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
    
    result = client.update_job(
        request={
            "job": updated_job,
            "update_mask": {"paths": update_mask}
        }
    )
    
    print(f"Updated scheduler job: {result.name}")
    return result


