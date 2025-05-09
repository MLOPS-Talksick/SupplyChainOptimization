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
from pydantic import BaseModel, EmailStr
from requests.auth import HTTPBasicAuth
import time
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import text
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
        headers = {
            "User-Agent": "airflow-client",
            "Content-Type": "application/json"
        }
        response = requests.post(
            AIRFLOW_URL,
            json=payload,
            auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
            headers=headers
        )
        response.raise_for_status()
        logging.info(f"Airflow DAG triggered successfully. Response: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        logging.error(f"Airflow DAG trigger failed: {e}")
        status = getattr(e.response, "status_code", None)
        text = getattr(e.response, "text", None)
        logging.error(f"Airflow response status code: {status}")
        logging.error(f"Airflow response body: {text}")
        raise HTTPException(
            status_code=status or 500,
            detail=f"Airflow DAG trigger failed: {text or str(e)}"
        )


    
    return {
        "message": "File uploaded to GCS and Airflow DAG triggered successfully.",
        "file": file.filename,
        "dag_run_id": dag_run_id
    }

@app.get("/monitoring", dependencies=[Depends(verify_token)])
def get_data(n: int = 5, predictions: bool = False):
    logging.info("Received /monitoring request.")
    
        
    table = "STATS"
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
            *
        FROM {table}
        ORDER BY check_date DESC LIMIT {n};"""
        with pool.connect() as db_conn:
            result = db_conn.execute(sqlalchemy.text(query))
            logging.info("Database query executed. First scalar value: " + str(result.scalar()))
        df = pd.read_sql(query, pool)
        logging.info(f"Data retrieved successfully. Rows count: {len(df)}")
    except Exception as e:
        logging.error(f"Database query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    return {"records": df.to_json(), "count": len(df)}


@app.get("/data", dependencies=[Depends(verify_token)])
def get_data(n: int = 5, predictions: bool = False):
    logging.info("Received /data request: n=%s, predictions=%s", n, predictions)

    # Ensure n is non‑negative
    n = max(0, n)

    # 1) Build connection pool
    try:
        def getconn():
            return connector.connect(
                conn_name,
                "pymysql",
                user=user,
                password=password,
                db=database,
                ip_type="PRIVATE"
            )
        pool = sqlalchemy.create_engine("mysql+pymysql://", creator=getconn)
        logging.info("Database connection pool created successfully.")
    except Exception as e:
        logging.error("Database connection failed: %s", e)
        raise HTTPException(status_code=500, detail="Database connection failed.")

    # 2) Pick table and fetch its max sale_date
    table = "PREDICT" if predictions else "SALES"
    try:
        with pool.connect() as conn:
            max_date = conn.execute(
                text(f"SELECT MAX(sale_date) FROM {table}")
            ).scalar()
    except Exception as e:
        logging.error("Failed to fetch max date from %s: %s", table, e)
        raise HTTPException(status_code=500, detail="Database error fetching date range.")

    if max_date is None:
        # No rows in the table
        logging.info("Table %s is empty; returning zero records.", table)
        return {"records": {}, "count": 0}

    # 3) Compute start_date = max_date - n days
    start_date = max_date - timedelta(days=n)
    logging.info(
        "Date window on %s: from %s through %s",
        table,
        start_date,      # no .date()
        max_date         # no .date()
    )

    # 4) Query rows in that window
    sql = f"""
        SELECT
          sale_date,
          product_name,
          total_quantity
        FROM {table}
        WHERE sale_date >= :start
        ORDER BY sale_date DESC;
    """
    try:
        df = pd.read_sql(
            text(sql),
            pool,
            params={"start": start_date}
        )
        logging.info("Query returned %d rows", len(df))
    except Exception as e:
        logging.error("Database query failed: %s", e)
        raise HTTPException(status_code=500, detail="Database query failed.")

    # 5) Return
    return {
        "records": df.to_json(date_unit="ms"),
        "count": len(df)
    }

@app.get("/get-stats", dependencies=[Depends(verify_token)])
def get_stats():
    logging.info("Received /get-stats request.")
    table = "SALES"
    
    # 1) Connection setup (unchanged)
    try:
        def getconn():
            return connector.connect(
                conn_name,
                "pymysql",
                user=user,
                password=password,
                db=database,
                ip_type="PRIVATE"
            )
        pool = sqlalchemy.create_engine(
            "mysql+pymysql://",
            creator=getconn,
        )
        logging.info("Database connection pool created successfully.")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    
    # 2) STATISTICS query
    try:
        stats_sql = f"""
            SELECT
              MIN(sale_date)               AS start_date,
              MAX(sale_date)               AS end_date,
              COUNT(*)                     AS total_entries,
              COUNT(DISTINCT product_name) AS total_products
            FROM {table};
        """
        with pool.connect() as conn:
            row = conn.execute(sqlalchemy.text(stats_sql)).one()
        
        # Convert dates to ISO strings (or None if empty)
        start_date     = row.start_date.isoformat() if row.start_date else None
        end_date       = row.end_date.isoformat()   if row.end_date   else None
        total_entries  = int(row.total_entries)
        total_products = int(row.total_products)

        logging.info(
            f"Stats → start: {start_date}, end: {end_date}, "
            f"entries: {total_entries}, products: {total_products}"
        )
    except Exception as e:
        logging.error(f"Failed to fetch stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics.")
    
    # 3) Return them in a nice JSON
    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_entries": total_entries,
        "total_products": total_products
    }


class EmailRequest(BaseModel):
    email: EmailStr

@app.post("/upload-email", dependencies=[Depends(verify_token)])
async def upload_email(payload: EmailRequest):
    """Accepts an email and uploads it as a text file to GCS."""
    email_address = payload.email
    file_name = "email.txt"
    bucket_name = "sco-user-email"

    # Basic validation of config
    
    try:
        logging.info("Received request to upload email to GCS bucket '%s'", bucket_name)
        # 1. Create a text file with the email content
        with open(file_name, "w") as f:
            f.write(email_address)
        logging.info("Created file %s with the provided email", file_name)

        # 2. Initialize GCS client and get the bucket
        storage_client = storage.Client()  # uses credentials from env by default&#8203;:contentReference[oaicite:2]{index=2}
        bucket = storage_client.bucket(bucket_name)

        # 3. Delete any existing objects in the bucket (to keep only one file)
        blobs = bucket.list_blobs()
        for blob in blobs:
            bucket.delete_blob(blob.name)
            logging.info("Deleted existing object '%s' from bucket", blob.name)

        # 4. Upload the new file to GCS (as 'email.txt')
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)
        logging.info("Uploaded file to GCS bucket '%s' as object '%s'", bucket_name, file_name)

        return {"detail": "Email file uploaded successfully to GCS."}
    except Exception as e:
        logging.error("Failed to upload email to GCS: %s", e, exc_info=True)
        # Return a generic error to client
        raise HTTPException(status_code=500, detail="Internal server error")

# Prediction endpoint
class PredictRequest(BaseModel):
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
            ip_type = "PRIVATE"
        )

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
    )
    logging.info("Database connection pool for prediction established.")
    return pool
        

@app.post("/predict", dependencies=[Depends(verify_token)])
async def get_prediction(request: PredictRequest):
    days = request.days
    logging.info(f"Received /predict request: days={days}")

    # 1) Compute the date range
    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT MAX(sale_date) FROM SALES;"))
            last_sale = result.scalar()
    except Exception as e:
        logging.error(f"Failed to fetch last sale_date: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching last sale date")

    # If no sales exist yet, start from today
    start_date = last_sale.date() if hasattr(last_sale, "date") else date.today()
    end_date = date.today() + timedelta(days=days)
    logging.info(f"Date range: {start_date} → {end_date}")

    # 2) Check PREDICT table for existing predictions
    try:
        df_existing = pd.read_sql(
            text("SELECT sale_date, product_name, total_quantity FROM PREDICT "
                 "WHERE sale_date BETWEEN :start AND :end ;"),
            engine,
            params={"start": start_date, "end": end_date}
        )
        df_existing["sale_date"] = pd.to_datetime(df_existing["sale_date"]).dt.date
        existing_dates = set(df_existing["sale_date"].unique())
        expected_dates = { start_date + timedelta(i) 
                           for i in range((end_date - start_date).days + 1) }
    except Exception as e:
        logging.error(f"Failed to query PREDICT table: {e}")
        raise HTTPException(status_code=500, detail="Database error checking predictions")

    # 3) If we have every date in the range, return from DB
    if expected_dates.issubset(existing_dates):
        logging.info("All predictions found in PREDICT; returning cached results.")
        records = df_existing.to_dict()
        return {"predictions": records}

    # 4) Otherwise call the model-serving endpoint for fresh predictions
    logging.info("Missing some dates in PREDICT; calling model-api")
    payload = {"days": days}
    try:
        resp = requests.post(f"{MODEL_SERVING_URL}/predict", json=payload)
        if resp.status_code == 500:
            logging.error("Model serving responded 500: Not enough data")
            raise HTTPException(status_code=500, detail="Not enough data for all products")
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Model prediction HTTP error: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed")

    data = resp.json()
    preds = data.get("preds")
    if preds is None:
        logging.error("Model response missing 'preds'")
        raise HTTPException(status_code=500, detail="Invalid model response")

    # 5) Load into DataFrame
    try:
        if isinstance(preds, str):
            df_new = pd.read_json(preds)
        else:
            df_new = pd.DataFrame.from_records(preds)
        df_new["sale_date"] = pd.to_datetime(df_new["sale_date"]).dt.date
        logging.info(f"Received {len(df_new)} new prediction rows")
    except Exception as e:
        logging.error(f"Failed to parse new preds: {e}")
        raise HTTPException(status_code=500, detail="Error parsing predictions")

    # 6) Return the newly fetched predictions
    return {"predictions": df_new.to_dict()}



@app.post("/validate_excel", dependencies=[Depends(verify_token)])
async def validate_excel(file: UploadFile = File(...)):
    logging.info(f"Received /validate_excel request with file: {file.filename}")

    # 1. Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("xls", "xlsx"):
        logging.error("Invalid file type for validate_excel.")
        raise HTTPException(status_code=400, detail="Invalid file type. Only .xls or .xlsx files are allowed.")

    # 2. Enforce 50 MB size limit
    file.file.seek(0, io.SEEK_END)
    if file.file.tell() > 50 * 1024 * 1024:
        logging.error("File too large in validate_excel.")
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 50MB.")
    file.file.seek(0)
    logging.info("File pointer reset successfully in validate_excel.")

    # 3. Read into DataFrame
    try:
        if ext == "xls":
            df = pd.read_excel(file.file, engine="xlrd")
        else:
            df = pd.read_excel(file.file, engine="openpyxl")
        logging.info(f"Excel file read. Rows: {df.shape[0]} Columns: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Failed to read Excel file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read the Excel file. Ensure it is valid .xls or .xlsx.")

    # 4. Canonicalize and validate headers
    def canonicalize(col: str) -> str:
        return col.strip().lower().replace(" ", "_")

    expected_columns = [
        "date", "unit_price", "transaction_id",
        "quantity", "producer_id", "store_location", "product_name"
    ]
    original_cols = df.columns.tolist()
    canon_cols = [canonicalize(c) for c in original_cols]

    if Counter(canon_cols) != Counter(expected_columns):
        logging.error("Excel header validation failed.")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid Excel header. Expected (any order): {expected_columns}. "
                f"Found (canonicalized): {canon_cols}"
            )
        )

    # rename columns in-place
    rename_map = {orig: canonicalize(orig) for orig in original_cols}
    df.rename(columns=rename_map, inplace=True)
    logging.info(f"Columns canonicalized to: {df.columns.tolist()}")

    # 5. Fetch existing products from DB
    try:
        def getconn():
            return connector.connect(
                conn_name,
                "pymysql",
                user=user,
                password=password,
                db=database,
                ip_type="PRIVATE"
            )

        pool = sqlalchemy.create_engine("mysql+pymysql://", creator=getconn)
        db_query = "SELECT DISTINCT product_name FROM PRODUCT;"
        with pool.connect() as conn:
            result = conn.execute(sqlalchemy.text(db_query))
            raw_db = {row[0] for row in result if row[0] is not None}

        # normalize DB names
        db_products = {p.strip().lower() for p in raw_db}
        logging.info(f"DB products (normalized): {db_products}")
    except Exception as e:
        logging.error(f"Database query failed: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching products.")

    # 6. Normalize Excel product names
    raw_excel = set(df["product_name"].dropna().astype(str).tolist())
    excel_products = {p.strip().lower() for p in raw_excel}
    logging.info(f"Excel products (normalized): {excel_products}")

    # 7. Compute new products
    missing = excel_products - db_products
    new_products = [p for p in raw_excel if p.strip().lower() in missing]
    logging.info(f"Identified {len(new_products)} new products: {new_products}")

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
