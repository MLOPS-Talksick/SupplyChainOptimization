# # main.py (excerpt)
# import os
# from datetime import date, timedelta
# from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
# from fastapi.responses import JSONResponse
# from google.cloud import storage
# import pymysql
# from google.cloud import aiplatform
# import requests
# from typing import List
# from pydantic import BaseModel
# from requests.auth import HTTPBasicAuth
# import time
# from dotenv import load_dotenv
# import sqlalchemy
# import pandas as pd
# from google.cloud.sql.connector import Connector


# load_dotenv()
# app = FastAPI()

# # Configuration from environment
# BUCKET_NAME = os.environ.get("BUCKET_NAME")
# # Database config
# host = os.getenv("DB_HOST")
# user = os.getenv("DB_USER")
# password = os.getenv("DB_PASS")
# database = os.getenv("DB_NAME")
# conn_name = os.getenv("INSTANCE_CONN_NAME")
# connector = Connector()
# # Vertex AI config
# PROJECT_ID = os.environ.get("PROJECT_ID")
# VERTEX_REGION = os.environ.get("VERTEX_REGION")
# VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
# API_TOKEN = os.environ.get("API_TOKEN")  # our simple token for auth
# GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
# AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID")
# VM_IP = os.environ.get("VM_IP")
# AIRFLOW_URL = f"http://{VM_IP}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
# AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME")
# AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD")


# # Simple token-based authentication dependency
# def verify_token(token: str = Header(None)):
#     if API_TOKEN is None:
#         # If no token is set on server, we could disable auth (not recommended for prod)
#         return True
#     if token is None or token != API_TOKEN:
#         # If the token header is missing or doesn't match, reject the request
#         raise HTTPException(status_code=401, detail="Unauthorized: invalid token")
#     return True


# @app.post("/upload", dependencies=[Depends(verify_token)])
# async def upload_file(file: UploadFile = File(...)):
#     """
#     Uploads an Excel file to GCS and triggers an Airflow DAG run.
#     """
#     # 1. Upload the file to Google Cloud Storage
#     storage_client = storage.Client()  
#     bucket = storage_client.bucket(GCS_BUCKET_NAME)
#     blob = bucket.blob(file.filename)
#     try:
#         # Use upload_from_file to stream the file to GCS
#         blob.upload_from_file(file.file)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"File upload failed: {e}")
    
#     # 2. Trigger the Airflow DAG after successful upload
#     # Prepare the DAG run payload
#     dag_run_id = f"manual_{int(time.time())}"  # e.g., manual_1697059096
#     payload = {
#         "dag_run_id": dag_run_id,
#         "conf": { "filename": file.filename }
#     }
#     try:
#         response = requests.post(
#             AIRFLOW_URL,
#             json=payload,
#             auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
#         )
#         response.raise_for_status()
#     except requests.RequestException as e:
#         # If the DAG trigger fails, you might still want to inform the user.
#         # Here we raise an error. Alternatively, you could return a success for upload 
#         # and a warning for the DAG trigger.
#         raise HTTPException(status_code=500, detail=f"Airflow DAG trigger failed: {e}")
    
#     # 3. Return a response indicating success
#     return {
#         "message": "File uploaded to GCS and Airflow DAG triggered successfully.",
#         "file": file.filename,
#         "dag_run_id": dag_run_id
#     }

# @app.get("/data", dependencies=[Depends(verify_token)])
# def get_data(n: int = 5):
#     """Fetch the last n records from the database."""
#     # Validate n
#     if n <= 0:
#         raise HTTPException(status_code=400, detail="Parameter 'n' must be positive.")
#     # Connect to Cloud SQL
#     try:

#         def getconn():
#             conn = connector.connect(
#                 conn_name,  # Cloud SQL instance connection name
#                 "pymysql",  # Database driver
#                 user=user,  # Database user
#                 password=password,  # Database password
#                 db=database,
#             )
#             return conn

#         pool = sqlalchemy.create_engine(
#             "mysql+pymysql://",  # or "postgresql+pg8000://" for PostgreSQL, "mssql+pytds://" for SQL Server
#             creator=getconn,
#         )
        

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
#     try:
        
#         query = f"""
#         SELECT 
#             sale_date, product_name, total_quantity
#         FROM SALES
#         ORDER BY sale_date DESC LIMIT {n};"""
#         with pool.connect() as db_conn:
#             result = db_conn.execute(sqlalchemy.text(query))
#             print(result.scalar())
#         df = pd.read_sql(query, pool)
#     except Exception as e:
#         # conn.close()
#         raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
#     # Convert rows to list of dicts
#     # result = [dict(zip(columns, row)) for row in rows]
#     return {"records": df.to_json(), "count": len(df)}




# # (Optional) Define a Pydantic model for the request body
# class PredictRequest(BaseModel):
#     product_names: List[str]
#     dates: List[str]

# class PredictRequest(BaseModel):
#     product_name: str
#     days: int

# @app.post("/predict")
# async def get_prediction(request: PredictRequest):
#     # Prepare the payload using the same keys as the working curl example
#     payload = {
#         "product_name": request.product_name,
#         "days": request.days
#     }
#     try:
#         response = requests.post(
#             "https://model-serving-148338842941.us-central1.run.app/predict", 
#             json=payload
#         )
#         response.raise_for_status()
#     except requests.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
#     predictions = response.json()
#     return {"predictions": predictions}

import os
from datetime import date, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Query
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
import io  # Needed for file pointer operations
from collections import Counter
import json

load_dotenv()
app = FastAPI()

# Configuration from environment
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# Database config
host = os.getenv("DB_HOST")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
database = os.getenv("DB_NAME")
conn_name = os.getenv("INSTANCE_CONN_NAME")
connector = Connector()
host = os.getenv("DB_HOST")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASS")
database = os.getenv("DB_NAME")
conn_name = os.getenv("INSTANCE_CONN_NAME")
connector = Connector()
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

# from fastapi import APIRouter, UploadFile, File, Query, HTTPException
# import pandas as pd
# import io, json
# from google.cloud import storage
# import requests

# router = APIRouter()

# @app.post("/upload")
# async def upload_file(
#     file: UploadFile = File(...),
#     deny_list: str = Query(None, description="Comma-separated product names to remove"),
#     rename_dict: str = Query(None, description='JSON dict of {"oldName":"newName"} pairs')
# ):
#     # Parse and validate query parameters
#     deny_items = []
#     if deny_list:
#         deny_items = [name.strip() for name in deny_list.split(",") if name.strip()]
#     rename_map = {}
#     if rename_dict:
#         try:
#             rename_map = json.loads(rename_dict)
#         except json.JSONDecodeError:
#             raise HTTPException(status_code=400, detail="rename_dict is not valid JSON")
#         if not isinstance(rename_map, dict):
#             raise HTTPException(status_code=400, detail="rename_dict must be a JSON object (dict)")
    
#     # Validate file extension
#     filename = file.filename
#     if not filename or not filename.lower().endswith((".xls", ".xlsx")):
#         raise HTTPException(status_code=400, detail="Invalid file format. Only .xls or .xlsx files are supported.")
    
#     # Read Excel file into DataFrame
#     try:
#         if filename.lower().endswith(".xls"):
#             df = pd.read_excel(file.file, engine="xlrd")
#         else:
#             df = pd.read_excel(file.file, engine="openpyxl")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Failed to read Excel file. Ensure the file is a valid .xls or .xlsx.")
#     try:
#         if filename.lower().endswith(".xls"):
#             df = pd.read_excel(file.file, engine="xlrd")
#         else:
#             df = pd.read_excel(file.file, engine="openpyxl")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Failed to read Excel file. Ensure the file is a valid .xls or .xlsx.")

#     # Canonicalize column names: lowercase, strip, replace spaces with underscores
#     df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#     # Now validate presence of 'product_name'
#     if "product_name" not in df.columns:
#         raise HTTPException(status_code=400, detail="Missing 'product_name' column in the Excel file.")
#     # Validate required column
#     if "product_name" not in df.columns:
#         raise HTTPException(status_code=400, detail="Missing 'product_name' column in the Excel file.")
    
#     # Drop rows with denied product names
#     if deny_items:
#         df = df.loc[~df['product_name'].isin(deny_items)].copy()
    
#     # Rename product names as per mapping
#     if rename_map:
#         df['product_name'].replace(rename_map, inplace=True)
    
#     # Save the modified DataFrame to a new Excel file in memory
#     output_buffer = io.BytesIO()
#     try:
#         if filename.lower().endswith(".xls"):
#             df.to_excel(output_buffer, index=False, engine="xlwt")
#             content_type = "application/vnd.ms-excel"
#         else:
#             df.to_excel(output_buffer, index=False, engine="openpyxl")
#             content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Error writing DataFrame to Excel format.")
#     output_buffer.seek(0)
    
#     # Upload the in-memory Excel file to GCS
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(GCS_BUCKET_NAME)           # GCS_BUCKET_NAME should be defined/configured
#     blob = bucket.blob(filename)  # using the original file name; adjust if needed for uniqueness
#     try:
#         blob.upload_from_file(output_buffer, content_type=content_type)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Failed to upload file to cloud storage.")
    
#     # Trigger the Airflow DAG via HTTP (existing functionality)
#     try:
#         response = requests.post(AIRFLOW_URL, json={"file_name": filename}, timeout=10)  # AIRFLOW_TRIGGER_URL configured elsewhere
#         response.raise_for_status()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Failed to trigger Airflow DAG.")
#     # Extract DAG run ID from Airflow response (assuming JSON response)
#     dag_run_id = None
#     try:
#         dag_run_id = response.json().get("dag_run_id")
#     except ValueError:
#         dag_run_id = None  # if response not JSON or no run_id
    
#     # Return response JSON
#     return {
#         "file_name": filename,
#         "dag_run_id": dag_run_id,
#         "message": "File processed and DAG triggered successfully."
#     }


@app.get("/data", dependencies=[Depends(verify_token)])
def get_data(n: int = 5):
    """Fetch the last n records from the database."""
    n = max(0,n)
    try:
        def getconn():
            conn = connector.connect(
                conn_name,      # Cloud SQL instance connection name
                "pymysql",      # Database driver
                user=user,      # Database user
                password=password,  # Database password
                db=database,    # Database name
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
        FROM SALES
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
            "https://model-serving-148338842941.us-central1.run.app/predict", 
            json=payload
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
    predictions = response.json()
    return {"predictions": predictions}


# New endpoint: validate_excel
# @app.post("/validate_excel", dependencies=[Depends(verify_token)])
# async def validate_excel(file: UploadFile = File(...)):
#     """
#     Validates an uploaded Excel file. This function:
#     1. Checks that the file is .xls or .xlsx and under 50MB.
#     2. Reads the Excel file and verifies that the headers exactly match:
#        sale_date, product_name, total_quantity.
#     3. Queries the SQL database to obtain unique product names from the SALES table.
#     4. Compares the product names from the Excel file with those in the database,
#        and returns any new product names.
#     """
#     # 1. Validate file extension.
#     ext = file.filename.rsplit('.', 1)[-1].lower()
#     if ext not in ("xls", "xlsx"):
#         raise HTTPException(status_code=400, detail="Invalid file type. Only .xls or .xlsx files are allowed.")
    
#     # 2. Enforce file size limit (50MB)
#     file.file.seek(0, io.SEEK_END)
#     file_size = file.file.tell()
#     if file_size > 50 * 1024 * 1024:
#         raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 50MB.")
#     file.file.seek(0)  # reset file pointer to beginning (synchronous)

#     # 3. Read the Excel file into a DataFrame using appropriate engine
#     try:
#         if ext == "xls":
#             df = pd.read_excel(file.file, engine="xlrd")
#         else:  # .xlsx
#             df = pd.read_excel(file.file, engine="openpyxl")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Failed to read the Excel file. Ensure it is a valid .xls or .xlsx file.")

#     # 4. Validate headers exactly.
#     expected_columns = ["Date", "Unit Price", "Quantity", "Transaction ID", "Store Location", "Product Name", "Producer ID"]
#     actual_columns = [col.strip() for col in df.columns.tolist()]
#     if Counter(actual_columns) != Counter(expected_columns):
#         raise HTTPException(status_code=400, detail=f"Invalid Excel header. Expected columns: {expected_columns}")

#     # 5. Query the database for existing unique product names.
#     try:
#         def getconn():
#             conn = connector.connect(
#                 conn_name,
#                 "pymysql",
#                 user=user,
#                 password=password,
#                 db=database,
#             )
#             return conn
        
#         pool = sqlalchemy.create_engine(
#             "mysql+pymysql://",
#             creator=getconn,
#         )
#         db_query = "SELECT DISTINCT product_name FROM SALES"
        
#         with pool.connect() as conn:
#             result = conn.execute(sqlalchemy.text(db_query))
#             db_products = {row[0] for row in result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail="Database error occurred while fetching products.")

#     # 6. Extract product names from the Excel file.
#     excel_products = set(df["product_name"].dropna().unique().tolist())
#     new_products = list(excel_products.difference(db_products))

#     # 7. Return the new product names.
#     return {"new_products": new_products}
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
