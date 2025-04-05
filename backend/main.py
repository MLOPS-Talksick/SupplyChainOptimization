# main.py (excerpt)
import os
from datetime import date, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from google.cloud import storage
import pymysql
from google.cloud import aiplatform

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
    # 1. Validate file type by extension or MIME type
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    # Accept .xls or .xlsx
    if not (filename.lower().endswith(".xls") or filename.lower().endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only .xls or .xlsx files are allowed.")
    # Optionally, check MIME type as well for extra safety
    if file.content_type not in ["application/vnd.ms-excel", 
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an Excel file.")

    # 2. Read file (in chunks to be memory-safe) and check size
    file_contents = await file.read()  # read the entire file into memory (be careful with very large files)
    file_size = len(file_contents)
    max_size = 50 * 1024 * 1024  # 50 MB in bytes
    if file_size > max_size:
        raise HTTPException(status_code=400, detail="File too large. Must be <= 50 MB.")
    
    # 3. Upload to Cloud Storage
    try:
        storage_client = storage.Client()  # uses ADC credentials (should be available on Cloud Run)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        # Upload the file contents
        blob.upload_from_string(file_contents)
    except Exception as e:
        # Log the exception (omitted here) and return error
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    
    # 4. Return success response
    return {"message": f"File '{filename}' uploaded successfully to bucket {BUCKET_NAME}.", "size": file_size}

@app.get("/data", dependencies=[Depends(verify_token)])
def get_data(n: int = 5):
    """Fetch the last n records from the database."""
    # Validate n
    if n <= 0:
        raise HTTPException(status_code=400, detail="Parameter 'n' must be positive.")
    # Connect to Cloud SQL
    try:
        conn = pymysql.connect(user=DB_USER, password=DB_PASS, database=DB_NAME,
                                unix_socket=f"/cloudsql/{INSTANCE_CONN_NAME}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    try:
        with conn.cursor() as cursor:
            # Example query: adjust table/columns as needed.
            cursor.execute("SELECT * FROM sales_data ORDER BY id DESC LIMIT %s;", (n,))
            rows = cursor.fetchall()
            # Get column names for constructing dict (cursor.description has info)
            columns = [desc[0] for desc in cursor.description]
        conn.close()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    # Convert rows to list of dicts
    result = [dict(zip(columns, row)) for row in rows]
    return {"records": result, "count": len(result)}


@app.get("/predict", dependencies=[Depends(verify_token)])
def get_predictions():
    """Get forecast predictions for all products for the next week."""
    # 1. Get distinct product names from the database
    try:
        conn = pymysql.connect(user=DB_USER, password=DB_PASS, database=DB_NAME,
                                unix_socket=f"/cloudsql/{INSTANCE_CONN_NAME}")
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT product_name FROM sales_data;")
            products = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error fetching products: {str(e)}")
    if not products:
        raise HTTPException(status_code=404, detail="No products found in database.")
    
    # 2. Generate dates from today to one week later
    today = date.today()
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(8)]  # 8 days including today
    
    # 3. Prepare instances and call Vertex AI endpoint
    try:
        aiplatform.init(project=PROJECT_ID, location=VERTEX_REGION)
        endpoint = aiplatform.Endpoint(VERTEX_ENDPOINT_ID)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vertex AI endpoint initialization failed: {str(e)}")
    # Build the list of instances for prediction
    instances = []
    for product in products:
        for d in dates:
            instances.append({"date": d, "product": product})
    # Call the prediction
    try:
        prediction_response = endpoint.predict(instances=instances)
        predictions = prediction_response.predictions  # list of predictions results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vertex AI prediction failed: {str(e)}")
    
    # 4. Format the results
    results = []
    for idx, pred in enumerate(predictions):
        # Each pred could be a single value or a structure. We'll just include it as-is.
        instance = instances[idx]
        results.append({
            "product": instance["product"],
            "date": instance["date"],
            "prediction": pred
        })
    return {"predictions": results, "count": len(results)}
