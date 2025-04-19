# import os
# import logging
# import pandas as pd
# import numpy as np
# from typing import Dict, Any, Optional
# from datetime import datetime, timedelta  # Added timedelta
# import time
# import json
# import requests
# from fastapi import Depends, HTTPException, APIRouter, Header, FastAPI  # Added Header
# from scipy import stats
# from google.cloud import scheduler_v1
# from google.protobuf import duration_pb2
# import sqlalchemy
# from google.cloud.sql.connector import Connector
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# from fastapi.responses import JSONResponse

# # Create a router for health check endpoints
# app = FastAPI()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# # logging = logging.getlogging(__name__)
# logging.info("Model health check module initialized")

# # Define threshold constants
# RMSE_THRESHOLD = 18.0  # As specified
# P_VALUE_THRESHOLD = 0.05  # For KS test
# MAPE_THRESHOLD = 0.25  # Example threshold, adjust as needed from your model training code

# # Environment variables
# PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
# REGION = os.environ.get('VERTEX_REGION')
# BUCKET_URI = os.environ.get('GCS_BUCKET_NAME')
# IMAGE_URI = os.environ.get('TRAINING_IMAGE_URI') 
# TRAINING_TRIGGER_URL = os.environ.get('TRAINING_TRIGGER_URL') 
# API_TOKEN = os.environ.get("API_TOKEN")

# # Database connection helper function
# def get_db_connection():
#     """Create a database connection using Cloud SQL Python Connector"""
#     host = os.getenv("MYSQL_HOST")
#     user = os.getenv("MYSQL_USER")
#     password = os.getenv("MYSQL_PASSWORD")
#     database = os.getenv("MYSQL_DATABASE")
#     conn_name = os.getenv("INSTANCE_CONN_NAME")
#     connector = Connector()
    
#     def getconn():
#         conn = connector.connect(
#             conn_name,      # Cloud SQL instance connection name
#             "pymysql",      # Database driver
#             user=user,      # Database user
#             password=password,  # Database password
#             db=database,    # Database name
#             ip_type="PRIVATE"
#         )
#         return conn

#     pool = sqlalchemy.create_engine(
#         "mysql+pymysql://",
#         creator=getconn,
#     )
#     logging.info("Database connection pool created for health check")
#     return pool

# # Add the verify_token function that was missing from the original code
# def verify_token(token: str = Header(None)):
#     if API_TOKEN is None:
#         logging.warning("No API_TOKEN set on server; skipping token verification.")
#         return True
#     if token is None or token != API_TOKEN:
#         logging.error("Invalid or missing token in request.")
#         raise HTTPException(status_code=401, detail="Unauthorized: invalid token")
#     logging.info("Token verification passed.")
#     return True

# @app.get("/health", tags=["Health"])
# async def basic_health_check():
#     """Basic health check to verify API is running"""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "service": "SupplyChainOptimization"
#     }


# @app.get("/model/health", tags=["Health"])
# async def model_health_check(token: str = Depends(verify_token)):
#     """
#     Check the health of the ML model by comparing actual vs. predicted values
#     Returns metrics and status information
#     """
#     check_date = datetime.now()
    
#     # Calculate date one month ago
#     one_month_ago = check_date - timedelta(days=30)
    
#     # Format dates for SQL query
#     current_date_str = check_date.strftime('%Y-%m-%d %H:%M:%S')
#     month_ago_str = one_month_ago.strftime('%Y-%m-%d %H:%M:%S')
    
#     metrics = {
#         "rmse": None,
#         "mape": None,
#         "p_value": None,
#         "ks_statistic": None,
#         "sample_size": 0
#     }
#     status = "unknown"
#     issues = []
#     retraining_triggered = False
    
#     try:
#         # Get database connection
#         db_engine = get_db_connection()
        
#         # Updated query to use date range for past month
#         sales_query = f"""
#         SELECT 
#             sale_date, product_name, total_quantity
#         FROM SALES
#         WHERE sale_date BETWEEN '{month_ago_str}' AND '{current_date_str}'
#         ORDER BY sale_date DESC;
#         """
        
#         # Updated query to use date range for past month
#         preds_query = f"""
#         SELECT 
#             sale_date, product_name, total_quantity
#         FROM PREDS
#         WHERE sale_date BETWEEN '{month_ago_str}' AND '{current_date_str}'
#         ORDER BY sale_date DESC;
#         """
        
#         # Execute queries
#         with db_engine.connect() as conn:
#             sales_df = pd.read_sql(sales_query, conn)
#             preds_df = pd.read_sql(preds_query, conn)
        
#         logging.info(f"Retrieved {len(sales_df)} sales records and {len(preds_df)} prediction records from {month_ago_str} to {current_date_str}")
        
#         # Merge datasets on date and product_name to align actual vs predicted values
#         merged_df = pd.merge(
#             sales_df, 
#             preds_df,
#             on=['sale_date', 'product_name'], 
#             how='inner',
#             suffixes=('_actual', '_predicted')
#         )
        
#         metrics["sample_size"] = len(merged_df)
        
#         if len(merged_df) == 0:
#             logging.warning("No matching records found between sales and predictions")
#             status = "warning"
#             issues.append("No matching records found between actual sales and predictions")
#         else:
#             # Calculate metrics
#             y_true = merged_df['total_quantity_actual'].values
#             y_pred = merged_df['total_quantity_predicted'].values
            
#             # RMSE calculation
#             metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            
#             # MAPE calculation - handle divide by zero
#             # Replace zeros with a small value to avoid division by zero
#             y_true_safe = np.where(y_true == 0, 1e-10, y_true)
#             metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
            
#             # Data drift - KS test between distributions
#             ks_statistic, p_value = stats.ks_2samp(y_true, y_pred)
#             metrics["ks_statistic"] = float(ks_statistic)
#             metrics["p_value"] = float(p_value)
            
#             # Determine overall status
#             status = "healthy"
            
#             if metrics["rmse"] > RMSE_THRESHOLD:
#                 status = "unhealthy"
#                 issues.append(f"RMSE ({metrics['rmse']:.2f}) exceeds threshold ({RMSE_THRESHOLD})")
            
#             if metrics["p_value"] < P_VALUE_THRESHOLD:
#                 status = "unhealthy"
#                 issues.append(f"Significant data drift detected (p-value: {metrics['p_value']:.4f})")
            
#             if metrics["mape"] > MAPE_THRESHOLD:
#                 status = "unhealthy"
#                 issues.append(f"MAPE ({metrics['mape']:.2f}) exceeds threshold ({MAPE_THRESHOLD})")
    
#     except Exception as e:
#         logging.error(f"Error during model health check: {str(e)}")
#         status = "error"
#         issues.append(f"Health check error: {str(e)}")
    
#     # Always store results, even if there was an error
#     health_record = {
#         'check_date': check_date,
#         'status': status,
#         'rmse': metrics["rmse"],
#         'mape': metrics["mape"],
#         'p_value': metrics["p_value"],
#         'ks_statistic': metrics["ks_statistic"],
#         'sample_size': metrics["sample_size"],
#         'issues': json.dumps(issues)
#     }
    
#     # Store health check results
#     store_health_check_results(health_record)
    
#     # Trigger retraining if model is unhealthy
#     if status == "unhealthy":
#         logging.warning(f"Model health issues detected: {', '.join(issues)}")
#         retraining_triggered = trigger_model_retraining()
#         health_record['retraining_triggered'] = retraining_triggered
#         # Update the record with retraining status
#         update_retraining_status(check_date, retraining_triggered)
    
#     # Prepare response
#     response = {
#         "status": status,
#         "timestamp": check_date.isoformat(),
#         "metrics": {
#             "rmse": metrics["rmse"],
#             "rmse_threshold": RMSE_THRESHOLD,
#             "mape": metrics["mape"],
#             "mape_threshold": MAPE_THRESHOLD,
#             "ks_statistic": metrics["ks_statistic"],
#             "p_value": metrics["p_value"],
#             "p_value_threshold": P_VALUE_THRESHOLD
#         },
#         "sample_size": metrics["sample_size"],
#         "issues": issues,
#         "retraining_triggered": retraining_triggered
#     }
    
#     if status == "error":
#         return JSONResponse(
#             status_code=500,
#             content=response
#         )
    
#     return response


# def trigger_model_retraining():
#     """
#     Trigger model retraining using the Cloud Run function
#     """
#     try:
#         payload = {
#             'PROJECT_ID': PROJECT_ID,
#             'REGION': REGION,
#             'BUCKET_URI': BUCKET_URI,
#             'IMAGE_URI': IMAGE_URI
#         }
        
#         logging.info(f"Triggering model retraining with payload: {payload}")
        
#         headers = {
#             'Content-Type': 'application/json'
#         }
        
#         response = requests.post(
#             TRAINING_TRIGGER_URL,
#             json=payload,
#             headers=headers
#         )
        
#         if response.status_code == 200:
#             logging.info(f"Model retraining triggered successfully: {response.json()}")
#             return True
#         else:
#             logging.error(f"Failed to trigger model retraining. Status code: {response.status_code}, Response: {response.text}")
#             return False
        
#     except Exception as e:
#         logging.error(f"Exception when triggering model retraining: {str(e)}")
#         return False


# def store_health_check_results(results):
#     """
#     Store health check results in database for monitoring
#     """
#     try:
#         # Get database connection
#         db_engine = get_db_connection()
        
#         # Insert into MODEL_HEALTH_STATS table
#         query = sqlalchemy.text("""
#         INSERT INTO MODEL_HEALTH_STATS 
#         (check_date, status, rmse, mape, p_value, ks_statistic, sample_size, issues)
#         VALUES 
#         (:check_date, :status, :rmse, :mape, :p_value, :ks_statistic, :sample_size, :issues)
#         """)
        
#         with db_engine.connect() as conn:
#             conn.execute(query, results)
#             conn.commit()
            
#         logging.info("Health check results stored in database")
#         return True
        
#     except Exception as e:
#         logging.error(f"Failed to store health check results: {str(e)}")
#         return False


# def update_retraining_status(check_date, retraining_triggered):
#     """
#     Update the retraining status for a health check record
#     """
#     try:
#         db_engine = get_db_connection()
        
#         query = sqlalchemy.text("""
#         UPDATE MODEL_HEALTH_STATS 
#         SET retraining_triggered = :retraining_triggered
#         WHERE check_date = :check_date
#         """)
        
#         with db_engine.connect() as conn:
#             conn.execute(query, {
#                 'retraining_triggered': retraining_triggered,
#                 'check_date': check_date
#             })
#             conn.commit()
            
#         logging.info(f"Retraining status updated to {retraining_triggered} for check date {check_date}")
#         return True
        
#     except Exception as e:
#         logging.error(f"Failed to update retraining status: {str(e)}")
#         return False



# # def create_model_health_stats_table():
# #     """Create the MODEL_HEALTH_STATS table if it doesn't exist"""
# #     try:
# #         db_engine = get_db_connection()
# #         create_table_query = sqlalchemy.text("""
# #         CREATE TABLE IF NOT EXISTS MODEL_HEALTH_STATS (
# #             id INT AUTO_INCREMENT PRIMARY KEY,
# #             check_date DATETIME NOT NULL,
# #             status VARCHAR(50) NOT NULL,
# #             rmse FLOAT,
# #             mape FLOAT,
# #             p_value FLOAT,
# #             ks_statistic FLOAT,
# #             sample_size INT NOT NULL,
# #             retraining_triggered BOOLEAN DEFAULT FALSE,
# #             issues TEXT,
# #             INDEX idx_check_date (check_date)
# #         );
# #         """)
        
# #         with db_engine.connect() as conn:
# #             conn.execute(create_table_query)
# #             conn.commit()
            
# #         logging.info("MODEL_HEALTH_STATS table created or already exists")
# #         return True
# #     except Exception as e:
# #         logging.error(f"Failed to create MODEL_HEALTH_STATS table: {str(e)}")
# #         return False


# # Make sure to initialize the table
# # @app.on_event("startup")
# # async def startup_event():
# #     """Run on API startup - ensures MODEL_HEALTH_STATS table exists"""
# #     create_model_health_stats_table()


# # Add manual trigger endpoint for testing
# @app.post("/model/health/trigger", tags=["Health"])
# async def trigger_health_check(token: str = Depends(verify_token)):
#     """
#     Manually trigger a model health check
#     """
#     result = await model_health_check(token)
#     return {
#         "message": "Manual health check completed",
#         "result": result
#     }


# # # Add manual retraining trigger for testing
# # @app.post("/model/retrain", tags=["Health"])
# # async def manual_retrain(token: str = Depends(verify_token)):
# #     """
# #     Manually trigger model retraining
# #     """
# #     retraining_success = trigger_model_retraining()
# #     if retraining_success:
# #         return {"message": "Model retraining triggered successfully"}
# #     else:
# #         raise HTTPException(status_code=500, detail="Failed to trigger model retraining")




import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

from fastapi import Depends, HTTPException, Header, FastAPI, Request
from fastapi.responses import JSONResponse

from scipy import stats
import sqlalchemy
from google.cloud.sql.connector import Connector
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from google.cloud import monitoring_v3

# ─── Configure Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO
)

# ─── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI()

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logging.info(f"Completed {request.method} {request.url} in {duration:.2f}s with status {response.status_code}")
    return response

logging.info("Model health check module initialized")

# ─── Thresholds ─────────────────────────────────────────────────────────────────
RMSE_THRESHOLD   = 18.0
P_VALUE_THRESHOLD = 0.05
MAPE_THRESHOLD   = 0.25

# ─── Environment ────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
REGION     = os.environ.get('VERTEX_REGION')
API_TOKEN  = os.environ.get("API_TOKEN")

# ─── Monitoring Client ──────────────────────────────────────────────────────────
monitoring_client = monitoring_v3.MetricServiceClient()
project_name      = f"projects/{PROJECT_ID}"


def log_metric(metric_name: str, value: float):
    logging.debug(f"Logging metric {metric_name} with value {value}")
    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/{metric_name}"
    series.resource.type = "global"
    series.resource.labels["project_id"] = PROJECT_ID

    now = time.time()
    seconds = int(now)
    nanos   = int((now - seconds) * 10**9)

    point = monitoring_v3.Point()
    point.value.double_value = value

    interval = monitoring_v3.TimeInterval()
    interval.end_time.seconds = seconds
    interval.end_time.nanos   = nanos
    point.interval = interval

    series.points = [point]
    monitoring_client.create_time_series(
        name=project_name,
        time_series=[series],
    )
    logging.debug(f"Sent time series for {metric_name}")

# ─── Database Connection ────────────────────────────────────────────────────────
def get_db_connection():
    logging.info("Creating database connection pool")
    connector = Connector()
    user     = os.getenv("MYSQL_USER")
    pwd      = os.getenv("MYSQL_PASSWORD")
    db       = os.getenv("MYSQL_DATABASE")
    conn_name = os.getenv("INSTANCE_CONN_NAME")

    def getconn():
        return connector.connect(
            conn_name, "pymysql",
            user=user, password=pwd, db=db,
            ip_type="PRIVATE"
        )

    engine = sqlalchemy.create_engine("mysql+pymysql://", creator=getconn)
    logging.info("Database connection pool created")
    return engine

# ─── Security ───────────────────────────────────────────────────────────────────
def verify_token(token: str = Header(None)):
    logging.debug("Verifying API token")
    if API_TOKEN is None:
        return True
    if token != API_TOKEN:
        logging.warning("Unauthorized access attempt with invalid token")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# ─── Health Check Endpoint ──────────────────────────────────────────────────────
@app.post("/model/health", tags=["Health"])
async def model_health_check(token: str = Depends(verify_token)):
    logging.info("Running model health check")
    now = datetime.now()
    start = (now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    end   = now.strftime("%Y-%m-%d %H:%M:%S")

    sales_q = f"""
      SELECT sale_date, product_name, total_quantity
      FROM SALES
      WHERE sale_date BETWEEN '{start}' AND '{end}';
    """
    preds_q = f"""
      SELECT sale_date, product_name, total_quantity
      FROM PREDICT
      WHERE sale_date BETWEEN '{start}' AND '{end}';
    """

    logging.debug(f"Sales query: {sales_q}")
    logging.debug(f"Predictions query: {preds_q}")

    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            sales_df = pd.read_sql(sales_q, conn)
            preds_df = pd.read_sql(preds_q, conn)
        logging.info(f"Fetched {len(sales_df)} sales rows and {len(preds_df)} prediction rows")

        merged = pd.merge(
            sales_df, preds_df,
            on=["sale_date", "product_name"],
            suffixes=("_actual", "_predicted")
        )
        n = len(merged)
        logging.info(f"Merged dataset has {n} rows")

        if n == 0:
            status, issues = "warning", ["No matching records in last 30 days"]
            metrics = {"sample_size": 0}
        else:
            y_true = merged["total_quantity_actual"].values
            y_pred = merged["total_quantity_predicted"].values

            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mape = float(mean_absolute_percentage_error(y_true, y_pred))
            ks_stat, p_val = stats.ks_2samp(y_true, y_pred)

            metrics = {
                "rmse": rmse,
                "mape": mape,
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "sample_size": n
            }
            logging.info(f"Calculated metrics: RMSE={rmse:.2f}, MAPE={mape:.2f}, KS={ks_stat:.2f}, p-value={p_val:.4f}")

            # Emit to Cloud Monitoring
            for name, val in metrics.items():
                if name in ("rmse", "mape", "ks_statistic", "p_value"):
                    log_metric(f"model/{name}", val)

            # Determine health
            status = "healthy"
            issues = []
            if rmse > RMSE_THRESHOLD:
                status = "unhealthy"
                issues.append(f"RMSE {rmse:.2f} > {RMSE_THRESHOLD}")
            if p_val < P_VALUE_THRESHOLD:
                status = "unhealthy"
                issues.append(f"p-value {p_val:.4f} < {P_VALUE_THRESHOLD}")

        response = {"status": status, "timestamp": now.isoformat(), "metrics": metrics, "issues": issues}
        logging.info(f"Health check result: {response}")
        return response

    except Exception as e:
        logging.error(f"Model health check error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )
