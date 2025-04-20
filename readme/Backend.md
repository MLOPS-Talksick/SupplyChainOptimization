
## Backend
```
backend/
├── main.py                        
├── requirements.txt
├── Dockerfile
```
---

### 2. Docker 
**File:** `Dockerfile`  
**Purpose:** Builds a lightweight, production-ready container for deploying a FastAPI application on Cloud Run.  

**Key Features:**  
- **Base Image:** Uses `python:3.11-slim` to keep the image lean and efficient.  
- **Environment Setup:** Configures Python and pip environment variables for optimal performance and cache control.  
- **App Structure:** Sets `/app` as the working directory and copies all source files into it.  
- **Dependency Installation:** Installs project dependencies from `requirements.txt` without caching.  
- **Port Exposure:** Opens port `8080` to allow external access via Cloud Run.  
- **App Execution:** Launches the FastAPI application using `uvicorn` for asynchronous HTTP serving.

```bash
# Build and run locally
docker build -t fastapi-app .
docker run -p 8080:8080 fastapi-app
```

---

### 3. FastAPI (Main Logic)  
**File:** `main.py`  
**Purpose:** Implements the backend logic for a supply chain forecasting API using FastAPI. It handles file uploads, data validation, database queries, predictions, monitoring, and scheduling through GCP services.  

**Key Features:**  
- **Security & Config:**  
  • Loads environment variables with `dotenv`.  
  • Token-based authentication for API access.  

- **File Upload & GCS Integration:**  
  • Validates and processes `.xls`/`.xlsx` files.  
  • Applies `deny_list` filters and `rename_dict` mappings.  
  • Uploads cleaned data to Google Cloud Storage.  
  • Triggers Airflow DAG on successful file upload.

- **Monitoring & Data Access:**  
  • `/monitoring` and `/data` endpoints retrieve recent rows from `STATS`, `PREDICT`, and `SALES` tables in Cloud SQL.  
  • `/get-stats` returns overall dataset metadata (start date, entry count, etc.).

- **Prediction Pipeline:**  
  • `/predict` checks existing forecasts in `PREDICT` table.  
  • If missing, triggers model via `MODEL_SERVING_URL`.  
  • Returns and optionally caches new predictions.

- **Email & Excel Validation:**  
  • `/upload-email` stores user email in GCS for newsletter/engagement.  
  • `/validate_excel` checks file structure, headers, and detects unseen products.

- **Scheduler Management:**  
  • `/update-cron-time` updates the Cloud Scheduler job to run Airflow DAGs at user-defined times.  
  • `update_scheduler_job()` handles all optional fields (retry, timezone, HTTP target, etc.).

```bash
# Run the FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8080
```

---
