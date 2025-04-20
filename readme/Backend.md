
## Backend
```
backend/
├── main.py                        
├── requirements.txt
├── Dockerfile
```

---

### 1. Requirements  
**File:** `requirements.txt`  
**Purpose:** Defines all dependencies for running the FastAPI-based backend with GCP, database, and ML integrations.  

**Key Features:**  
- **API & Validation:** `fastapi`, `uvicorn`, `pydantic`, `python-multipart`  
- **Excel Support:** `openpyxl`, `xlrd`, `xlwt`  
- **Cloud Services:** `google-cloud-storage`, `google-cloud-scheduler`, `google-cloud-aiplatform`  
- **Database Access:** `cloud-sql-python-connector`, `pymysql`, `sqlalchemy`  
- **Utils:** `requests`, `pandas`, `python-dotenv`, `python-dateutil`, `protobuf`

```bash
pip install -r requirements.txt
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

### 3. Main Backend Logic
**File:** `main.py`  
**Purpose:** Provides API endpoints for file uploads, forecasting, validation, and monitoring in the supply chain ML pipeline using FastAPI and GCP services.  

**Key Features:**  
- **Auth & Config:** Loads environment vars and uses token-based API auth.  
- **Upload & Trigger:** Uploads Excel to GCS and triggers Airflow DAG.  
- **Monitoring:** Fetches recent stats and sales/prediction records from Cloud SQL.  
- **Prediction:** Retrieves cached forecasts or triggers model serving API.  
- **Validation:** Validates Excel structure and flags unseen products.  
- **Scheduler:** Updates GCP Cloud Scheduler job via cron expression.  
- **Email Storage:** Saves user email to a GCS bucket.

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

---
