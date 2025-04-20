# Data Pipeline

```
Data_Pipeline/
├── scripts/
│   ├── __init__.py                
│   ├── dvc_versioning.py          
│   ├── logger.py                  
│   ├── post_validation.py         
│   ├── pre_validation.py          
│   ├── preprocessing.py           
│   └── utils.py                   
├── tests/
│   ├── __init__.py
│   ├── requirements-test.txt      
│   ├── testDataPreprocessing.py
│   ├── testDvcVersioning.py
│   ├── testPostValidation.py
│   ├── testPreValidation.py
│   └── testUtils.py
├── Dockerfile                     
├── requirements.txt               
```Absolutely! Here's your updated doc with **numbered subheadings** using the **file names without extensions**, like `1. requirements`, `2. init`, etc., while keeping everything else clean and tight:
```
---

### 1. Requirements  
**File:** `requirements.txt`  
**Purpose:** Specifies all Python dependencies required for running the Airflow-based data pipeline, including data processing, cloud operations, email alerts, and data versioning.  
**Key Features:**  
- **Airflow & Providers:** `apache-airflow` for DAG-based orchestration; Google & Docker providers for GCP and containerized task support.  
- **Data Processing:** `numpy`, `pandas`, `polars` for fast, scalable data handling; `pyarrow` for cross-format compatibility.  
- **GCP Integration:** `google-cloud-storage`, `google-api-core` for interacting with Google Cloud services.  
- **Excel Support:** `openpyxl`, `XlsxWriter`, `fastexcel` for reading/writing Excel files.  
- **Utility Libraries:** `python-dotenv` for env management; `pyyaml` for YAML config parsing.  
- **Email Alerts:** `sendgrid` to send anomaly or pipeline failure notifications.  
- **Data Versioning:** `dvc`, `dvc-gs` for managing and syncing datasets on GCS.  
```bash
pip install -r requirements.txt
```

---

### 2. Init  
**File:** `__init__.py`  
**Purpose:** Initializes the `Data_Pipeline.scripts` module and re-exports key components for streamlined access.  
**Key Features:**  
- Re-exports main functions from `pre_validation.py`, `preprocessing.py`, `post_validation.py`.  
- Exposes utility methods and logging for use across the pipeline.  

---

### 3. Dvc Versioning 
**File:** `dvc_versioning.py`  
**Purpose:** Handles dataset versioning using DVC and GCS for reproducible pipelines.  
**Key Features:**  
- Sets up GCS-backed DVC remotes and pushes tracked files.  
- Maintains metadata for versioned datasets.  
- Includes utilities for bucket checks, file listing, and command execution.  
```bash
python scripts/dvc_versioning.py \
  --cache_bucket your-input-bucket \
  --dvc_remote your-dvc-remote-bucket \
  --gcp_key_path /path/to/gcp-key.json
```

---

### 4. Logger  
**File:** `logger.py`  
**Purpose:** Creates a consistent logger compatible with both Airflow and standard Python environments.  
**Key Features:**  
- Automatically uses Airflow logging if available.  
- Supports standard log levels and dynamic verbosity.  

---

### 5. Pre Validation  
**File:** `pre_validation.py`  
**Purpose:** Validates raw input files from GCS and removes/flags those with schema issues.  
**Key Features:**  
- Validates schema against predefined columns.  
- Optionally deletes invalid files.  
- Sends validation summary emails.  
```bash
python scripts/pre_validation.py --bucket full-raw-data --keep_invalid
```

---

### 6. Preprocessing  
**File:** `preprocessing.py`  
**Purpose:** Cleans, transforms, and prepares raw data for modeling.  
**Key Features:**  
- Standardizes formats, fills missing values, removes invalid records.  
- Detects anomalies and performs feature engineering.  
- Uploads processed data to GCS and triggers post-validation.  
```bash
python scripts/preprocessing.py \
  --source_bucket full-raw-data \
  --destination_bucket fully-processed-data \
  --cache_bucket metadata-stats \
  --delete_after
```

---

### 7. Post Validation  
**File:** `post_validation.py`  
**Purpose:** Validates cleaned datasets and computes statistical summaries.  
**Key Features:**  
- Checks data types and missing fields.  
- Sends anomaly alerts via email.  
- Generates product-level metrics and uploads to GCS.  

---

### 8. Utils  
**File:** `utils.py`  
**Purpose:** Provides shared utilities for GCS operations, validation, and alerting.  
**Key Features:**  
- Loads, uploads, and manages GCS files.  
- Sends emails and anomaly alerts.  
- Supports validation error tracking.  

---

### 9. Dockerfile  
**File:** `Dockerfile`  
**Purpose:** Defines a lightweight and efficient container environment for running the Python-based data pipeline application.  
**Key Features:**  
- Uses `python:3.10-slim` base image.  
- Sets env variables for runtime and pip performance.  
- Separates dependency install to leverage Docker caching.  
- Installs dependencies without pip cache.  
- Copies application code and sets `/app/scripts` as working dir.  
- Default command runs `main.py`.  
```bash
docker build -t your-image-name .
```
**Run Command:**  
```bash
docker run --rm your-image-name
```

---
