![Data Pipeline Architecture](/Media/data_pipeline.png)
**File:** `requirements.txt`  

**Purpose:**  
Specifies all Python dependencies required for running the Airflow-based data pipeline, including data processing, cloud operations, email alerts, and data versioning.

**Key Features:**
- **Airflow & Providers:**  
  - `apache-airflow` for DAG-based orchestration.  
  - Google & Docker providers for GCP and containerized task support.

- **Data Processing:**  
  - `numpy`, `pandas`, and `polars` for fast, scalable data handling.  
  - `pyarrow` for cross-format compatibility.

- **GCP Integration:**  
  - `google-cloud-storage` and `google-api-core` for interacting with Google Cloud services.

- **Excel Support:**  
  - `openpyxl`, `XlsxWriter`, and `fastexcel` for reading/writing Excel files.

- **Utility Libraries:**  
  - `python-dotenv` for environment variable management.  
  - `pyyaml` for YAML configuration parsing.

- **Email Alerts:**  
  - `sendgrid` to send anomaly or pipeline failure notifications.

- **Data Versioning:**  
  - `dvc` and `dvc-gs` for managing and syncing versioned datasets on GCS.

**Installation Command:**  
```bash
pip install -r requirements.txt
```
---

**File:** `__init__.py`  

**Purpose:**  
Initializes the `Data_Pipeline.scripts` module and re-exports key components for streamlined access.

**Key Features:**
- Re-exports main functions from [`pre_validation.py`](./pre_validation.py), [`preprocessing.py`](./preprocessing.py), and [`post_validation.py`](./post_validation.py).
- Exposes utility methods and logging for use across the pipeline.

**Execution Command:**  
_Not directly executed; supports module initialization._

---

**File:** `dvc_versioning.py`  

**Purpose:**  
Handles dataset versioning using DVC and GCS for reproducible pipelines.

**Key Features:**
- Sets up GCS-backed DVC remotes and pushes tracked files.
- Maintains metadata for versioned datasets.
- Includes utilities for bucket checks, file listing, and command execution.

**Execution Command:**  
```bash
python scripts/dvc_versioning.py \
  --cache_bucket your-input-bucket \
  --dvc_remote your-dvc-remote-bucket \
  --gcp_key_path /path/to/gcp-key.json
```

---

**File:** `logger.py`  

**Purpose:**  
Creates a consistent logger compatible with both Airflow and standard Python environments.

**Key Features:**
- Automatically uses Airflow logging if available.
- Supports standard log levels and dynamic verbosity.

**Execution Command:**  
_Not meant to be executed directly._

---

**File:** `pre_validation.py`  

**Purpose:**  
Validates raw input files from GCS and removes/flags those with schema issues.

**Key Features:**
- Validates schema against predefined columns.
- Optionally deletes invalid files.
- Sends validation summary emails.

**Execution Command:**  
```bash
python scripts/pre_validation.py --bucket full-raw-data --keep_invalid
```

---

**File:** `preprocessing.py`  

**Purpose:**  
Cleans, transforms, and prepares raw data for modeling.

**Key Features:**
- Standardizes formats, fills missing values, and removes invalid records.
- Detects anomalies and performs feature engineering.
- Uploads processed data to GCS and triggers post-validation.

**Execution Command:**  
```bash
python scripts/preprocessing.py \
  --source_bucket full-raw-data \
  --destination_bucket fully-processed-data \
  --cache_bucket metadata-stats \
  --delete_after
```

---

**File:** `post_validation.py`  

**Purpose:**  
Validates cleaned datasets and computes statistical summaries.

**Key Features:**
- Checks data types and missing fields.
- Sends anomaly alerts via email.
- Generates product-level metrics and uploads them to GCS.

**Execution Command:**  
_Not meant to be executed directly._

---

**File:** `utils.py`  

**Purpose:**  
Provides shared utilities for GCS operations, validation, and alerting.

**Key Features:**
- Loads, uploads, and manages GCS files.
- Sends emails and anomaly alerts.
- Supports validation error tracking.

**Execution Command:**  
_Not meant to be executed directly._

--- 

**File:** `Dockerfile`  

**Purpose:**  
Defines a lightweight and efficient container environment for running the Python-based data pipeline application.

**Key Features:**
- Uses the official `python:3.10-slim` base image for minimal footprint.
- Sets environment variables to improve Python runtime behavior and pip performance.
- Separates dependency installation (`requirements.txt`) to leverage Docker layer caching.
- Installs all Python dependencies without using the pip cache.
- Copies the full application code into the container.
- Sets `/app/scripts` as the working directory to target script execution.
- Specifies a default container command to run `main.py`.

**Build Command:**  
```bash
docker build -t your-image-name .
```

**Run Command:**  
```bash
docker run --rm your-image-name
```

--- 
