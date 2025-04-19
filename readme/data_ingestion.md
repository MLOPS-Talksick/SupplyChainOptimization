
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

# ===============================
# Core Data Libraries
# ===============================
numpy>=1.21.0,<1.24.0           # Numerical operations
pandas>=1.3.0,<1.5.0            # Tabular data processing
polars-lts-cpu==1.22.0          # High-performance DataFrame library (Polars LTS)

# ===============================
# File I/O Support
# ===============================
pyarrow>=10.0.0                 # Parquet & Arrow format handling
openpyxl==3.1.5                 # Read Excel files (.xlsx)
XlsxWriter==3.2.2               # Write Excel files (.xlsx)

# ===============================
# Environment & Config
# ===============================
python-dotenv==1.0.1            # Load env vars from .env file
pyyaml>=6.0.0                   # YAML config file handling

# ===============================
# Testing & Mocking
# ===============================
pytest>=7.0.0                   # Unit testing framework
pytest-cov>=4.0.0               # Code coverage plugin
mock>=5.0.0                     # Mocking library for tests

# ===============================
# Google Cloud Platform (GCP) Mocks
# ===============================
google-api-core>=2.8.2,<2.9.0   # Core utilities for GCP APIs
google-cloud-storage>=1.30.0,<2.0.0  # GCS client library

# ===============================
# DVC (Data Version Control)
# ===============================
dvc==3.30.1                     # Core DVC CLI & logic
dvc-gs==3.0.1                   # Google Storage remote support for DVC
