**File:** `__init__.py`

**Purpose:**  
Initializes the `Data_Pipeline.scripts` module and re-exports key components for seamless access across the pipeline.

**Key Features:**
- Imports main functions from [`pre_validation.py`](./pre_validation.py), [`preprocessing.py`](./preprocessing.py), and [`post_validation.py`](./post_validation.py).
- Exposes utility functions from [`utils.py`](./utils.py) such as:
  - `setup_gcp_credentials`
  - `load_bucket_data`
  - `send_email`
  - `upload_to_gcs`
- Imports the logger from [`logger.py`](./logger.py).
- Makes all essential components accessible via absolute imports through the `__all__` declaration.

**Execution Command:**  
_Not directly executed as a script; supports package/module initialization._

---


**File:** `dvc_versioning.py`  

**Purpose:**  
Automates versioning of files stored in a Google Cloud Storage (GCS) bucket using [DVC (Data Version Control)](https://dvc.org/). It ensures reproducibility of datasets by tracking and pushing file versions to a DVC remote hosted on GCS.

**Key Features:**
- Connects to and verifies the existence of GCP buckets for both cached data and DVC remote storage.
- Initializes a temporary DVC project, adds GCS files, and pushes them to a configured remote.
- Maintains a version history by saving custom metadata (file name, size, MD5, timestamp) to GCS.
- Offers support for debugging with temporary directory retention and verbose DVC config inspection.
- Utility methods for:
  - Running shell commands (`run_command`)
  - Bucket management (`ensure_bucket_exists`, `clear_bucket`)
  - File listing and metadata creation (`list_bucket_files`, `save_version_metadata`)
  - DVC configuration (`setup_and_verify_dvc_remote`, `debug_dvc_setup`)

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
Provides a flexible logging utility that uses Apache Airflow's logging system if available, or falls back to standard Python logging. Ensures consistent and formatted logging across all scripts.

**Key Features:**
- Configures a global logger instance (`logger`) for use across modules.
- Automatically uses Airflow’s `LoggingMixin` when running within an Airflow environment.
- Falls back to a standard `logging.StreamHandler` when Airflow is not present.
- Supports log levels: `info`, `warning`, `error`, `debug`, `critical`, and `exception`.
- Includes a `setLevel()` method to dynamically change the logging verbosity.

**Execution Command:**  
_Not meant to be executed directly. Used as a utility module to be imported in other scripts._

---

**File:** `pre_validation.py`  

**Purpose:**  
Performs schema validation on raw input files stored in a GCS bucket. Ensures required columns exist, deletes invalid files if necessary, and sends an email report summarizing validation results.

**Key Features:**
- Defines required schema for raw data via `PRE_VALIDATION_COLUMNS`.
- Loads and validates files from a specified GCS bucket using `load_bucket_data()`.
- Deletes files with missing required columns or empty datasets via `delete_blob_from_bucket()`.
- Collects validation errors and optionally sends a report email using `send_email()`.
- Supports batch processing of all files in a bucket and aggregates results into a single validation report.
- Accepts CLI arguments for:
  - Target bucket (`--bucket`)
  - Retaining invalid files (`--keep_invalid`)

**Execution Command:**  
```bash
python scripts/pre_validation.py --bucket full-raw-data --keep_invalid
```

---


**File:** `preprocessing.py`  

**Purpose:**  
Cleans, transforms, and engineers features from raw transaction data in GCS. It ensures the data is standardized, validated, and ready for modeling by applying a multi-step preprocessing pipeline.

**Key Features:**
- **Data Cleaning:**  
  - Standardizes date formats using regex patterns and fallback parsing.
  - Converts data types, enforces lowercase strings, and removes invalid entries.
  - Fills missing values in the `Unit Price` column using hierarchical time-based modes.

- **Data Validation & Filtering:**  
  - Filters out rows with missing required fields or invalid product names.
  - Removes exact duplicate transaction records.
  - Detects anomalies using IQR bounds and timestamp-based rules.

- **Feature Engineering:**  
  - Aggregates transactions to daily product-level quantities.
  - Extracts lag features and time-series features such as day, week, and rolling averages.

- **Post-Validation:**  
  - Performs post-validation checks and statistics generation using `post_validation()`.

- **File Management & Uploads:**  
  - Uploads cleaned data to the target and optional cache GCS buckets.
  - Deletes raw files post-processing if configured via CLI.
  - Sends anomaly alerts via email if data issues are detected.

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
Performs post-processing validation on cleaned data to ensure type correctness, presence of required fields, and statistical profiling. Sends alerts if any data anomalies are found.

**Key Features:**
- Validates columns like `Product Name`, `Total Quantity`, and `Date` for correct types and formats.
- Identifies and logs missing or invalid entries; appends detailed anomaly reasons.
- Sends email alerts with attached CSVs for rows that fail validation using `send_anomaly_alert()` from [`utils.py`](./utils.py).
- Generates grouped statistical summaries (mean, std, min, max, median, skewness, etc.) by product and uploads them to a GCS bucket as a `.json` file using `upload_to_gcs()`.
- Provides a `post_validation()` entry function to execute the entire flow (validation + stats generation + alerting) in one call.

**Execution Command:**  
_Not meant to be run directly. Called as a module function in the pipeline._  

---
