
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
