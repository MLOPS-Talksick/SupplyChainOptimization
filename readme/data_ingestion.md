__init__.py
Initializes the scripts package and re-exports key components for streamlined access.​

Re-exported Components:

Logger: Sets up the logging mechanism.

Main Functions: Imports the main functions from pre_validation, preprocessing, and post_validation modules.

Utilities: Includes functions for GCP credential setup, data loading, email sending, and uploading to Google Cloud Storage.



dvc_versioning.py
📦 Purpose: Automates versioning of files in a GCP bucket using DVC (Data Version Control).

Key Features:

Connects to GCP buckets and sets up a DVC remote.

Downloads files from a "cache" bucket, tracks them with DVC, and pushes them to the remote bucket.

Saves custom version metadata (timestamp, file info) to GCS.

Includes options for debugging, temporary directory retention, and remote bucket clearing.

Utility Functions:

track_bucket_data() – Main function handling GCS-DVC interactions.

setup_and_verify_dvc_remote() – Initializes and configures DVC.

save_version_metadata() – Stores file version history in GCS.

run_command() – Wrapper for shell commands with logging.

