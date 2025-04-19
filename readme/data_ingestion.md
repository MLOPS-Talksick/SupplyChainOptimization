__init__.py
Purpose:
Initializes the Data_Pipeline.scripts module and re-exports key components for seamless access across the pipeline.

Key Features:

Imports main functions from pre_validation.py, preprocessing.py, and post_validation.py.

Exposes utility functions from utils.py such as:

setup_gcp_credentials

load_bucket_data

send_email

upload_to_gcs

Imports the logger from logger.py.

Makes all essential components accessible via absolute imports through the __all__ declaration.

Execution Command:
Not directly executed as a script; supports package/module initialization.

