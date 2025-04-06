FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app\
    GOOGLE_APPLICATION_CREDENTIALS="/app/secret/gcp-key.json"



# Set the working directory
WORKDIR /app


COPY secret/gcp-key.json /app/secret/gcp-key.json

# Copy Data_Pipeline scripts
COPY Data_Pipeline/scripts/logger.py /app/Data_Pipeline/scripts/logger.py
COPY Data_Pipeline/scripts/utils.py /app/Data_Pipeline/scripts/utils.py

# Copy ML_Models scripts
COPY ML_Models/scripts/utils.py /app/ML_Models/scripts/utils.py
COPY ML_Models/database.env /app/ML_Models/scripts/database.env
# COPY ML_Models/scripts/model_xgboost.py /app/ML_Models/scripts/model_xgboost.py
COPY ML_Models/scripts/model_lstm.py /app/ML_Models/scripts/model_lstm.py

# Copy model_development/model_training files
COPY model_development/model_training/requirements.txt /app/model_development/model_training/requirements.txt

# Set the working directory to the model_training folder
WORKDIR /app/model_development/model_training

# Install Python dependencies from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/ML_Models/scripts

# Set the entry point to run the `model_xgboost.py` script
CMD ["python", "model_lstm.py"]