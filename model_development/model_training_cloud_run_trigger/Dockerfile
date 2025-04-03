FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app\
    GOOGLE_APPLICATION_CREDENTIALS="/app/secret/gcp-key.json"

WORKDIR /app

COPY secret/gcp-key.json /app/secret/gcp-key.json

COPY model_development/model_training_cloud_run_trigger/ ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Service must listen to $PORT environment variable
ENV PORT 8080

# Set the entrypoint
CMD ["python", "main.py"]