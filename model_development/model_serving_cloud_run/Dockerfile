FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app

WORKDIR /app

# Copy application code
COPY model_development/model_serving_cloud_run/ ./

RUN pip install --no-cache-dir -r requirements.txt

# Service must listen to $PORT environment variable
ENV PORT 8080

# Run the web service
CMD ["python", "main.py"]