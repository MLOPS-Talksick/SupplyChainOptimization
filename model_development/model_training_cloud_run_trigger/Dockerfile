FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app

WORKDIR /app


COPY model_development/model_training_cloud_run_trigger/ ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Service must listen to $PORT environment variable
ENV PORT 8080

# Set the entrypoint
# CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
CMD ["gunicorn",
     "--bind", "0.0.0.0:8080",
     "--workers", "1",
     "--timeout", "1800",
     "main:app"]
