# Use the official lightweight Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app

# Create app directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start Uvicorn on the FastAPI app
CMD ["uvicorn", "health_check:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]
