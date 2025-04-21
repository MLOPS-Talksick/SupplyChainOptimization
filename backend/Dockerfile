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

# Copy the FastAPI app code (and other needed files) into the image
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Command to start the Uvicorn server for FastAPI 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]