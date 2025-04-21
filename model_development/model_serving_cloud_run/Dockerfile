FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app

# Create app directory
WORKDIR /app

# Copy only necessary code
COPY model_development/model_serving_cloud_run/ ./

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Tell Cloud Run to expect traffic on this port
ENV PORT=8080 

# Start the app with gunicorn
CMD ["gunicorn","--workers=1","--threads=2","--timeout=900","-b","0.0.0.0:8080","main:app"]