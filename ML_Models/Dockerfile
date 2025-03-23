FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY ML_Models/requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the necessary directories
COPY ML_Models /app/ML_Models
COPY Data_Pipeline/scripts /app/Data_Pipeline/scripts

# Set working directory for ML scripts
WORKDIR /app/ML_Models/scripts

# Default command 
CMD ["python", "model_xgboost.py"]
