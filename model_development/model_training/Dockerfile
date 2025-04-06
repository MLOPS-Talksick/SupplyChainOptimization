FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc git && rm -rf /var/lib/apt/lists/*

# Copy project files (all relative to root because build context is root)
COPY Data_Pipeline/scripts/ /app/Data_Pipeline/scripts/
COPY ML_Models/scripts/ /app/ML_Models/scripts/
COPY ML_Models/database.env /app/ML_Models/database.env
COPY model_development/model_training/ /app/model_development/model_training/

# Install Python dependencies
WORKDIR /app/model_development/model_training
RUN pip install --no-cache-dir -r requirements.txt

# Set final working directory
WORKDIR /app/ML_Models/scripts

# Entrypoint
CMD ["python", "model_lstm.py"]