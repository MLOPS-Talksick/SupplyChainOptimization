FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (to leverage Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY train_xgboost.py .

# Run the script when the container starts
CMD ["python", "train_xgboost.py"]
