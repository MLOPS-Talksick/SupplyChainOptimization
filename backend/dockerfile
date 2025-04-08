# Use the official lightweight Python 3.11 image
FROM python:3.11-slim

# Set working directory inside the container
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