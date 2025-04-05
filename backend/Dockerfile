# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies (if any needed, e.g. mysql client libraries - but PyMySQL is pure Python so no extra needed)
# RUN apt-get update && apt-get install -y <packages> && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
# (First copy only requirements to leverage Docker layer caching if possible)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code
COPY . .

# Expose port (Cloud Run will use $PORT env, but we set 8080 as default)
EXPOSE 8080

# Command to run the app with Uvicorn
# Use the PORT environment variable (default to 8080 if not set)
ENV PORT=8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
