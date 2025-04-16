#!/bin/bash
set -e

if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "Activating service account using credentials file: $GOOGLE_APPLICATION_CREDENTIALS"
  gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
fi

# Use your project ID for uniqueness
BUCKET_NAME="tf-state-${GCP_PROJECT_ID:-primordial-veld-450618-n4}"  # fallback if env var not set
REGION="us-central1"

echo "Checking if bucket $BUCKET_NAME exists..."
if ! gsutil ls -b gs://$BUCKET_NAME/ &>/dev/null; then
  echo "Bucket $BUCKET_NAME does not exist. Creating..."
  gsutil mb -l $REGION gs://$BUCKET_NAME/
else
  echo "Bucket $BUCKET_NAME already exists."
fi


# Use your project ID for uniqueness
BUCKET_NAME="tf-state-deploy-${GCP_PROJECT_ID:-primordial-veld-450618-n4}"  # fallback if env var not set
REGION="us-central1"

echo "Checking if bucket $BUCKET_NAME exists..."
if ! gsutil ls -b gs://$BUCKET_NAME/ &>/dev/null; then
  echo "Bucket $BUCKET_NAME does not exist. Creating..."
  gsutil mb -l $REGION gs://$BUCKET_NAME/
else
  echo "Bucket $BUCKET_NAME already exists."
fi