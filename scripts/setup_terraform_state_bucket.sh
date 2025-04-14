#!/bin/bash
set -e

# Activate service account credentials explicitly (if necessary)
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "Activating service account using credentials file: $GOOGLE_APPLICATION_CREDENTIALS"
  gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
fi

# Variables - replace these with your values or pass them in as environment variables.
BUCKET_NAME="my-terraform-state-bucket"
REGION="us-central1"

# Check if the bucket exists
if ! gsutil ls -b gs://$BUCKET_NAME/; then
  echo "Bucket $BUCKET_NAME does not exist. Creating..."
  # Create the bucket in the specified region.
  gsutil mb -l $REGION gs://$BUCKET_NAME/
else
  echo "Bucket $BUCKET_NAME already exists."
fi