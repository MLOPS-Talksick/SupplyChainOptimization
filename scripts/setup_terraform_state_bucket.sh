#!/bin/bash
set -e

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