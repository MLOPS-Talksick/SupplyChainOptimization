#!/bin/bash
set -e

# Change directory to the bootstrap folder where your Terraform config files reside.
cd "$(dirname "$0")/../bootstrap"

# Set your GCP project ID
PROJECT_ID="primordial-veld-450618-n4"

#####################################
# 1. Import Terraform Service Account
#####################################

# Define the Terraform service account details
SA_TF_ID="terraform-service-account"
SA_TF_EMAIL="${SA_TF_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== Checking Terraform Service Account: ${SA_TF_EMAIL} ==="

# Temporarily disable exit on error so we can capture gcloud output.
set +e
OUTPUT=$(gcloud iam service-accounts describe "${SA_TF_EMAIL}" --project "${PROJECT_ID}" 2>&1)
EXIT_CODE=$?
set -e

echo "gcloud exit code: $EXIT_CODE"
echo "gcloud output:"
echo "$OUTPUT"

if [ $EXIT_CODE -eq 0 ]; then
  echo "Service Account ${SA_TF_EMAIL} exists. Importing into Terraform..."
  terraform import google_service_account.terraform_sa "projects/${PROJECT_ID}/serviceAccounts/${SA_TF_EMAIL}"
else
  echo "Service Account ${SA_TF_EMAIL} not found. Terraform will create it."
fi