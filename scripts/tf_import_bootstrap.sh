#!/bin/bash
set -e

# Change directory to the bootstrap folder where your Terraform config files reside.
cd "$(dirname "$0")/../bootstrap"

# Set your GCP project ID
PROJECT_ID="primordial-veld-450618-n4"
# Define REGION based on GCP_LOCATION or default to us-central1.
REGION="${GCP_LOCATION:-us-central1}"

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

#####################################
# 2. Import VM Service Account
#####################################

# Define the VM service account details
SA_VM_ID="vm-service-account"
SA_VM_EMAIL="${SA_VM_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== Checking VM Service Account: ${SA_VM_EMAIL} ==="

# Disable exit on error to capture gcloud output.
set +e
OUTPUT_VM=$(gcloud iam service-accounts describe "${SA_VM_EMAIL}" --project "${PROJECT_ID}" 2>&1)
EXIT_CODE_VM=$?
set -e

echo "gcloud exit code for VM SA: $EXIT_CODE_VM"
echo "gcloud output for VM SA:"
echo "$OUTPUT_VM"

if [ $EXIT_CODE_VM -eq 0 ]; then
  echo "Service Account ${SA_VM_EMAIL} exists. Importing into Terraform..."
  terraform import google_service_account.vm_service_account "projects/${PROJECT_ID}/serviceAccounts/${SA_VM_EMAIL}"
else
  echo "Service Account ${SA_VM_EMAIL} not found. Terraform will create it."
fi

#####################################
# Artifact Registry: airflow-docker-image
#####################################
ARTIFACT_REGISTRY_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
REPO_FORMAT="${REPO_FORMAT:-DOCKER}"
echo "Checking Artifact Registry (${ARTIFACT_REGISTRY_NAME})..."

EXISTING_REPO=""
for i in {1..3}; do
  echo "Attempt $i to list Artifact Registry repositories..."
  OUTPUT=$(gcloud artifacts repositories list \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --filter="name:${ARTIFACT_REGISTRY_NAME}" \
    --format="value(name)" 2>&1)
  
  if [[ "$OUTPUT" =~ "Permission" ]]; then
    echo "Received permission error: $OUTPUT"
    sleep 5
  elif [[ -n "$OUTPUT" ]]; then
    EXISTING_REPO="$OUTPUT"
    break
  else
    echo "No repository found on attempt $i. Retrying in 5 seconds..."
    sleep 5
  fi
done

if [[ -n "$EXISTING_REPO" ]]; then 
    echo "Artifact Registry ${ARTIFACT_REGISTRY_NAME} exists. Importing..."
    terraform import google_artifact_registry_repository.airflow_docker_repo "projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REGISTRY_NAME}"
else
    echo "Artifact Registry ${ARTIFACT_REGISTRY_NAME} not found or access denied. Terraform will create it."
fi

echo "=== Import process completed ==="
