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
# Artifact Registry: airflow-docker-image
#####################################
ARTIFACT_REGISTRY_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
REPO_FORMAT="${REPO_FORMAT:-DOCKER}"
REGION="${GCP_LOCATION:-us-central1}"
echo "Checking Artifact Registry (${ARTIFACT_REGISTRY_NAME}) in region ${REGION}..."

EXISTING_REPO=""
for i in {1..3}; do
  echo "Attempt $i to list Artifact Registry repositories..."
  OUTPUT=$(gcloud artifacts repositories list \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --filter="name:${ARTIFACT_REGISTRY_NAME}" \
    --format="value(name)" 2>&1)
  
  # Trim whitespace from the output
  TRIMMED_OUTPUT=$(echo "$OUTPUT" | xargs)
  echo "Output: '$TRIMMED_OUTPUT'"
  
  if [[ "$TRIMMED_OUTPUT" =~ "Permission" ]]; then
    echo "Received permission error: $TRIMMED_OUTPUT"
    sleep 5
  elif [[ -n "$TRIMMED_OUTPUT" ]]; then
    EXISTING_REPO="$TRIMMED_OUTPUT"
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
