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
if gcloud iam service-accounts describe "${SA_TF_EMAIL}" --project "${PROJECT_ID}" &>/dev/null; then
  echo "Service Account ${SA_TF_EMAIL} exists. Importing into Terraform..."
  terraform import google_service_account.terraform_sa "projects/${PROJECT_ID}/serviceAccounts/${SA_TF_EMAIL}"
else
  echo "Service Account ${SA_TF_EMAIL} not found. Terraform will create it."
fi

# Import IAM roles for the Terraform service account.
roles=(
  "roles/artifactregistry.admin"
  "roles/artifactregistry.createOnPushRepoAdmin"
  "roles/artifactregistry.reader"
  "roles/artifactregistry.repoAdmin"
  "roles/artifactregistry.writer"
  "roles/composer.admin"
  "roles/cloudfunctions.admin"
  "roles/cloudsql.admin"
  "roles/compute.admin"
  "roles/compute.serviceAgent"
  "roles/compute.storageAdmin"
  "roles/iam.serviceAccountCreator"
  "roles/resourcemanager.projectIamAdmin"
  "roles/secretmanager.admin"
  "roles/servicenetworking.networksAdmin"
  "roles/storage.admin"
)

for role in "${roles[@]}"; do
  IMPORT_ID="${PROJECT_ID}/${role}/serviceAccount:${SA_TF_EMAIL}"
  echo "Importing IAM binding for role ${role} on ${SA_TF_EMAIL}..."
  terraform import "google_project_iam_member.terraform_sa_roles[\"${role}\"]" "${IMPORT_ID}" || \
    echo "Role ${role} not assigned or already imported."
done

#####################################
# 2. Import VM Service Account
#####################################

# Define the VM service account details
SA_VM_ID="vm-service-account"
SA_VM_EMAIL="${SA_VM_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== Checking VM Service Account: ${SA_VM_EMAIL} ==="
if gcloud iam service-accounts describe "${SA_VM_EMAIL}" --project "${PROJECT_ID}" &>/dev/null; then
  echo "Service Account ${SA_VM_EMAIL} exists. Importing into Terraform..."
  terraform import google_service_account.vm_service_account "projects/${PROJECT_ID}/serviceAccounts/${SA_VM_EMAIL}"
else
  echo "Service Account ${SA_VM_EMAIL} not found. Terraform will create it."
fi

# Import IAM roles for the VM service account.
vm_roles=(
  "roles/composer.admin"
  "roles/composer.serviceAgent"
  "roles/compute.admin"
  "roles/compute.instanceAdmin.v1"
  "roles/compute.viewer"
  "roles/secretmanager.admin"
  "roles/secretmanager.secretAccessor"
  "roles/storage.admin"
  "roles/storage.objectAdmin"
  "roles/storage.objectViewer"
)

for role in "${vm_roles[@]}"; do
  IMPORT_ID="${PROJECT_ID}/${role}/serviceAccount:${SA_VM_EMAIL}"
  echo "Importing IAM binding for role ${role} on ${SA_VM_EMAIL}..."
  terraform import "google_project_iam_member.vm_service_account_roles[\"${role}\"]" "${IMPORT_ID}" || \
    echo "Role ${role} not assigned or already imported."
done

echo "=== Import process completed ==="