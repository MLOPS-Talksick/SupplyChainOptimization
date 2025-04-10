#!/bin/bash
set -e

# Change directory to the folder containing Terraform configuration files
cd infrastructure

# Variables – adjust these as needed or export them from your environment
PROJECT_ID="primordial-veld-450618-n4"
REGION="us-central1"
GLOBAL_FLAG="--global"

echo "=== Checking and Importing Existing Resources into Terraform State ==="

# ######################################
# # 1. Service Account: airflow-service-account
# ######################################
# SA_EMAIL="airflow-service-account@${PROJECT_ID}.iam.gserviceaccount.com"
# echo "Checking Service Account (${SA_EMAIL})..."
# if gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" &>/dev/null; then
#     echo "Service Account exists. Importing..."
#     terraform import google_service_account.terraform_sa "projects/${PROJECT_ID}/serviceAccounts/${SA_EMAIL}"
# else
#     echo "Service Account not found. It will be created by Terraform."
# fi

# ######################################
# # 2. IAM Binding for Artifact Registry Reader
# ######################################
# IAM_MEMBER="serviceAccount:${SA_EMAIL}"
# ROLE="roles/artifactregistry.reader"
# echo "Checking IAM Binding for ${IAM_MEMBER} with role ${ROLE}..."
# if gcloud projects get-iam-policy "${PROJECT_ID}" --flatten="bindings[].members" \
#     --format="table(bindings.role)" --filter="bindings.role=${ROLE} AND bindings.members:${IAM_MEMBER}" | grep "${ROLE}" &>/dev/null; then
#     echo "IAM Binding exists. Importing..."
#     terraform import google_project_iam_member.airflow_sa_artifact_registry "${PROJECT_ID}/${ROLE}/${IAM_MEMBER}"
# else
#     echo "IAM Binding not found. Terraform will create it."
# fi


# Declare an associative array mapping secret names to their Terraform secret resource addresses.
declare -A secrets=(
  ["postgres_user"]="google_secret_manager_secret.postgres_user"
  ["postgres_password"]="google_secret_manager_secret.postgres_password"
  ["postgres_db"]="google_secret_manager_secret.postgres_db"
  ["airflow_database_password"]="google_secret_manager_secret.airflow_database_password"
  ["redis_password"]="google_secret_manager_secret.redis_password"
  ["airflow_fernet_key"]="google_secret_manager_secret.airflow_fernet_key"
  ["airflow_admin_username"]="google_secret_manager_secret.airflow_admin_username"
  ["airflow_admin_password"]="google_secret_manager_secret.airflow_admin_password"
  ["airflow_admin_firstname"]="google_secret_manager_secret.airflow_admin_firstname"
  ["airflow_admin_lastname"]="google_secret_manager_secret.airflow_admin_lastname"
  ["airflow_admin_email"]="google_secret_manager_secret.airflow_admin_email"
  ["airflow_uid"]="google_secret_manager_secret.airflow_uid"
  ["docker_gid"]="google_secret_manager_secret.docker_gid"
)

# Function: Import a secret and its version if it exists in GCP and isn't already in Terraform state.
import_secret() {
  local secret_name="$1"
  local resource_address="$2"
  # Extract the suffix from the resource address (e.g., "postgres_db" from "google_secret_manager_secret.postgres_db")
  local resource_suffix="${resource_address#*.}"
  # Build the version resource address with the correct type prefix.
  local version_resource_address="google_secret_manager_secret_version.${resource_suffix}_version"
  local secret_id="projects/${PROJECT_ID}/secrets/${secret_name}"

  # Check if the secret exists in GCP.
  if gcloud secrets describe "${secret_name}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Secret ${secret_name} exists in GCP."

    # Import the secret if not already in Terraform state.
    if ! terraform state list | grep -q "${resource_address}"; then
      echo "Importing ${resource_address}..."
      terraform import "${resource_address}" "${secret_id}"
    else
      echo "${resource_address} is already imported."
    fi

    # Import the secret version (assumed to be version 1) if not already imported.
    local version_id="${secret_id}/versions/1"
    if ! terraform state list | grep -q "${version_resource_address}"; then
      echo "Importing ${version_resource_address}..."
      terraform import "${version_resource_address}" "${version_id}"
    else
      echo "${version_resource_address} is already imported."
    fi
  else
    echo "Secret ${secret_name} does not exist in GCP. Skipping import."
  fi
}

echo "Starting secret import process..."
for secret in "${!secrets[@]}"; do
  import_secret "$secret" "${secrets[$secret]}"
  echo "---------------------------------------------"
done
echo "Secret import process completed."


######################################
# 3. VPC Network: airflow-vpc
######################################
NETWORK_NAME="airflow-vpc"
echo "Checking VPC network (${NETWORK_NAME})..."
if gcloud compute networks describe "${NETWORK_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "VPC network exists. Importing..."
    terraform import google_compute_network.airflow_vpc "projects/${PROJECT_ID}/global/networks/${NETWORK_NAME}"
else
    echo "VPC network not found. Terraform will create it."
fi

######################################
# 4. Subnetwork: airflow-subnet
######################################
SUBNET_NAME="airflow-subnet"
echo "Checking Subnetwork (${SUBNET_NAME}) in region ${REGION}..."
if gcloud compute networks subnets describe "${SUBNET_NAME}" --region "${REGION}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Subnetwork exists. Importing..."
    terraform import google_compute_subnetwork.airflow_subnet "projects/${PROJECT_ID}/regions/${REGION}/subnetworks/${SUBNET_NAME}"
else
    echo "Subnetwork not found. Terraform will create it."
fi



######################################
# 4.2 Global Address: private-ip-range
######################################
GLOBAL_ADDRESS="private-ip-range"
echo "Checking Global Address (${GLOBAL_ADDRESS})..."
if gcloud compute addresses describe "${GLOBAL_ADDRESS}" --global --project "${PROJECT_ID}" &>/dev/null; then
    echo "Global Address ${GLOBAL_ADDRESS} exists. Importing..."
    terraform import google_compute_global_address.private_ip_range "projects/${PROJECT_ID}/global/addresses/${GLOBAL_ADDRESS}"
else
    echo "Global Address ${GLOBAL_ADDRESS} not found. Terraform will create it."
fi

######################################
# 5. Firewall Rule: allow-airflow-server
######################################
FIREWALL_NAME="allow-airflow-server"
echo "Checking Firewall Rule (${FIREWALL_NAME})..."
if gcloud compute firewall-rules describe "${FIREWALL_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Firewall Rule exists. Importing..."
    terraform import google_compute_firewall.airflow_firewall "projects/${PROJECT_ID}/global/firewalls/${FIREWALL_NAME}"
else
    echo "Firewall Rule not found. Terraform will create it."
fi


######################################
# 5.1 Firewall: allow-internal-sql (allow_internal_sql)
######################################
SQL_FIREWALL="allow-internal-sql"
echo "Checking Firewall (${SQL_FIREWALL})..."
if gcloud compute firewall-rules describe "${SQL_FIREWALL}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Firewall ${SQL_FIREWALL} exists. Importing..."
    terraform import google_compute_firewall.allow_internal_sql "projects/${PROJECT_ID}/global/firewalls/${SQL_FIREWALL}"
else
    echo "Firewall ${SQL_FIREWALL} not found. Terraform will create it."
fi


######################################
# 6. Instance Template: airflow-instance-template
######################################
INSTANCE_TEMPLATE_NAME="airflow-instance-template"
echo "Checking Instance Template (${INSTANCE_TEMPLATE_NAME})..."
if gcloud compute instance-templates describe "${INSTANCE_TEMPLATE_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Instance Template exists. Importing..."
    terraform import google_compute_instance_template.airflow_template "projects/${PROJECT_ID}/global/instanceTemplates/${INSTANCE_TEMPLATE_NAME}"
else
    echo "Instance Template not found. Terraform will create it."
fi


######################################
# 7. Instance Group Manager: airflow-mig
######################################
IGM_NAME="airflow-mig"
echo "Checking Instance Group Manager (${IGM_NAME}) in region ${REGION}..."
if gcloud compute instance-groups managed describe "${IGM_NAME}" --region "${REGION}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Instance Group Manager exists. Importing..."
    terraform import google_compute_region_instance_group_manager.airflow_mig "projects/${PROJECT_ID}/regions/${REGION}/instanceGroupManagers/${IGM_NAME}"
else
    echo "Instance Group Manager not found. Terraform will create it."
fi

######################################
# 8. Health Check: airflow-health-check
######################################
HEALTH_CHECK_NAME="airflow-health-check"
echo "Checking Health Check (${HEALTH_CHECK_NAME})..."
if gcloud compute health-checks describe "${HEALTH_CHECK_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Health Check exists. Importing..."
    terraform import google_compute_health_check.http_health_check "projects/${PROJECT_ID}/global/healthChecks/${HEALTH_CHECK_NAME}"
else
    echo "Health Check not found. Terraform will create it."
fi

######################################
# 9. Backend Service: airflow-backend-service
######################################
BACKEND_SERVICE_NAME="airflow-backend-service"
echo "Checking Backend Service (${BACKEND_SERVICE_NAME})..."
if gcloud compute backend-services describe "${BACKEND_SERVICE_NAME}" --global --project "${PROJECT_ID}" &>/dev/null; then
    echo "Backend Service exists. Importing..."
    terraform import google_compute_backend_service.airflow_backend "projects/${PROJECT_ID}/global/backendServices/${BACKEND_SERVICE_NAME}"
else
    echo "Backend Service not found. Terraform will create it."
fi

######################################
# 10. URL Map: airflow-url-map
######################################
URL_MAP_NAME="lb-url-map"
echo "Checking URL Map (${URL_MAP_NAME})..."
if gcloud compute url-maps describe "${URL_MAP_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "URL Map exists. Importing..."
    terraform import google_compute_url_map.lb_url_map "projects/${PROJECT_ID}/global/urlMaps/${URL_MAP_NAME}"
else
    echo "URL Map not found. Terraform will create it."
fi

######################################
# 11. Target HTTP Proxy: airflow-http-proxy
######################################
HTTP_PROXY_NAME="airflow-http-proxy"
echo "Checking Target HTTP Proxy (${HTTP_PROXY_NAME})..."
if gcloud compute target-http-proxies describe "${HTTP_PROXY_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Target HTTP Proxy exists. Importing..."
    terraform import google_compute_target_http_proxy.airflow_http_proxy "projects/${PROJECT_ID}/global/targetHttpProxies/${HTTP_PROXY_NAME}"
else
    echo "Target HTTP Proxy not found. Terraform will create it."
fi

######################################
# 12. Global Forwarding Rule: airflow-global-forwarding-rule
######################################
FORWARDING_RULE_NAME="airflow-global-forwarding-rule"
echo "Checking Global Forwarding Rule (${FORWARDING_RULE_NAME})..."
if gcloud compute forwarding-rules describe "${FORWARDING_RULE_NAME}" ${GLOBAL_FLAG} --project "${PROJECT_ID}" &>/dev/null; then
    echo "Global Forwarding Rule exists. Importing..."
    terraform import google_compute_global_forwarding_rule.airflow_http_forwarding_rule "projects/${PROJECT_ID}/global/forwardingRules/${FORWARDING_RULE_NAME}"
else
    echo "Global Forwarding Rule not found. Terraform will create it."
fi


######################################
# Autoscaler: airflow-autoscaler
######################################
AUTOSCALER_NAME="airflow-autoscaler"
echo "Attempting to import Autoscaler (${AUTOSCALER_NAME}) in region ${REGION}..."
terraform import google_compute_region_autoscaler.airflow_autoscaler "projects/${PROJECT_ID}/regions/${REGION}/autoscalers/${AUTOSCALER_NAME}" || echo "Autoscaler already exists; skipping import."


######################################
# 13. Cloud SQL Instance: transaction-database
######################################
INSTANCE_NAME="mlops-sql-2"
echo "Checking Cloud SQL Instance (${INSTANCE_NAME})..."
if gcloud sql instances describe "${INSTANCE_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Cloud SQL Instance exists. Importing..."
    terraform import google_sql_database_instance.instance "${INSTANCE_NAME}"
else
    echo "Cloud SQL Instance not found. Terraform will create it."
fi

######################################
# 14. Cloud SQL Database: ${DATABASE_NAME}
######################################
DATABASE_NAME="transaction"  # or set DATABASE_NAME="your_database_name" if not using a variable
echo "Checking Cloud SQL Database (${DATABASE_NAME})..."
if gcloud sql databases describe "${DATABASE_NAME}" --instance="${INSTANCE_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Cloud SQL Database exists. Importing..."
    terraform import google_sql_database.database "${INSTANCE_NAME}/${DATABASE_NAME}"
else
    echo "Cloud SQL Database not found. Terraform will create it."
fi


######################################
# 16. GCS Buckets
######################################
# List of GCS bucket names managed by Terraform
BUCKETS=("full-raw-data-test" "fully-processed-data-test")

for bucket in "${BUCKETS[@]}"; do
    echo "Checking GCS Bucket (${bucket})..."
    if gsutil ls -b "gs://${bucket}" &>/dev/null; then
        echo "Bucket ${bucket} exists. Importing..."
        terraform import "google_storage_bucket.buckets[\"${bucket}\"]" "${bucket}"
    else
        echo "Bucket ${bucket} not found. Terraform will create it."
    fi
done

# 15. Cloud Function: processDataFunction
FUNCTION_NAME="processDataFunction"
echo "Checking Cloud Function (${FUNCTION_NAME}) in region ${REGION}..."
if gcloud functions describe "${FUNCTION_NAME}" --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Cloud Function ${FUNCTION_NAME} exists. Importing..."
    terraform import google_cloudfunctions_function.process_data_function "projects/${PROJECT_ID}/locations/${REGION}/functions/${FUNCTION_NAME}"
else
    echo "Cloud Function ${FUNCTION_NAME} not found. Terraform will create it."
fi

######################################
# 17. Cloud Run V2 Services
######################################

# Cloud Run V2 Service: model-serving
MODEL_SERVING="model-serving"
echo "Checking Cloud Run V2 Service (${MODEL_SERVING})..."
if gcloud run services describe "${MODEL_SERVING}" --region="${REGION}" --project="${PROJECT_ID}" --platform=managed &>/dev/null; then
    echo "Cloud Run Service ${MODEL_SERVING} exists. Importing..."
    terraform import google_cloud_run_v2_service.model_serving "projects/${PROJECT_ID}/locations/${REGION}/services/${MODEL_SERVING}"
else
    echo "Cloud Run Service ${MODEL_SERVING} not found. Terraform will create it."
fi

# Cloud Run V2 Service: model-training-trigger
MODEL_TRAINING_TRIGGER="model-training-trigger"
echo "Checking Cloud Run V2 Service (${MODEL_TRAINING_TRIGGER})..."
if gcloud run services describe "${MODEL_TRAINING_TRIGGER}" --region="${REGION}" --project="${PROJECT_ID}" --platform=managed &>/dev/null; then
    echo "Cloud Run Service ${MODEL_TRAINING_TRIGGER} exists. Importing..."
    terraform import google_cloud_run_v2_service.model_training_trigger "projects/${PROJECT_ID}/locations/${REGION}/services/${MODEL_TRAINING_TRIGGER}"
else
    echo "Cloud Run Service ${MODEL_TRAINING_TRIGGER} not found. Terraform will create it."
fi

######################################
# 18. Cloud Run V2 Job: model-training-job
######################################
MODEL_TRAINING_JOB="model-training-job"
echo "Checking Cloud Run V2 Job (${MODEL_TRAINING_JOB})..."

# Use correct gcloud command without --platform for V2
if gcloud run jobs describe "${MODEL_TRAINING_JOB}" --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "✅ Cloud Run Job '${MODEL_TRAINING_JOB}' exists. Importing to Terraform..."
    terraform import google_cloud_run_v2_job.model_training_job "projects/${PROJECT_ID}/locations/${REGION}/jobs/${MODEL_TRAINING_JOB}"
else
    echo "❌ Cloud Run Job '${MODEL_TRAINING_JOB}' not found. Terraform will create it."
fi

echo "✅ Import check completed."


######################################
# 19. Cloud Run V2 Service: backend
######################################
CLOUDRUN_BACKEND="cloudrun-backend"
echo "Checking Cloud Run V2 Service (${CLOUDRUN_BACKEND})..."

# Check if service exists
if gcloud run services describe "${CLOUDRUN_BACKEND}" --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "✅ Cloud Run V2 Service '${CLOUDRUN_BACKEND}' exists. Importing to Terraform..."
    terraform import google_cloud_run_v2_service.backend "projects/${PROJECT_ID}/locations/${REGION}/services/${CLOUDRUN_BACKEND}"
else
    echo "❌ Cloud Run V2 Service '${CLOUDRUN_BACKEND}' not found. Terraform will create it."
fi

echo "✅ Import check completed."


### Firewall rule for cloud Run
FIREWALL_RULE="allow-cloudrun-to-airflow"
echo "Checking if firewall rule (${FIREWALL_RULE}) exists..."
if gcloud compute firewall-rules describe "${FIREWALL_RULE}" \
  --project="${PROJECT_ID}" &>/dev/null; then
  echo "✅ Firewall rule '${FIREWALL_RULE}' exists. Importing..."
  terraform import google_compute_firewall.allow_cloudrun_to_airflow \
    "projects/${PROJECT_ID}/global/firewalls/${FIREWALL_RULE}"
else
  echo "❌ Firewall rule '${FIREWALL_RULE}' not found. Terraform will create it."
fi


CLOUDRUN_CONNECTOR="cloudrun-connector"
echo "Checking VPC Access Connector (${CLOUDRUN_CONNECTOR})..."
if gcloud compute networks vpc-access connectors describe "${CLOUDRUN_CONNECTOR}" \
  --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "✅ VPC Connector '${CLOUDRUN_CONNECTOR}' exists. Importing..."
    terraform import google_vpc_access_connector.cloudrun_connector "projects/${PROJECT_ID}/locations/${REGION}/connectors/${CLOUDRUN_CONNECTOR}"
else
    echo "❌ VPC Connector '${CLOUDRUN_CONNECTOR}' not found. Terraform will create it."
fi


######################################
# 20. Serverless NEG for Cloud Run Backend
######################################
CLOUDRUN_NEG_NAME="cloudrun-neg"
echo "Checking NEG (${CLOUDRUN_NEG_NAME}) in region ${REGION}..."
if gcloud compute network-endpoint-groups describe "${CLOUDRUN_NEG_NAME}" --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "✅ Serverless NEG '${CLOUDRUN_NEG_NAME}' exists. Importing to Terraform..."
    terraform import google_compute_region_network_endpoint_group.cloudrun_neg "projects/${PROJECT_ID}/regions/${REGION}/networkEndpointGroups/${CLOUDRUN_NEG_NAME}"
else
    echo "❌ Serverless NEG '${CLOUDRUN_NEG_NAME}' not found. Terraform will create it."
fi


# Import: Backend Service for Cloud Run
BACKEND_SERVICE_NAME="cloudrun-backend"
echo "Checking Backend Service (${BACKEND_SERVICE_NAME})..."
if gcloud compute backend-services describe "${BACKEND_SERVICE_NAME}" --global --project="${PROJECT_ID}" &>/dev/null; then
    echo "✅ Backend Service exists. Importing..."
    terraform import google_compute_backend_service.cloudrun_backend "projects/${PROJECT_ID}/global/backendServices/${BACKEND_SERVICE_NAME}"
else
    echo "❌ Backend Service not found."
fi

# Import: Firewall Rule - allow-cloudrun-to-sql
FIREWALL_RULE="allow-cloudrun-to-sql"
echo "Checking Firewall rule (${FIREWALL_RULE})..."
if gcloud compute firewall-rules describe "${FIREWALL_RULE}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "✅ Firewall rule exists. Importing..."
    terraform import google_compute_firewall.allow_cloudrun_to_sql "projects/${PROJECT_ID}/global/firewalls/${FIREWALL_RULE}"
else
    echo "❌ Firewall rule not found."
fi

######################################
# Import Service Networking Peering Connection
######################################
echo "Checking if VPC peering with servicenetworking.googleapis.com exists on airflow-vpc..."
if gcloud services vpc-peerings list --network=airflow-vpc --project=primordial-veld-450618-n4 --format="value(peering)" | grep servicenetworking.googleapis.com &>/dev/null; then
    echo "✅ VPC Peering with servicenetworking.googleapis.com exists. Importing..."
    terraform import google_service_networking_connection.private_vpc_connection "projects/primordial-veld-450618-n4/global/networks/airflow-vpc/services/servicenetworking.googleapis.com"
else
    echo "❌ Peering connection not found or already imported."
fi
