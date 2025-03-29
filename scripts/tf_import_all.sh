#!/bin/bash
set -e


# Ensure the GCP_SERVICE_ACCOUNT_KEY is set in the environment (from GitHub Secrets)
if [ -z "$GCP_SERVICE_ACCOUNT_KEY" ]; then
  echo "GCP_SERVICE_ACCOUNT_KEY is not set"
  exit 1
fi

# Export the variable for Terraform
export TF_VAR_gcp_service_account_key="$GCP_SERVICE_ACCOUNT_KEY"

# Change directory to the folder containing Terraform configuration files
cd infrastructure

# Variables – adjust these as needed or export them from your environment
PROJECT_ID="primordial-veld-450618-n4"
REGION="us-central1"
GLOBAL_FLAG="--global"

echo "=== Checking and Importing Existing Resources into Terraform State ==="

######################################
# 1. Service Account: airflow-service-account
######################################
SA_EMAIL="airflow-service-account@${PROJECT_ID}.iam.gserviceaccount.com"
echo "Checking Service Account (${SA_EMAIL})..."
if gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Service Account exists. Importing..."
    terraform import google_service_account.airflow_sa "projects/${PROJECT_ID}/serviceAccounts/${SA_EMAIL}"
else
    echo "Service Account not found. It will be created by Terraform."
fi

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


######################################
# Artifact Registry: airflow-docker-image
######################################
ARTIFACT_REGISTRY_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
REPO_FORMAT="${REPO_FORMAT:-docker}"
echo "Checking Artifact Registry (${ARTIFACT_REGISTRY_NAME})..."
EXISTING_REPO=$(gcloud artifacts repositories list \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --filter="name=projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REGISTRY_NAME}" \
  --format="value(name)")
if [[ -n "$EXISTING_REPO" ]]; then
    echo "Artifact Registry ${ARTIFACT_REGISTRY_NAME} exists. Importing..."
    terraform import google_artifact_registry_repository.airflow_docker_repo "projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REGISTRY_NAME}"
else
    echo "Artifact Registry ${ARTIFACT_REGISTRY_NAME} not found. Terraform will create it."
fi


######################################
# 3. VPC Network: airflow-network
######################################
NETWORK_NAME="airflow-network"
echo "Checking VPC network (${NETWORK_NAME})..."
if gcloud compute networks describe "${NETWORK_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "VPC network exists. Importing..."
    terraform import google_compute_network.airflow_vpc "projects/${PROJECT_ID}/global/networks/${NETWORK_NAME}"
else
    echo "VPC network not found. Terraform will create it."
fi

######################################
# 3.1 VPC Network: my-vpc-network (my_network)
######################################
MY_NETWORK_NAME="my-vpc-network"
echo "Checking VPC network (${MY_NETWORK_NAME})..."
if gcloud compute networks describe "${MY_NETWORK_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "VPC network ${MY_NETWORK_NAME} exists. Importing..."
    terraform import google_compute_network.my_network "projects/${PROJECT_ID}/global/networks/${MY_NETWORK_NAME}"
else
    echo "VPC network ${MY_NETWORK_NAME} not found. Terraform will create it."
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
# 4.1 Subnetwork: my-subnet (my_subnet)
######################################
MY_SUBNET="my-subnet"
echo "Checking Subnetwork (${MY_SUBNET}) in region ${REGION}..."
if gcloud compute networks subnets describe "${MY_SUBNET}" --region "${REGION}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "Subnetwork ${MY_SUBNET} exists. Importing..."
    terraform import google_compute_subnetwork.my_subnet "projects/${PROJECT_ID}/regions/${REGION}/subnetworks/${MY_SUBNET}"
else
    echo "Subnetwork ${MY_SUBNET} not found. Terraform will create it."
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
URL_MAP_NAME="airflow-url-map"
echo "Checking URL Map (${URL_MAP_NAME})..."
if gcloud compute url-maps describe "${URL_MAP_NAME}" --project "${PROJECT_ID}" &>/dev/null; then
    echo "URL Map exists. Importing..."
    terraform import google_compute_url_map.airflow_url_map "projects/${PROJECT_ID}/global/urlMaps/${URL_MAP_NAME}"
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
INSTANCE_NAME="transaction-database"
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


echo "=== Import Check Completed ==="
