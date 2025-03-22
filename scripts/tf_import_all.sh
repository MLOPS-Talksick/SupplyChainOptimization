#!/bin/bash
set -e

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

######################################
# 2. IAM Binding for Artifact Registry Reader
######################################
IAM_MEMBER="serviceAccount:${SA_EMAIL}"
ROLE="roles/artifactregistry.reader"
# Encode the slash: replace "/" with "~"
ENCODED_ROLE=$(echo "$ROLE" | sed 's/\//~/g')
echo "Checking IAM Binding for ${IAM_MEMBER} with role ${ROLE}..."
if gcloud projects get-iam-policy "${PROJECT_ID}" --flatten="bindings[].members" \
    --format="table(bindings.role)" --filter="bindings.role=${ROLE} AND bindings.members:${IAM_MEMBER}" | grep "${ROLE}" &>/dev/null; then
    echo "IAM Binding exists. Importing..."
    terraform import google_project_iam_member.airflow_sa_artifact_registry "${PROJECT_ID}/${ENCODED_ROLE}/${IAM_MEMBER}"
else
    echo "IAM Binding not found. Terraform will create it."
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

echo "=== Import Check Completed ==="
