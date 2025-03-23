#!/bin/bash
exec > /var/log/startup-script.log 2>&1
set -ex

# Install Docker if it's not already installed
if ! command -v docker &>/dev/null; then
    sudo apt-get update -y
    echo "Adding Docker repository..."
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update -y
    sudo apt-get install -y docker.io docker-ce docker-ce-cli containerd.io
fi

# Install Docker Compose if it's not already installed
if ! command -v docker-compose &>/dev/null; then
    echo "Docker Compose not found. Installing latest version..."
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K.*?(?=")')
    sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
fi

# Install Git if it's not already installed
if ! command -v git &>/dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y git
fi

# Create /opt/airflow if it doesn't exist and change to that directory
mkdir -p /opt/airflow
cd /opt/airflow

# Clone your repository (replace with your actual repo URL)
git clone https://github.com/MLOPS-Talksick/SupplyChainOptimization.git .

# Optionally check out a specific branch or tag
# git checkout terraform-infra-meet-2

# Configure Docker access
sudo usermod -aG docker ubuntu
newgrp docker
sudo systemctl restart docker
sudo chmod 666 /var/run/docker.sock

# Ensure GCP key file is created (assuming the variable is set via Terraform)
if [ -f /opt/airflow/gcp-key.json ]; then
    sudo rm -f /opt/airflow/gcp-key.json
fi
echo "${var.gcp_service_account_key}" | jq . > /opt/airflow/gcp-key.json
chmod 644 /opt/airflow/gcp-key.json
sudo chown ubuntu:docker /opt/airflow/gcp-key.json

# Fix log directory permissions
sudo mkdir -p /opt/airflow/logs
sudo chmod -R 777 /opt/airflow/logs
sudo chown -R ubuntu:ubuntu /opt/airflow/logs

# Pull the latest image from Artifact Registry and restart containers
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker compose pull || true
docker compose down || true
docker volume rm airflow_postgres-db-volume || true
docker compose up -d --remove-orphans

echo "Airflow containers are now running!"