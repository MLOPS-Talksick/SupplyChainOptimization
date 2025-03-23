resource "google_compute_instance_template" "airflow_template" {
  name_prefix  = "airflow-instance-"
  project      = var.project_id
  machine_type = var.machine_type

  disk {
    # Use a custom image that already has Docker and Docker Compose installed,
    # or if you want to install them at boot, use a base image (e.g., Ubuntu)
    source_image = var.custom_image_name != "" ? var.custom_image_name : "projects/ubuntu-os-cloud/global/images/family/${var.image_family}"
    auto_delete  = true
    boot         = true
  }

  network_interface {
    network    = google_compute_network.airflow_vpc.self_link
    subnetwork = google_compute_subnetwork.airflow_subnet.self_link

    # This block allocates an ephemeral external IP to the instance.
    access_config {}
  }


  service_account {
    email  = google_service_account.airflow_sa.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }


  metadata = {
    # Startup script to run docker-compose up -d
    startup-script = <<-EOF
      #!/bin/bash
      exec > /var/log/startup-script.log 2>&1
      set -ex

      # Install Docker if it's not already installed
      if ! command -v docker &>/dev/null; then
          sudo apt-get update -y
          echo "Adding Docker repository..."
          sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
          curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
          sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $$ (lsb_release -cs) stable"
          sudo apt-get update -y
          sudo apt-get install -y docker-ce docker-ce-cli containerd.io
      fi

      # Install Docker Compose if it's not already installed
      if ! command -v docker-compose &>/dev/null; then
          echo "❌ Docker Compose not found. Installing latest version..."
          DOCKER_COMPOSE_VERSION=\$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K.*?(?=")')
          sudo curl -L "https://github.com/docker/compose/releases/download/\${DOCKER_COMPOSE_VERSION}/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
          sudo chmod +x /usr/local/bin/docker-compose
          sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
      fi

      # Install Git if it's not already installed
      if ! command -v git &>/dev/null; then
          apt-get update -y
          apt-get install -y git
      fi

      # Clone your repository (replace the URL with your repo)
      git clone https://github.com/MLOPS-Talksick/SupplyChainOptimization.git .

      # Optionally, check out a specific branch or tag:
      # git checkout terraform-infra-meet-2

      sudo usermod -aG docker ubuntu
      newgrp docker
      sudo systemctl restart docker
      sudo chmod 666 /var/run/docker.sock

      mkdir -p /opt/airflow
      echo "airflow dir created."
      echo "🚀 Ensuring GCP Key File exists..."
      if [ -d /opt/airflow/gcp-key.json ]; then
          echo "⚠️ Found directory at /opt/airflow/gcp-key.json. Removing it..."
          sudo rm -rf /opt/airflow/gcp-key.json
      fi
      echo "🚀 Creating GCP Key File..."
      echo "${var.gcp_service_account_key}" | jq . > /opt/airflow/gcp-key.json
      chmod 644 /opt/airflow/gcp-key.json
      sudo chown ubuntu:docker /opt/airflow/gcp-key.json
      echo "✅ GCP Key File Created."

      echo "🚀 Fixing Airflow log directory permissions..."
      sudo mkdir -p /opt/airflow/logs
      sudo chmod -R 777 /opt/airflow/logs
      sudo chown -R \$USER:\$USER /opt/airflow/logs
      
      cd /opt/airflow

      echo "🚀 Pulling the latest image from Artifact Registry..."
      gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
      docker compose pull || true

      echo "🚀 Stopping any running containers..."
      docker compose down || true

      # Remove postgres volume if you want to reset the DB (warning: this clears data)
      docker volume rm airflow_postgres-db-volume || true

      echo "🚀 Starting Airflow using Docker Compose..."
      docker compose up -d --remove-orphans

      echo "✅ Airflow successfully started!"
    EOF
  }

  tags = ["airflow-server"]
}
