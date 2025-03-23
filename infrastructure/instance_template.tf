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
  startup-script = <<-EOF
      #!/bin/bash
      exec > /var/log/startup-script.log 2>&1
      set -ex

      # Update package lists and install Docker, Docker Compose, and Git using apt-get
      sudo apt-get update -y
      sudo apt-get install -y docker.io docker-compose git

      # Start and enable Docker service
      sudo systemctl start docker
      sudo systemctl enable docker

      # Create /opt/airflow if it doesn't exist and switch to that directory
      mkdir -p /opt/airflow
      cd /opt/airflow

      # Clone your repository (replace the URL with your repository URL)
      git clone https://github.com/MLOPS-Talksick/SupplyChainOptimization.git .

      # Optionally, check out a specific branch or tag:
      git checkout terraform-infra-meet-2

      # Add the ubuntu user to the docker group and adjust permissions
      sudo usermod -aG docker ubuntu
      sudo chmod 666 /var/run/docker.sock

      echo "airflow dir created."

      # Ensure the GCP key file is set up
      echo "Ensuring GCP Key File exists..."
      if [ -f /opt/airflow/gcp-key.json ]; then
          echo "Found file at /opt/airflow/gcp-key.json. Removing it..."
          sudo rm -f /opt/airflow/gcp-key.json
      fi
      echo "Creating GCP Key File..."
      printf '%b' "${var.gcp_service_account_key}" | jq . > /opt/airflow/gcp-key.json
      chmod 644 /opt/airflow/gcp-key.json
      sudo chown ubuntu:docker /opt/airflow/gcp-key.json
      echo "GCP Key File Created."

      # Fix permissions for Airflow logs
      echo "Fixing Airflow log directory permissions..."
      sudo mkdir -p /opt/airflow/logs
      sudo chmod -R 777 /opt/airflow/logs
      sudo chown -R ubuntu:docker /opt/airflow/logs

      cd /opt/airflow

      # Configure Docker authentication for Artifact Registry
      gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

      # Pull the latest images using docker-compose
      docker-compose pull || true

      # Stop and remove any running containers
      docker-compose down || true

      # Optionally remove the Postgres volume if you want to reset the DB (warning: this clears data)
      docker volume rm airflow_postgres-db-volume || true

      # Start the services using docker-compose
      docker-compose up -d --remove-orphans

      echo "Airflow successfully started!"
  EOF
}

  tags = ["airflow-server"]
}
