resource "google_compute_instance_template" "airflow_template" {
  name_prefix  = "airflow-instance-"
  project      = var.project_id
  machine_type = var.machine_type

  disk {
    boot         = true
    auto_delete  = true
    source_image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
  }

  network_interface {
    network    = google_compute_network.airflow_vpc.self_link
    subnetwork = google_compute_subnetwork.airflow_subnet.self_link

    # This block allocates an ephemeral external IP to the instance.
    access_config {}
  }


  service_account {
    email  = var.service_account_email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }


  lifecycle {
    ignore_changes = [
      metadata_fingerprint,
      self_link
    ]
  }

  metadata = {
  startup-script = <<-EOF
      #!/bin/bash
      exec > /var/log/startup-script.log 2>&1
      set -ex

      # Retrieve secrets from GCP Secret Manager and export them as environment variables
      export POSTGRES_USER=$(gcloud secrets versions access latest --secret="postgres_user")
      export POSTGRES_PASSWORD=$(gcloud secrets versions access latest --secret="postgres_password")
      export POSTGRES_DB=$(gcloud secrets versions access latest --secret="postgres_db")
      export AIRFLOW_DATABASE_PASSWORD=$(gcloud secrets versions access latest --secret="airflow_database_password")
      export REDIS_PASSWORD=$(gcloud secrets versions access latest --secret="redis_password")
      export AIRFLOW_FERNET_KEY=$(gcloud secrets versions access latest --secret="airflow_fernet_key")
      export AIRFLOW_ADMIN_USERNAME=$(gcloud secrets versions access latest --secret="airflow_admin_username")
      export AIRFLOW_ADMIN_PASSWORD=$(gcloud secrets versions access latest --secret="airflow_admin_password")
      export AIRFLOW_ADMIN_FIRSTNAME=$(gcloud secrets versions access latest --secret="airflow_admin_firstname")
      export AIRFLOW_ADMIN_LASTNAME=$(gcloud secrets versions access latest --secret="airflow_admin_lastname")
      export AIRFLOW_ADMIN_EMAIL=$(gcloud secrets versions access latest --secret="airflow_admin_email")
      export AIRFLOW_UID=$(gcloud secrets versions access latest --secret="airflow_uid")
      export DOCKER_GID=$(gcloud secrets versions access latest --secret="docker_gid")


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
      git checkout testing-service-account-terraform

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
      echo "Creating GCP Key File..."
      cat > /opt/airflow/gcp-key.json <<EOKEY
      ${var.gcp_service_account_key}
      EOKEY

      
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

depends_on = [
    google_secret_manager_secret.postgres_user,
    google_secret_manager_secret.postgres_password,
    google_secret_manager_secret.postgres_db,
    google_secret_manager_secret.airflow_database_password,
    google_secret_manager_secret.redis_password,
    google_secret_manager_secret.airflow_fernet_key,
    google_secret_manager_secret.airflow_admin_username,
    google_secret_manager_secret.airflow_admin_password,
    google_secret_manager_secret.airflow_admin_firstname,
    google_secret_manager_secret.airflow_admin_lastname,
    google_secret_manager_secret.airflow_admin_email,
    google_secret_manager_secret.airflow_uid,
    google_secret_manager_secret.docker_gid,
  ]

  tags = ["airflow-server"]
}
