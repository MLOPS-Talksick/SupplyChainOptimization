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
    startup-script = <<-EOT
      #!/bin/bash
      set -ex

      # Install Docker if it's not already installed (if needed)
      if ! command -v docker &>/dev/null; then
          apt-get update -y
          apt-get install -y docker.io
      fi

      # Optionally install Docker Compose if not available
      if ! command -v docker-compose &>/dev/null; then
          apt-get install -y docker-compose
      fi

      # Navigate to the directory containing your docker-compose.yml
      cd /opt/airflow

      # Run docker-compose to start your services
      docker compose down || true
      docker volume rm airflow_postgres-db-volume || true
      docker compose up -d --remove-orphans

      echo "✅ Airflow containers are now running!"

    EOT
  }

  tags = ["airflow-server"]
}
