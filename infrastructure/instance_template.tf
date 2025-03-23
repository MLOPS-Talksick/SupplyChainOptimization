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
    startup-script = file("scripts\\startup_script.sh")
  }

  tags = ["airflow-server"]
}
