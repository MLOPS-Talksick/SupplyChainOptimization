resource "google_cloud_run_service" "backend" {
  name     = "fastapi-backend"
  location = var.region

  template {
    spec {
      containers {
        image = var.backend_image_uri
        env {
          name  = "VM_IP"
          value = var.airflow_lb_ip
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}