resource "google_compute_region_instance_group_manager" "airflow_mig" {
  name               = "airflow-mig"
  project            = var.project_id
  region             = var.region
  base_instance_name = "airflow-instance"
  target_size        = 1

  version {
    instance_template = google_compute_instance_template.airflow_template.self_link
  }
  
  named_port {
    name = "http"
    port = 8080
  }
}
