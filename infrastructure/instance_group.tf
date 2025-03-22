resource "google_compute_region_instance_group_manager" "airflow_mig" {
  name               = "airflow-mig"
  project            = var.project_id
  region             = var.region
  base_instance_name = "airflow-instance"
  version {
    instance_template = google_compute_instance_template.airflow_template.self_link
  }
  target_size = 1
  // Additional configuration such as auto-healing or auto-scaling...
}
