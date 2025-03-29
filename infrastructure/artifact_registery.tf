resource "google_artifact_registry_repository" "airflow_docker_repo" {
  repository_id     = var.artifact_registry_name  # e.g., "airflow-docker-image"
  name              = var.artifact_registry_name
  project           = var.project_id
  location          = "us"
  format            = var.repo_format
  description       = "Docker repository for data-pipeline"
}
