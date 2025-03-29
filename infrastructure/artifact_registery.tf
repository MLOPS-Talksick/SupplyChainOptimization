resource "google_artifact_registry_repository" "airflow_docker_repo" {
  repository_id     = var.artifact_registry_name  # e.g., "airflow-docker-image"
  project           = var.project_id
  format            = var.repo_format             # e.g., "docker"
  description       = "Docker repository for data-pipeline"
}
