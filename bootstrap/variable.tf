variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region"
  default     = "us-central1"
}

variable "gcp_service_account_key" {
  description = "Bootstrap service account key (in JSON format)"
  type        = string
  sensitive   = true
}

variable "artifact_registry_name" {
  description = "Name of the artifact registry repository."
  type        = string
  default     = "airflow-docker-image"
}

variable "repo_format" {
  description = "Format for the repository. For Docker, use 'DOCKER'."
  type        = string
  default     = "DOCKER"
}

