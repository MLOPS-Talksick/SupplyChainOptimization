variable "project_id" {
  type        = string
  description = "GCP project ID"
  default = "primordial-veld-450618-n4"
}

variable "region" {
  type        = string
  description = "GCP region"
  default     = "us-central1"
}

variable "airflow_lb_ip" {
  description = "The IP of the Airflow load balancer"
  type        = string
}

variable "backend_image_uri" {
  type        = string
  description = "URI for the backend Docker image in Artifact Registry"
}