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

variable "gcp_service_account_key" {
  description = "Bootstrap service account key (in JSON format)"
  type        = string
  sensitive   = true
}
