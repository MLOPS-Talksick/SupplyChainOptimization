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

variable "zone" {
  type        = string
  description = "GCP zone"
  default     = "us-central1-a"
}

variable "network_name" {
  type        = string
  default     = "airflow-network"
}

variable "subnet_name" {
  type        = string
  default     = "airflow-subnet"
}

variable "machine_type" {
  type        = string
  default     = "e2-standard-4"
}

variable "image_family" {
  type        = string
  default     = "ubuntu-2204-lts"
}

# If you want a custom image name that you create from a base VM
variable "custom_image_name" {
  type        = string
  default     = ""
}

variable "gcp_service_account_key" {
  type      = string
  sensitive = true
  description = "GCP Service Account key in JSON format"
}
