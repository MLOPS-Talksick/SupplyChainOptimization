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
  type        = string
  sensitive   = true
  description = "GCP Service Account key in JSON format"
}

variable "service_account_email" {
  description = "Email of the Terraform service account"
  type        = string
}


variable "database_name" {
  description = "Name of the database to create"
  type        = string
  default     = "transaction"
}

variable "allocated_ip_range_name" {
  description = "Name for the reserved IP range for Cloud SQL private connectivity"
  type        = string
  default     = "sql-private-ip-range"
}


variable "bucket_names" {
  type    = set(string)
  default = ["full-raw-data-test", "fully-processed-data-test"]
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


variable "postgres_user" {
  description = "PostgreSQL user"
  type        = string
  sensitive   = true
}

variable "postgres_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
}

variable "postgres_db" {
  description = "PostgreSQL database name"
  type        = string
  sensitive   = true
}

variable "airflow_database_password" {
  description = "Airflow DB password"
  type        = string
  sensitive   = true
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
}

variable "airflow_fernet_key" {
  description = "Airflow Fernet key"
  type        = string
  sensitive   = true
}

variable "airflow_admin_username" {
  description = "Airflow admin username"
  type        = string
  sensitive   = true
}

variable "airflow_admin_password" {
  description = "Airflow admin password"
  type        = string
  sensitive   = true
}

variable "airflow_admin_firstname" {
  description = "Airflow admin first name"
  type        = string
  sensitive   = true
}

variable "airflow_admin_lastname" {
  description = "Airflow admin last name"
  type        = string
  sensitive   = true
}

variable "airflow_admin_email" {
  description = "Airflow admin email"
  type        = string
  sensitive   = true
}

variable "airflow_uid" {
  description = "Airflow UID"
  type        = string
  sensitive   = true
}

variable "docker_gid" {
  description = "Docker group ID"
  type        = string
  sensitive   = true
}

variable "airflow_image_name" {
  description = "Docker group ID"
  type        = string
  default     = "apache/airflow:2.7.3-python3.10"
}