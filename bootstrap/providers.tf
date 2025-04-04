terraform {
  required_version = ">= 1.0.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.27.0"
    }
  }
}

provider "google" {
  credentials = file(var.gcp_service_account_key)
  project     = var.project_id
  region      = var.region
}