terraform {
  required_version = ">= 1.0.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.27.0"
    }
  }

  backend "gcs" {
    bucket = "my-terraform-state-bucket"   # Your bucket name
    prefix = "service-account/terraform"     # A directory-like prefix to isolate your state files
  }
}

provider "google" {
  credentials = var.gcp_service_account_key
  project     = var.project_id
  region      = var.region
}