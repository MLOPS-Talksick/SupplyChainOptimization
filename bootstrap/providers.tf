terraform {
  required_version = ">= 1.0.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.61.0"
    }
  }
}

provider "google" {
  credentials = var.bootstrap_gcp_key
  project     = var.project_id
  region      = var.region
}