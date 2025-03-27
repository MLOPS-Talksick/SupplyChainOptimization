terraform {
  required_version = ">= 1.0.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"  # or latest stable
    }

    mysql = {
      source  = "petoju/mysql"
      version = "~> 3.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}


provider "google-beta" {
  region = "us-central1"
  zone   = "us-central1-a"
}