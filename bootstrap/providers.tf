provider "google" {
  credentials = var.bootstrap_gcp_key
  project     = var.project_id
  region      = var.region
}