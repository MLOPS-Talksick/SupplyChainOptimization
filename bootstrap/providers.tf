provider "google" {
  project = var.project_id
  region  = var.region
  # Intentionally no "credentials = ..." line,
  # so it picks up credentials from the environment.
}
