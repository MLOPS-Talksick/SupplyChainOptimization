resource "google_service_account" "airflow_sa" {
  account_id   = "airflow-service-account"
  display_name = "Airflow VM Service Account"
}

resource "google_project_iam_member" "airflow_sa_artifact_registry" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.airflow_sa.email}"
}
