resource "google_cloud_scheduler_job" "model_health_check_job" {
  name             = "lstm-health-check-job"
  description      = "Cloud Scheduler job that targets the health‑check Cloud Run service"
  schedule         = "0 6 * * 0"
  time_zone        = "America/New_York"
  project          = var.project_id
  region           = var.region
  attempt_deadline = "320s"

  retry_config {
    retry_count          = 3
    min_backoff_duration = "10s"
    max_backoff_duration = "300s"
    max_retry_duration   = "600s"
    max_doublings        = 5
  }

  http_target {
    uri         = "${google_cloud_run_v2_service.model_health_check.uri}/model/health"
    http_method = "POST"
    headers     = { "Content-Type" = "application/json" }

    oidc_token {
      service_account_email = var.service_account_email
      # For Cloud Run, audience must match the service URI
      audience              = google_cloud_run_v2_service.model_health_check.uri
    }
  }
}


resource "google_cloud_scheduler_job" "prediction_job" {
  name             = "prediction-job"
  description      = "Cloud Scheduler job that targets the predict endpoint"
  schedule         = "0 9 * * 0"
  time_zone        = "America/New_York"
  project          = var.project_id
  region           = var.region
  attempt_deadline = "320s"

  retry_config {
    retry_count          = 3
    min_backoff_duration = "10s"
    max_backoff_duration = "300s"
    max_retry_duration   = "600s"
    max_doublings        = 5
  }

  http_target {
    uri         = "${google_cloud_run_v2_service.backend.uri}/predict"
    http_method = "POST"

    # send {"days": 7} in the request body
    body = jsonencode({ "days" = 7 })

    headers = {
      "Content-Type" = "application/json"
    }

    oidc_token {
      service_account_email = var.service_account_email
      audience              = google_cloud_run_v2_service.backend.uri
    }
  }
}