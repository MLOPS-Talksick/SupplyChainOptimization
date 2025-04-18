resource "google_cloud_scheduler_job" "job" {
  name             = "lstm-health-check-job"
  description      = "Cloud Scheduler job that targets a Cloud Run HTTP function"
  schedule         = "0 8 * * 0"
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
    http_method = "GET"
    headers     = { "Content-Type" = "application/json" }
    body = base64encode(jsonencode({
            days = 7
          }))

    dynamic "oidc_token" {
      for_each = var.service_account_email != null ? [1] : []
      content {
        service_account_email = var.service_account_email
        audience              = google_cloud_run_v2_service.model_health_check.uri
      }
    }
  }
}
