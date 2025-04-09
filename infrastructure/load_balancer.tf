resource "google_compute_health_check" "http_health_check" {
  name               = "airflow-health-check"
  project            = var.project_id
  check_interval_sec = 10
  timeout_sec        = 5
  http_health_check {
    port = 8080
  }
}

resource "google_compute_backend_service" "airflow_backend" {
  name         = "airflow-backend-service"
  project      = var.project_id
  port_name    = "http"
  protocol     = "HTTP"
  timeout_sec  = 300

  health_checks = [
    google_compute_health_check.http_health_check.self_link
  ]

  backend {
    # This references the instance group defined elsewhere.
    group = google_compute_region_instance_group_manager.airflow_mig.instance_group
  }
}


resource "google_compute_backend_service" "cloudrun_backend" {
  name                  = "cloudrun-backend"
  protocol              = "HTTP"
  port_name             = "http"
  load_balancing_scheme = "EXTERNAL"

  backend {
    group = google_compute_region_network_endpoint_group.cloudrun_neg.id
  }
}



resource "google_compute_url_map" "airflow_url_map" {
  name            = "airflow-url-map"
  default_service = google_compute_backend_service.airflow_backend.self_link

  host_rule {
    hosts        = ["*"]
    path_matcher = "routing-paths"
  }

  path_matcher {
    name            = "routing-paths"
    default_service = google_compute_backend_service.airflow_backend.self_link

    path_rule {
      paths   = ["/api/*"]
      service = google_compute_backend_service.cloudrun_backend.self_link
    }
  }
}

resource "google_compute_target_http_proxy" "airflow_http_proxy" {
  name    = "airflow-http-proxy"
  url_map = google_compute_url_map.airflow_url_map.self_link
}


resource "google_compute_global_forwarding_rule" "airflow_http_forwarding_rule" {
  name    = "airflow-global-forwarding-rule"
  project = var.project_id
  target  = google_compute_target_http_proxy.airflow_http_proxy.self_link
  port_range = "80"
}

locals {
  airflow_lb_ip = google_compute_global_forwarding_rule.airflow_http_forwarding_rule.ip_address
}
