output "airflow_lb_ip" {
  description = "Static IP exposed by the LB for Airflow"
  value       = google_compute_global_forwarding_rule.airflow_http_forwarding_rule.ip_address
}