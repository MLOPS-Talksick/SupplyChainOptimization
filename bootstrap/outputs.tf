output "terraform_sa_private_key" {
  description = "Full JSON credentials for the Terraform service account"
  value       = google_service_account_key.terraform_sa_key.private_key
  sensitive   = true
}