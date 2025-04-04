output "terraform_sa_private_key" {
  description = "Private key (in JSON format) for the bootstrap service account"
  value       = google_service_account_key.terraform_sa_key.private_key
  sensitive   = true
}