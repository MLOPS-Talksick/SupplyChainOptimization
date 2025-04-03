output "terraform_sa_private_key" {
  description = "The private key (in JSON format) for the Terraform service account"
  value       = google_service_account_key.terraform_sa_key.private_key
  sensitive   = true
}

output "terraform_sa_email" {
  description = "The email address of the Terraform service account"
  value       = google_service_account.terraform_sa.email
}