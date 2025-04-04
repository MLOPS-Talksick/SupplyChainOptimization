output "terraform_sa_private_key" {
  # google_service_account_key.my_sa_key.private_key is base64
  value     = base64decode(google_service_account_key.terraform_sa_key.private_key)
  sensitive = true
}