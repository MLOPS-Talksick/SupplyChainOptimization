######################################################
# Create the Service Account
######################################################
resource "google_service_account" "terraform_sa" {
  account_id   = "terraform-service-account"
  display_name = "Terraform Service Account"
  project      = var.project_id
}

######################################################
# Create the JSON Key
######################################################
resource "google_service_account_key" "terraform_sa_key" {
  service_account_id = google_service_account.terraform_sa.email
  key_algorithm      = "KEY_ALG_RSA_2048"
  private_key_type   = "TYPE_GOOGLE_CREDENTIALS_FILE"

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [private_key, public_key_data]
  }
}

######################################################
# Save the JSON Key to a local file
######################################################
resource "local_file" "sa_key_file" {
  content  = base64decode(trimspace(google_service_account_key.terraform_sa_key.private_key))
  filename = "${path.module}/sa_key.json"
}

resource "local_file" "sa_email_file" {
  content  = google_service_account.terraform_sa.email
  filename = "${path.module}/sa_email.txt"
}


locals {
  roles = [
    "roles/run.admin",
    "roles/eventarc.admin",
    "roles/logging.logWriter",
    "roles/storage.admin",
    "roles/aiplatform.admin",
    "roles/artifactregistry.admin",                           // Artifact Registry Administrator
    "roles/artifactregistry.createOnPushRepoAdmin",           // Artifact Registry Create-on-Push Repository Administrator (custom role; adjust if needed)
    "roles/artifactregistry.reader",                          // Artifact Registry Reader
    "roles/artifactregistry.repoAdmin",                       // Artifact Registry Repository Administrator
    "roles/artifactregistry.writer",                          // Artifact Registry Writer
    # "roles/artifact-admin",                                 // artifact-admin (custom role; adjust if needed)
    "roles/composer.admin",                                   // Cloud Composer Admin Role
    "roles/cloudfunctions.admin",                             // Cloud Functions Admin
    "roles/cloudsql.admin",                                   // Cloud SQL Admin
    "roles/compute.admin",                                    // Compute Admin
    "roles/compute.serviceAgent",                             // Compute Engine Service Agent
    "roles/compute.storageAdmin",                             // Compute Storage Admin
    "roles/iam.serviceAccountCreator",                        // Create Service Accounts
    "roles/resourcemanager.projectIamAdmin",                  // Project IAM Admin
    "roles/secretmanager.admin",                              // Secret Manager Admin
    "roles/servicenetworking.networksAdmin",                  // Service Networking Admin
    "roles/storage.admin",                                    // Storage Admin
    "roles/vpcaccess.admin",                                   // VPC access
    "roles/run.invoker",
    "roles/cloudsql.client",
    "roles/cloudscheduler.admin",
    "roles/monitoring.metricWriter",
    "roles/monitoring.admin",
    "roles/pubsub.admin"
  ]
}

resource "google_project_iam_member" "terraform_sa_roles" {
  for_each = toset(local.roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.terraform_sa.email}"
}