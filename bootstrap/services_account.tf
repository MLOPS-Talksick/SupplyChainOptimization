resource "google_service_account" "terraform_sa" {
  account_id   = "terraform-service-account"
  display_name = "Terraform Service Account"
  project      = var.project_id
}


resource "google_service_account_key" "terraform_sa_key" {
  service_account_id = google_service_account.terraform_sa.name
  key_algorithm      = "KEY_ALG_RSA_2048"
  private_key_type   = "TYPE_GOOGLE_CREDENTIALS_FILE"

  keepers = {
    force_new_key = timestamp()
  }
}


locals {
  roles = [
    "roles/artifactregistry.admin",                         // Artifact Registry Administrator
    "roles/artifactregistry.createOnPushRepoAdmin",     // Artifact Registry Create-on-Push Repository Administrator (custom role; adjust if needed)
    "roles/artifactregistry.reader",                          // Artifact Registry Reader
    "roles/artifactregistry.repoAdmin",                       // Artifact Registry Repository Administrator
    "roles/artifactregistry.writer",                          // Artifact Registry Writer
    "roles/artifact-admin",                                   // artifact-admin (custom role; adjust if needed)
    "roles/composer.admin",                                   // Cloud Composer Admin Role
    "roles/cloudfunctions.admin",                             // Cloud Functions Admin
    "roles/cloudsql.admin",                                   // Cloud SQL Admin
    "roles/compute.admin",                                    // Compute Admin
    "roles/compute.serviceAgent",                             // Compute Engine Service Agent
    "roles/compute.storageAdmin",                             // Compute Storage Admin
    "roles/iam.serviceAccountCreator",                        // Create Service Accounts
    "roles/resourcemanager.projectIamAdmin",                  // Project IAM Admin
    "roles/secretmanager.admin",                              // Secret Manager Admin
    "roles/servicenetworking.networksAdmin",                          // Service Networking Admin
    "roles/storage.admin"                                     // Storage Admin
  ]
}

resource "google_project_iam_member" "terraform_sa_roles" {
  for_each = toset(local.roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.terraform_sa.email}"
}

// Create the VM service account
resource "google_service_account" "vm_service_account" {
  account_id   = "vm-service-account"
  display_name = "VM Service Account for Inter-Service Communication"
  project      = var.project_id
}

locals {
  vm_service_account_roles = [
    "roles/composer.admin",                 // Cloud Composer Admin Role
    "roles/composer.serviceAgent",          // Cloud Composer API Service Agent
    "roles/compute.admin",                  // Compute Admin
    "roles/compute.instanceAdmin.v1",       // Compute Instance Admin (v1)
    "roles/compute.viewer",                 // Compute Viewer
    "roles/secretmanager.admin",            // Secret Manager Admin
    "roles/secretmanager.secretAccessor",   // Secret Manager Secret Accessor
    "roles/storage.admin",                  // Storage Admin
    "roles/storage.objectAdmin",            // Storage Object Admin
    "roles/storage.objectViewer"            // Storage Object Viewer
  ]
}

resource "google_project_iam_member" "vm_service_account_roles" {
  for_each = toset(local.vm_service_account_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.vm_service_account.email}"
}
