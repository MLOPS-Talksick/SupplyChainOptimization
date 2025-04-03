# resource "google_service_account" "airflow_sa" {
#   account_id   = "airflow-service-account"
#   display_name = "Airflow VM Service Account"
# }

# resource "google_project_iam_member" "airflow_sa_artifact_registry" {
#   project = var.project_id
#   role    = "roles/artifactregistry.reader"
#   member  = "serviceAccount:${google_service_account.airflow_sa.email}"
# }


resource "google_service_account" "terraform_sa" {
  account_id   = "terraform-service-account"
  display_name = "Terraform Service Account"
}

locals {
  roles = [
    "roles/artifactregistry.admin",                         // Artifact Registry Administrator
    "roles/artifactregistry.createOnPushRepositoryAdmin",     // Artifact Registry Create-on-Push Repository Administrator (custom role; adjust if needed)
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
    "roles/servicenetworking.admin",                          // Service Networking Admin
    "roles/storage.admin"                                     // Storage Admin
  ]
}


resource "google_project_iam_member" "terraform_sa_roles" {
  for_each = toset(local.roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.terraform_sa.email}"
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

resource "google_service_account" "vm_service_account" {
  account_id   = "vm-service-account"
  display_name = "VM Service Account for Inter-Service Communication"
}

resource "google_project_iam_member" "vm_service_account_roles" {
  for_each = toset(local.vm_service_account_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.vm_service_account.email}"
}