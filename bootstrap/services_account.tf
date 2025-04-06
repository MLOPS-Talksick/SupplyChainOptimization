######################################################
# Variables
######################################################

variable "create_service_account" {
  description = "Whether to create the Terraform service account. Set to false if it already exists."
  type        = bool
  default     = true
}

variable "create_vm_service_account" {
  description = "Whether to create the VM service account. Set to false if it already exists."
  type        = bool
  default     = true
}

######################################################
# Terraform Service Account (Conditional)
######################################################

# Create the service account if desired.
resource "google_service_account" "terraform_sa" {
  count        = var.create_service_account ? 1 : 0
  account_id   = "terraform-service-account"
  display_name = "Terraform Service Account"
  project      = var.project_id
}

# Look up the existing service account if not creating it.
data "google_service_account" "existing_sa" {
  count      = var.create_service_account ? 0 : 1
  account_id = "terraform-service-account"
  project    = var.project_id
}

# Select the appropriate service account email.
locals {
  terraform_sa_email = var.create_service_account ? google_service_account.terraform_sa[0].email : data.google_service_account.existing_sa[0].email
}

######################################################
# JSON Key and Local Files for Terraform SA (Only When Creating)
######################################################

resource "google_service_account_key" "terraform_sa_key" {
  count              = var.create_service_account ? 1 : 0
  service_account_id = google_service_account.terraform_sa[0].email
  key_algorithm      = "KEY_ALG_RSA_2048"
  private_key_type   = "TYPE_GOOGLE_CREDENTIALS_FILE"
}

resource "local_file" "sa_key_file" {
  count    = var.create_service_account ? 1 : 0
  content  = base64decode(trimspace(google_service_account_key.terraform_sa_key[0].private_key))
  filename = "${path.module}/sa_key.json"
}

resource "local_file" "sa_email_file" {
  content  = local.terraform_sa_email
  filename = "${path.module}/sa_email.txt"
}

######################################################
# IAM Bindings for Terraform SA
######################################################

locals {
  roles = [
    "roles/artifactregistry.admin",
    "roles/artifactregistry.createOnPushRepoAdmin",
    "roles/artifactregistry.reader",
    "roles/artifactregistry.repoAdmin",
    "roles/artifactregistry.writer",
    "roles/composer.admin",
    "roles/cloudfunctions.admin",
    "roles/cloudsql.admin",
    "roles/compute.admin",
    "roles/compute.serviceAgent",
    "roles/compute.storageAdmin",
    "roles/iam.serviceAccountCreator",
    "roles/resourcemanager.projectIamAdmin",
    "roles/secretmanager.admin",
    "roles/servicenetworking.networksAdmin",
    "roles/storage.admin"
  ]
}

resource "google_project_iam_member" "terraform_sa_roles" {
  for_each = toset(local.roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${local.terraform_sa_email}"
}

######################################################
# VM Service Account (Conditional)
######################################################

# Create the VM service account if desired.
resource "google_service_account" "vm_service_account" {
  count        = var.create_vm_service_account ? 1 : 0
  account_id   = "vm-service-account"
  display_name = "VM Service Account for Inter-Service Communication"
  project      = var.project_id
}

# Look up the existing VM service account if not creating it.
data "google_service_account" "existing_vm_sa" {
  count      = var.create_vm_service_account ? 0 : 1
  account_id = "vm-service-account"
  project    = var.project_id
}

# Select the appropriate VM service account email.
locals {
  vm_sa_email = var.create_vm_service_account ? google_service_account.vm_service_account[0].email : data.google_service_account.existing_vm_sa[0].email
}

######################################################
# IAM Bindings for VM Service Account
######################################################

locals {
  vm_service_account_roles = [
    "roles/composer.admin",
    "roles/composer.serviceAgent",
    "roles/compute.admin",
    "roles/compute.instanceAdmin.v1",
    "roles/compute.viewer",
    "roles/secretmanager.admin",
    "roles/secretmanager.secretAccessor",
    "roles/storage.admin",
    "roles/storage.objectAdmin",
    "roles/storage.objectViewer"
  ]
}

resource "google_project_iam_member" "vm_service_account_roles" {
  for_each = toset(local.vm_service_account_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${local.vm_sa_email}"
}